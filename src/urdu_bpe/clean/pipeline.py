"""The cleaning pipeline: extract -> normalise -> filter -> deduplicate -> split.

Stage 1 (parallel, one process per source): normalise every document, apply the line filters and
write ``stage1/<source>.jsonl`` plus the 64-bit hashes of each document's distinct lines.
Stage 2 (one process, sources in priority order): drop exact duplicate documents, boilerplate lines
(identical lines seen in many documents) and repeated long paragraphs, then repetitive/short
documents. Stage 3: split *by document* into train/val/test with a fixed seed and write the files.

Every step records how many documents, lines and bytes it removed (``cleaning_stats.json``); the
data card is generated from those numbers.
"""

from __future__ import annotations

import json
import logging
import random
from array import array
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from urdu_bpe.clean.filters import (
    EOT,
    OCR_SOURCES,
    WORD,
    OcrLineFilter,
    clean_lines,
    fails_ocr_check,
    is_repetitive,
    line_hash,
)
from urdu_bpe.normalize import normalize, rule_hits

log = logging.getLogger("clean")

# Earlier sources win when the same document/paragraph occurs twice: curated text first, scraped
# originals before the CC-100 web crawl, which re-hosts a lot of the same news.
SOURCE_ORDER = [
    "urdu_wikipedia",
    "urdu_wikisource",
    "urdu_wikibooks",
    "archive_org_books",
    "aik_rozan",
    "daily_ausaf",
    "mashriq_tv",
    "ary_urdu",
    "hum_news",
    "ummat",
    "daily_qudrat",
    "express_urdu",
    "cc100_urdu",
]


@dataclass(frozen=True)
class CleanConfig:
    seed: int = 2026
    val_fraction: float = 0.01
    test_fraction: float = 0.01
    min_doc_chars: int = 200
    boilerplate_min_docs: int = 30  # a line of >= boilerplate_min_chars repeated in this many docs
    boilerplate_min_chars: int = 8
    boilerplate_short_min_docs: int = 300  # shorter lines need far more repeats to count
    long_line_chars: int = 120  # paragraphs at least this long are deduplicated across documents
    min_docs_for_eval_split: int = 200  # smaller sources go entirely to train
    ocr_min_known_share: float = 0.85  # OCR lines: share of words that the clean corpus also uses
    ocr_min_word_count: int = 5  # a word counts as "known" if the trusted corpus has it this often
    ocr_vocab_docs_per_source: int = (
        20_000  # documents per trusted source used to build that vocabulary
    )
    # Collected but kept out of the corpus: the OCR filters leave ~4% of the Internet Archive books
    # and what survives is still noisy, so it would add junk merges for almost no text.
    exclude_sources: tuple[str, ...] = ("archive_org_books",)


def ordered_sources(available: list[str]) -> list[str]:
    known = [s for s in SOURCE_ORDER if s in available]
    return known + sorted(set(available) - set(known))


def _nbytes(text: str) -> int:
    return len(text.encode("utf-8", "surrogatepass"))


def _bump(table: dict[str, list[int]], key: str, docs_or_lines: int, nbytes: int) -> None:
    entry = table.setdefault(key, [0, 0])
    entry[0] += docs_or_lines
    entry[1] += nbytes


# stage 1
def stage1_source(raw_dir: str, out_dir: str, name: str, force: bool = False) -> dict:
    raw = Path(raw_dir) / name / "docs.jsonl"
    stage = Path(out_dir) / "stage1"
    stage.mkdir(parents=True, exist_ok=True)
    out, stats_path, hashes_path = (
        stage / f"{name}.jsonl",
        stage / f"{name}.json",
        stage / f"{name}.npy",
    )
    st = raw.stat()
    signature = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if not force and stats_path.exists() and out.exists() and hashes_path.exists():
        cached = json.loads(stats_path.read_text())
        if cached.get("signature") == signature:
            log.info("%s: stage 1 up to date, skipped", name)
            return cached

    stats: dict = {
        "source": name,
        "docs_in": 0,
        "bytes_in": 0,
        "bytes_after_normalize": 0,
        "rule_hits": Counter(),
        "line_filters": {},
        "doc_drops": {},
    }
    hashes, lengths = array("Q"), array("H")
    docs_out = bytes_out = 0
    with raw.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            text = rec["text"]
            stats["docs_in"] += 1
            stats["bytes_in"] += _nbytes(text)
            stats["rule_hits"].update(rule_hits(text))
            text = normalize(text)
            stats["bytes_after_normalize"] += _nbytes(text)
            lines = clean_lines(text, stats["line_filters"])
            text = "\n".join(lines)
            if fails_ocr_check(name, text):
                _bump(stats["doc_drops"], "ocr_noise_doc", 1, _nbytes(text))
                continue
            if not text:
                _bump(stats["doc_drops"], "empty_after_filters", 1, 0)
                continue
            for h, ln in {line_hash(x): len(x) for x in lines}.items():
                hashes.append(h)
                lengths.append(min(ln, 65535))
            fout.write(
                json.dumps(
                    {
                        "id": f"{name}:{rec['id']}",
                        "source": name,
                        "genre": rec["genre"],
                        "text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            docs_out += 1
            bytes_out += _nbytes(text)
    np.save(
        hashes_path,
        np.stack(
            [
                np.frombuffer(hashes, dtype=np.uint64),
                np.frombuffer(lengths, dtype=np.uint16).astype(np.uint64),
            ]
        ),
    )
    stats |= {
        "docs_out": docs_out,
        "bytes_out": bytes_out,
        "signature": signature,
        "rule_hits": dict(stats["rule_hits"]),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    log.info(
        "%s: %d -> %d docs, %.1f -> %.1f MB",
        name,
        stats["docs_in"],
        docs_out,
        stats["bytes_in"] / 1e6,
        bytes_out / 1e6,
    )
    return stats


# stage 2
def boilerplate_hashes(stage_dir: Path, names: list[str], cfg: CleanConfig) -> set[int]:
    """Hashes of lines repeated across many documents (menus, share buttons, sign-offs)."""
    arrays = [np.load(stage_dir / f"{n}.npy") for n in names]
    hashes = np.concatenate([a[0] for a in arrays])
    lengths = np.concatenate([a[1] for a in arrays])
    unique, first, counts = np.unique(hashes, return_index=True, return_counts=True)
    ulen = lengths[first]
    mask = ((counts >= cfg.boilerplate_min_docs) & (ulen >= cfg.boilerplate_min_chars)) | (
        counts >= cfg.boilerplate_short_min_docs
    )
    return set(unique[mask].tolist())


def trusted_vocabulary(stage_dir: Path, names: list[str], cfg: CleanConfig) -> frozenset[str]:
    """Words used at least ``ocr_min_word_count`` times in the first documents of every non-OCR source."""
    counts: Counter[str] = Counter()
    for name in names:
        if name in OCR_SOURCES:
            continue
        with (stage_dir / f"{name}.jsonl").open(encoding="utf-8") as fh:
            for i, raw_line in enumerate(fh):
                if i >= cfg.ocr_vocab_docs_per_source:
                    break
                counts.update(WORD.findall(json.loads(raw_line)["text"]))
    return frozenset(w for w, n in counts.items() if n >= cfg.ocr_min_word_count)


def stage2(out_dir: Path, names: list[str], cfg: CleanConfig) -> tuple[list[dict], dict]:
    """Deduplicate. Returns (index of kept documents with file offsets, per-source stats)."""
    stage = out_dir / "stage1"
    boilerplate = boilerplate_hashes(stage, names, cfg)
    log.info("%d boilerplate line hashes", len(boilerplate))
    ocr_filter: OcrLineFilter | None = None
    if any(n in OCR_SOURCES for n in names):
        known = trusted_vocabulary(stage, names, cfg)
        log.info("%d trusted words for the OCR line filter", len(known))
        ocr_filter = OcrLineFilter(known, min_known_share=cfg.ocr_min_known_share)
    seen_docs: set[int] = set()
    seen_long: set[int] = set()
    index: list[dict] = []
    per_source: dict[str, dict] = {}
    with (out_dir / "stage2.jsonl").open("w", encoding="utf-8") as fout:
        for name in names:
            drops: dict[str, list[int]] = {}
            kept_docs = kept_bytes = 0
            with (stage / f"{name}.jsonl").open(encoding="utf-8") as fin:
                for raw_line in fin:
                    rec = json.loads(raw_line)
                    text = rec["text"]
                    doc_hash = line_hash(text)
                    if doc_hash in seen_docs:
                        _bump(drops, "duplicate_doc", 1, _nbytes(text))
                        continue
                    kept: list[str] = []
                    pending: set[int] = set()
                    for line in text.split("\n"):
                        if ocr_filter is not None and name in OCR_SOURCES:
                            reason = ocr_filter.reason(line)
                            if reason:
                                _bump(drops, reason, 1, _nbytes(line))
                                continue
                        h = line_hash(line)
                        if h in boilerplate:
                            _bump(drops, "boilerplate_line", 1, _nbytes(line))
                        elif len(line) >= cfg.long_line_chars and (h in seen_long or h in pending):
                            _bump(drops, "duplicate_paragraph", 1, _nbytes(line))
                        else:
                            if len(line) >= cfg.long_line_chars:
                                pending.add(h)
                            kept.append(line)
                    body = "\n".join(kept)
                    if is_repetitive(kept):
                        _bump(drops, "repetitive_doc", 1, _nbytes(body))
                    elif len(body) < cfg.min_doc_chars:
                        _bump(drops, "too_short_doc", 1, _nbytes(body))
                    else:
                        seen_docs.add(doc_hash)
                        seen_long |= pending
                        offset = fout.tell()
                        fout.write(json.dumps({**rec, "text": body}, ensure_ascii=False) + "\n")
                        nbytes = _nbytes(body)
                        index.append(
                            {
                                "id": rec["id"],
                                "source": name,
                                "genre": rec["genre"],
                                "bytes": nbytes,
                                "offset": offset,
                            }
                        )
                        kept_docs += 1
                        kept_bytes += nbytes
            per_source[name] = {"drops": drops, "docs_out": kept_docs, "bytes_out": kept_bytes}
            log.info("%s: %d docs, %.1f MB after dedup", name, kept_docs, kept_bytes / 1e6)
    return index, per_source


# stage 3
def assign_splits(index: list[dict], cfg: CleanConfig) -> dict[str, list[int]]:
    """Split by document, stratified by source so val/test contain every genre.

    Per source, documents are shuffled with a seed derived from (seed, source) and handed to test,
    then val, until each holds its fraction of that source's *bytes*; the rest is train. Sources with
    too few documents to spare any go entirely to train.
    """
    by_source: dict[str, list[int]] = {}
    for i, doc in enumerate(index):
        by_source.setdefault(doc["source"], []).append(i)
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for source, members in by_source.items():
        if len(members) < cfg.min_docs_for_eval_split:
            splits["train"] += members
            continue
        order = members[:]
        random.Random(f"{cfg.seed}:{source}").shuffle(order)
        total = sum(index[i]["bytes"] for i in order)
        test_quota, val_quota = cfg.test_fraction * total, cfg.val_fraction * total
        taken = {"test": 0, "val": 0}
        for i in order:
            for part, quota in (("test", test_quota), ("val", val_quota)):
                if taken[part] < quota:
                    splits[part].append(i)
                    taken[part] += index[i]["bytes"]
                    break
            else:
                splits["train"].append(i)
    return splits


def read_stage2(path: Path, offsets: list[int]) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for offset in offsets:
            fh.seek(offset)
            yield json.loads(fh.readline())


def write_splits(
    out_dir: Path, index: list[dict], splits: dict[str, list[int]], cfg: CleanConfig
) -> dict:
    summary: dict[str, dict] = {}
    for part, members in splits.items():
        order = members[:]
        random.Random(f"{cfg.seed}:order:{part}").shuffle(order)  # train is not grouped by source
        nbytes = 0
        with (out_dir / f"{part}.txt").open("w", encoding="utf-8", newline="\n") as fh:
            for rec in read_stage2(out_dir / "stage2.jsonl", [index[i]["offset"] for i in order]):
                fh.write(rec["text"] + EOT)  # no separator: the marker itself is the boundary
                nbytes += _nbytes(rec["text"]) + len(EOT)
        summary[part] = {
            "docs": len(order),
            "bytes": nbytes,
            "by_source": dict(Counter(index[i]["source"] for i in order)),
            "bytes_by_genre": _bytes_by(index, order, "genre"),
            "bytes_by_source": _bytes_by(index, order, "source"),
        }
    manifest = {
        "seed": cfg.seed,
        "val_ids": [index[i]["id"] for i in splits["val"]],
        "test_ids": [index[i]["id"] for i in splits["test"]],
        "train_docs": len(splits["train"]),
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
    return summary


def _bytes_by(index: list[dict], members: list[int], key: str) -> dict[str, int]:
    out: Counter[str] = Counter()
    for i in members:
        out[index[i][key]] += index[i]["bytes"]
    return dict(out)


# driver
def summarize_steps(stage1: dict[str, dict], stage2_stats: dict[str, dict]) -> list[dict]:
    """The data card's cleaning-step table: what each step removed, in order."""

    def total(rows: list[dict], key: str, reason: str, col: int) -> int:
        return sum(r.get(key, {}).get(reason, [0, 0])[col] for r in rows)

    s1, s2 = list(stage1.values()), list(stage2_stats.values())
    steps = [
        {
            "step": "extract (collectors)",
            "docs": 0,
            "lines": 0,
            "bytes": 0,
            "note": f"{sum(r['docs_in'] for r in s1):,} documents, {sum(r['bytes_in'] for r in s1) / 1e6:,.1f} MB extracted",
        },
        {
            "step": "normalise (normalize.py)",
            "docs": 0,
            "lines": 0,
            "bytes": sum(r["bytes_in"] - r["bytes_after_normalize"] for r in s1),
            "note": "net byte change: ZWNJ/tatweel/invisible removed, digits and letters remapped",
        },
    ]
    for reason in sorted({k for r in s1 for k in r["line_filters"]}):
        steps.append(
            {
                "step": f"line filter: {reason}",
                "docs": 0,
                "lines": total(s1, "line_filters", reason, 0),
                "bytes": total(s1, "line_filters", reason, 1),
                "note": "",
            }
        )
    steps.append(
        {
            "step": "document filter: ocr_noise_doc",
            "docs": total(s1, "doc_drops", "ocr_noise_doc", 0),
            "lines": 0,
            "bytes": total(s1, "doc_drops", "ocr_noise_doc", 1),
            "note": "",
        }
    )
    for reason in (
        "ocr_unknown_words",
        "ocr_fragmented_words",
        "ocr_digit_noise",
        "ocr_line_too_short",
        "duplicate_doc",
        "boilerplate_line",
        "duplicate_paragraph",
        "repetitive_doc",
        "too_short_doc",
    ):
        steps.append(
            {
                "step": f"dedup/filter: {reason}",
                "docs": total(s2, "drops", reason, 0) if reason.endswith("doc") else 0,
                "lines": total(s2, "drops", reason, 0) if not reason.endswith("doc") else 0,
                "bytes": total(s2, "drops", reason, 1),
                "note": "",
            }
        )
    return steps


def run(
    raw_dir: Path, out_dir: Path, cfg: CleanConfig, *, workers: int = 4, force: bool = False
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    available = [p.parent.name for p in raw_dir.glob("*/docs.jsonl")]
    excluded = [n for n in available if n in cfg.exclude_sources]
    names = ordered_sources([n for n in available if n not in cfg.exclude_sources])
    if excluded:
        log.info("excluded from the corpus: %s", excluded)
    log.info("sources: %s", names)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(stage1_source, str(raw_dir), str(out_dir), n, force) for n in names]
        stage1 = {n: f.result() for n, f in zip(names, futures, strict=True)}
    index, stage2_stats = stage2(out_dir, names, cfg)
    splits = assign_splits(index, cfg)
    summary = write_splits(out_dir, index, splits, cfg)
    report = {
        "config": asdict(cfg),
        "excluded_sources": excluded,
        "stage1": stage1,
        "stage2": stage2_stats,
        "steps": summarize_steps(stage1, stage2_stats),
        "splits": summary,
    }
    (out_dir / "cleaning_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report
