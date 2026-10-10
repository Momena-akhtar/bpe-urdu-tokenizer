"""Generate DATA_CARD.md from raw metadata, cleaning_stats.json and the final split files."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import regex

from urdu_bpe.clean.filters import EOT
from urdu_bpe.collect import registry
from urdu_bpe.collect.record import iter_documents

_WORD = regex.compile(r"[\p{L}\p{M}\p{N}]+")

STATIC_BIASES = [
    "Register: written, formal Urdu (news, essays, encyclopedia). Conversational text, social media, "
    "SMS-style writing and Roman Urdu are almost absent.",
    "Place and topics: the scraped outlets are Pakistani (Karachi, Islamabad, Peshawar, Quetta), so "
    "politics, crime, cricket, prices and the Iran-Israel conflict are over-represented; Indian "
    "Urdu, older literary Urdu and dialects are under-represented.",
    "CC-100 is only the first slice of its file, so it reflects whichever crawl shards come first, "
    "and it contains pages that also appear in the scraped news (removed only when identical).",
    "Deduplication is exact (documents, lines, long paragraphs). Near-duplicates, such as the same wire "
    "story re-worded by two outlets, remain, so val/test are not guaranteed leak-free against train.",
    "Normalisation choices alter the text: ZWNJ is deleted, digits become ASCII, Quranic annotation "
    "marks are dropped, so the tokenizer never sees those variants.",
    "Books come from Wikisource and Wikibooks only: the Internet Archive OCR books (40 MB collected) "
    "were excluded because even after line-level OCR filtering the surviving text stayed noisy, so "
    "long-form literary prose and Urdu poetry are thin.",
    "Scraped publisher text is copyrighted; it is used for coursework only and is not redistributed.",
]


def count_split(path: Path) -> dict:
    docs = words = 0
    vocab: dict[str, int] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            docs += line.count(EOT)
            for w in _WORD.findall(line.replace(EOT, " ")):
                words += 1
                vocab[w] = vocab.get(w, 0) + 1
    return {"bytes": path.stat().st_size, "docs": docs, "words": words, "vocab": vocab}


def source_periods(raw_dir: Path, names: list[str]) -> dict[str, tuple[str, str, str]]:
    """(earliest, median, latest) publication month of each source whose documents carry a date."""
    periods: dict[str, tuple[str, str, str]] = {}
    for name in names:
        path = raw_dir / name / "docs.jsonl"
        if not path.exists():
            continue
        months = sorted(r["published"][:7] for r in iter_documents(path) if r.get("published"))
        if months:
            periods[name] = (months[0], months[len(months) // 2], months[-1])
    return periods


def build_card(clean_dir: Path, raw_dir: Path, *, today: str | None = None) -> str:
    stats = json.loads((clean_dir / "cleaning_stats.json").read_text())
    counts = {part: count_split(clean_dir / f"{part}.txt") for part in ("train", "val", "test")}
    all_vocab = set().union(*(c["vocab"].keys() for c in counts.values()))
    train_vocab = counts["train"]["vocab"]
    test_tokens = counts["test"]["words"]
    oov = sum(n for w, n in counts["test"]["vocab"].items() if w not in train_vocab)
    total_bytes = sum(c["bytes"] for c in counts.values())

    lines = [f"# Data card: Urdu BPE corpus ({today or date.today().isoformat()})", ""]
    lines += [
        "Cleaned Urdu text for training a byte-level BPE tokenizer and, in the next assignment, a "
        "GPT-style model. Built by `usage/01_collect.py` then `usage/02_clean.py` (fixed split seed "
        f"{stats['config']['seed']}). Documents are separated by `<|endoftext|>`; the split is by "
        "document, stratified by source, 98/1/1 by bytes.",
        "",
        "## Sources",
        "",
        "| source | genre | URL | licence | collected | raw MB | cleaned MB |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, (spec, _) in registry.FACTORIES.items():
        meta_path = raw_dir / name / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        cleaned = stats["stage2"].get(name, {}).get("bytes_out", 0)
        own = " \u2605" if spec.self_collected else ""
        kept = (
            "excluded (OCR noise)"
            if name in stats.get("excluded_sources", [])
            else f"{cleaned / 1e6:,.1f}"
        )
        lines.append(
            f"| {spec.title}{own} | {spec.genre} | {spec.homepage} | {spec.licence} | "
            f"{meta['first_collected'][:10]} | {meta['text_bytes'] / 1e6:,.1f} | {kept} |"
        )
    self_raw = sum(
        json.loads((raw_dir / n / "meta.json").read_text())["text_bytes"]
        for n in registry.SELF_COLLECTED
        if (raw_dir / n / "meta.json").exists()
    )
    lines += [
        "",
        f"\u2605 = self-collected by scraping or public APIs (robots.txt obeyed; request logs and robots.txt "
        f"snapshots in `evidence/` of each source): **{self_raw / 1e6:,.1f} MB** of extracted text.",
        "",
        "## Corpus totals",
        "",
        "| split | MB | documents | words |",
        "|---|---|---|---|",
    ]
    for part, c in counts.items():
        lines.append(
            f"| {part} | {c['bytes'] / 1e6:,.1f} ({100 * c['bytes'] / total_bytes:.2f}%) | "
            f"{c['docs']:,} | {c['words']:,} |"
        )
    lines += [
        f"| **total** | **{total_bytes / 1e6:,.1f}** | **{sum(c['docs'] for c in counts.values()):,}** | "
        f"**{sum(c['words'] for c in counts.values()):,}** |",
        "",
        f"Unique words: {len(all_vocab):,} overall, {len(train_vocab):,} in train. "
        f"{100 * oov / max(test_tokens, 1):.2f}% of test word tokens never occur in train. "
        "A word is a maximal run of letters, combining marks or digits.",
        "",
        "## Cleaning steps (in order; MB removed from the extracted text)",
        "",
        "| step | removed MB | docs / lines |",
        "|---|---|---|",
    ]
    for step in stats["steps"]:
        if not (step["bytes"] or step["docs"] or step["lines"]) and not step["note"]:
            continue  # steps that never fired (e.g. the OCR filters, source excluded)
        count = f"{step['docs']:,} docs" if step["docs"] else ""
        count = count or (f"{step['lines']:,} lines" if step["lines"] else "")
        lines.append(f"| {step['step']} | {step['bytes'] / 1e6:,.2f} | {count} |")
    lines += ["", "## Known gaps and biases", ""]
    train = stats["splits"]["train"]
    genre_share = sorted(train["bytes_by_genre"].items(), key=lambda kv: -kv[1])
    lines.append(
        "- Genre mix of train by bytes: "
        + ", ".join(f"{g} {100 * b / train['bytes']:.1f}%" for g, b in genre_share)
        + "."
    )
    top_source, top_bytes = max(train["bytes_by_source"].items(), key=lambda kv: kv[1])
    lines.append(
        f"- Largest single source: {top_source} at {100 * top_bytes / train['bytes']:.1f}% of train."
    )
    periods = source_periods(raw_dir, [n for n in registry.SELF_COLLECTED if n in stats["stage2"]])
    recent = {n: p for n, p in periods.items() if p[0] >= "2025-01"}
    spans = {n: p for n, p in periods.items() if p[0] < "2025-01"}
    lines.append(
        "- Time: "
        + (
            f"{len(recent)} of the {len(periods)} dated scraped sources ("
            + ", ".join(sorted(recent))
            + f") only reach back {min(p[0] for p in recent.values())} to {max(p[2] for p in recent.values())} "
            "because the sites were walked newest-first, so the news reflects one recent year of events. "
            if recent
            else ""
        )
        + "Sources spanning years: "
        + "; ".join(f"{n} {p[0]} to {p[2]} (median {p[1]})" for n, p in sorted(spans.items()))
        + "."
    )
    lines += [f"- {b}" for b in STATIC_BIASES]
    lines += ["", "## Considered and rejected", ""]
    lines.append("; ".join(f"**{name}**: {why}" for name, why in registry.REJECTED.items()) + ".")
    return "\n".join(lines) + "\n"
