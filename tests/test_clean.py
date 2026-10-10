import hashlib
import json
import random
from pathlib import Path

from conftest import URDU_WORDS
from urdu_bpe.clean.filters import EOT, classify_line, clean_lines, is_repetitive
from urdu_bpe.clean.pipeline import CleanConfig, run


def test_line_classification_keeps_code_mixing():
    assert classify_line("یہ ایک اردو جملہ ہے") is None
    assert classify_line("یہ ایک iPhone کا نیا ماڈل ہے") is None  # natural code-mixing survives
    assert classify_line("This line is only English") == "no_urdu_letters"
    assert classify_line("12345 ---- ٭٭٭") == "no_urdu_letters"
    assert classify_line("ا") == "too_few_urdu_letters"
    assert (
        classify_line("ایک Breaking News Update Live Coverage From Everywhere Today")
        == "mostly_not_urdu"
    )
    assert classify_line("خراب � متن") == "replacement_character"


def test_clean_lines_strips_urls_markers_and_counts_bytes():
    counts: dict[str, list[int]] = {}
    text = f"خبر دیکھیں https://example.com/x?a=1 اور مزید\nonly english here\n{EOT}\nخبر{EOT}ختم\n\nرابطہ a.b@mail.com کریں"
    out = clean_lines(text, counts)
    assert out == ["خبر دیکھیں اور مزید", "خبر ختم", "رابطہ کریں"]
    assert counts["no_urdu_letters"][0] == 2  # the English line and the bare marker line
    assert counts["url_or_email_removed"][0] == 2 and counts["endoftext_marker"][1] == 2 * len(EOT)


def test_is_repetitive():
    assert is_repetitive(["الف"] * 12)
    assert not is_repetitive([f"سطر {i}" for i in range(12)])
    assert not is_repetitive(["الف"] * 5)  # too short to judge


def _doc(rng: random.Random, n_lines: int = 8) -> str:
    return "\n".join(" ".join(rng.choices(URDU_WORDS, k=14)) + "۔" for _ in range(n_lines))


def _write_raw(raw: Path, name: str, genre: str, docs: list[str]) -> None:
    folder = raw / name
    folder.mkdir(parents=True)
    with (folder / "docs.jsonl").open("w", encoding="utf-8") as fh:
        for i, text in enumerate(docs):
            fh.write(
                json.dumps(
                    {"id": f"{name}-{i}", "source": name, "genre": genre, "text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _build(tmp_path: Path) -> tuple[Path, Path]:
    rng = random.Random(7)
    boiler = "مزید پڑھیں خبریں سب سے پہلے یہاں"
    news = [_doc(rng) + "\n" + boiler for _ in range(300)]
    news.append(news[5])  # exact duplicate document
    news.append("ٹھیک ہے")  # too short
    news.append("كتاب\u200cـ ٢٠٢٤ " + _doc(rng))  # needs normalising
    wiki = [_doc(rng) for _ in range(60)]  # < 200 docs: goes entirely to train
    wiki.append(news[9].split("\n")[0] + "\n" + _doc(rng))  # shares a paragraph (<120 chars: kept)
    raw = tmp_path / "raw"
    _write_raw(raw, "ary_urdu", "news", news)
    _write_raw(raw, "urdu_wikipedia", "encyclopedia", wiki)
    return raw, tmp_path / "clean"


def test_pipeline_end_to_end(tmp_path: Path):
    raw, out = _build(tmp_path)
    report = run(raw, out, CleanConfig(seed=3), workers=2)
    parts = {p: (out / f"{p}.txt").read_text(encoding="utf-8") for p in ("train", "val", "test")}

    # structure: a marker after every document, none inside one
    for text in parts.values():
        assert text.endswith(EOT)
        assert all(EOT not in doc for doc in text.split(EOT))
    docs = {p: len(t.split(EOT)) - 1 for p, t in parts.items()}
    assert docs["val"] >= 1 and docs["test"] >= 1 and docs["train"] > 0.9 * sum(docs.values())
    assert report["splits"]["train"]["docs"] == docs["train"]

    # the wiki source is too small for eval splits: only news is in val/test
    assert set(report["splits"]["val"]["by_source"]) == {"ary_urdu"}

    # cleaning effects
    everything = "".join(parts.values())
    assert "مزید پڑھیں خبریں سب سے پہلے یہاں" not in everything  # boilerplate (300 docs)
    assert "\u200c" not in everything and "٢" not in everything and "ك" not in everything
    assert "کتاب 2024" in everything
    drops = report["stage2"]["ary_urdu"]["drops"]
    assert drops["duplicate_doc"][0] == 1 and drops["too_short_doc"][0] >= 1
    assert drops["boilerplate_line"][0] >= 300
    assert any(s["step"] == "dedup/filter: boilerplate_line" for s in report["steps"])

    # split is by document: no document id in two splits
    manifest = json.loads((out / "split_manifest.json").read_text())
    assert not set(manifest["val_ids"]) & set(manifest["test_ids"])


def test_split_is_deterministic_and_seed_dependent(tmp_path: Path):
    def digest(root: Path, seed: int) -> str:
        raw, out = _build(root)
        run(raw, out, CleanConfig(seed=seed), workers=1)
        return hashlib.sha256(
            b"".join((out / f"{p}.txt").read_bytes() for p in ("train", "val", "test"))
        ).hexdigest()

    a = digest(tmp_path / "a", 5)
    assert a == digest(tmp_path / "b", 5)
    assert a != digest(tmp_path / "c", 6)


def test_ocr_line_filter_keeps_known_words_and_drops_garbage():
    from urdu_bpe.clean.filters import OcrLineFilter

    known = frozenset(["اس", "طرح", "ہفتوں", "مہینوں", "اور", "برسوں", "کا", "شمار", "بھی", "ہے"])
    f = OcrLineFilter(known)
    assert f.reason("اس طرح ہفتوں مہینوں اور برسوں کا شمار") is None
    assert (
        f.reason("اس طرح ہفتوں میوں اور برسوں کا شار") == "ocr_unknown_words"
    )  # 2 of 8 words unknown
    assert f.reason("سنا وربا قالت جن بخاب با نکیا") == "ocr_unknown_words"
    assert f.reason("اس طرح") == "ocr_line_too_short"
    shred = OcrLineFilter(
        frozenset(["میا", "ےکی", "و", "سے", "دی", "یت", "وا", "ا", "اور", "کا", "ہے"])
    )
    assert (
        shred.reason("میا ےکی و سے دی یت وا سے") == "ocr_fragmented_words"
    )  # all known, all fragments
    assert shred.reason("و ا و ا و ا و ا") == "ocr_fragmented_words"
    prose = frozenset(["یہ", "ایک", "بہت", "اچھی", "کتاب", "ہے", "جسے", "مصنف", "نے", "لکھا"])
    assert (
        OcrLineFilter(prose).reason("یہ ایک بہت اچھی کتاب ہے جسے مصنف نے لکھا 7 0 5 4 3")
        == "ocr_digit_noise"
    )
    assert OcrLineFilter(prose).reason("یہ ایک بہت اچھی کتاب ہے جسے مصنف نے لکھا") is None


def test_pipeline_filters_ocr_source_lines(tmp_path: Path):
    rng = random.Random(9)
    good = lambda: " ".join(rng.choices(URDU_WORDS, k=14)) + "۔"  # noqa: E731
    garbage = "سنا وربا قالت جن بخاب با نکیا وا کے کو کا ر ضز اع"
    books = ["\n".join([good() for _ in range(6)] + [garbage, garbage + " ایک"]) for _ in range(12)]
    news = ["\n".join(good() for _ in range(8)) for _ in range(30)]
    raw = tmp_path / "raw"
    _write_raw(raw, "archive_org_books", "books", books)
    _write_raw(raw, "ary_urdu", "news", news)
    report = run(
        raw,
        tmp_path / "clean",
        CleanConfig(seed=1, ocr_min_word_count=1, exclude_sources=()),
        workers=1,
    )
    drops = report["stage2"]["archive_org_books"]["drops"]
    survivors = (
        12 - report["stage1"]["archive_org_books"]["doc_drops"].get("ocr_noise_doc", [0, 0])[0]
    )
    assert (
        survivors > 0 and drops["ocr_unknown_words"][0] == 2 * survivors
    )  # both garbage lines per book
    text = "".join(
        (tmp_path / "clean" / f"{p}.txt").read_text(encoding="utf-8")
        for p in ("train", "val", "test")
    )
    assert "قالت" not in text


def test_excluded_source_stays_out_of_the_corpus(tmp_path: Path):
    raw, out = _build(tmp_path)
    run(raw, out, CleanConfig(seed=3, exclude_sources=("urdu_wikipedia",)), workers=1)
    report = json.loads((out / "cleaning_stats.json").read_text())
    assert (
        report["excluded_sources"] == ["urdu_wikipedia"]
        and "urdu_wikipedia" not in report["stage2"]
    )
