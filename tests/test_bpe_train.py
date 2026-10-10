import random
from pathlib import Path

from conftest import make_text
from urdu_bpe.bpe import BPETokenizer
from urdu_bpe.bpe_train import (
    FastBPETrainer,
    count_words_file,
    count_words_text,
    load_word_counts,
    save_word_counts,
    train_fast,
    train_reference,
)

SANITY = "aaabdaaabac"
# aa=4; then (aa,a) ties (a,b) at 2 and wins as the greater pair of bytes; then (aaa,b)=2.
SANITY_MERGES = [(97, 97), (257, 97), (258, 98)]


def test_sanity_example_both_trainers():
    assert train_reference(SANITY, 3) == SANITY_MERGES
    assert train_fast(count_words_text(SANITY), 3) == SANITY_MERGES


def test_ties_break_to_the_greater_pair_of_bytes():
    # words "xy" and " xz": every pair occurs once; (x,z) = (b"x", b"z") is the greatest tuple
    text = "xy xz"
    assert train_reference(text, 1) == [(120, 122)]
    assert train_fast(count_words_text(text), 1) == [(120, 122)]
    # and between (b" ", b"x") and (b"x", b"y") the second is greater
    assert train_reference("xy", 1) == [(120, 121)]


def test_fast_equals_reference_on_synthetic_corpus():
    text = make_text(seed=3, n_docs=25)
    ref = train_reference(text, 250)
    fast = train_fast(count_words_text(text), 250)
    assert len(ref) == 250 and ref == fast


def test_both_stop_together_when_pairs_run_out():
    text = "ab ab abc"
    ref, fast = train_reference(text, 100), train_fast(count_words_text(text), 100)
    assert ref == fast and 0 < len(ref) < 100


def test_overlapping_pairs_agree():
    text = "aaaa aaaaa aaa" * 3 + " ababab abababa"
    assert train_reference(text, 30) == train_fast(count_words_text(text), 30)


def test_trained_tokenizer_matches_merge_state():
    text = make_text(seed=4, n_docs=10)
    counts = count_words_text(text)
    merges = train_fast(counts, 120)
    tok = BPETokenizer(merges)
    # no word can still contain a learned pair adjacent after encoding, and encoding is lossless
    for word in list(counts)[:200]:
        ids = tok.encode_ordinary(word.decode())
        assert tok.decode(ids) == word.decode()
        assert not any(p in tok._rank for p in zip(ids, ids[1:], strict=False))


def test_checkpoint_resume_is_exact():
    counts = count_words_text(make_text(seed=5, n_docs=20))
    full = FastBPETrainer(counts).train(150)
    resumed = FastBPETrainer(counts).train(150, forced=full[:60])
    assert resumed == full


def test_word_counts_exclude_marker_and_parallel_equals_serial(tmp_path: Path):
    text = make_text(seed=6, n_docs=30)
    path = tmp_path / "train.txt"
    path.write_text(text, encoding="utf-8")
    serial = count_words_text(text)
    assert b"<|endoftext|>" not in serial and not any(b"endoftext" in w for w in serial)
    parallel = count_words_file(path, workers=2, chunk_bytes=3000)
    assert parallel == serial
    limited = count_words_file(path, workers=1, chunk_bytes=3000, limit=3000)
    assert sum(limited.values()) < sum(serial.values())


def test_word_count_cache(tmp_path: Path):
    counts = count_words_text(make_text(seed=7, n_docs=5))
    cache = tmp_path / "wc.pkl"
    save_word_counts(counts, cache, {"size": 1})
    assert load_word_counts(cache, {"size": 1}) == counts
    assert load_word_counts(cache, {"size": 2}) is None
    assert load_word_counts(tmp_path / "missing.pkl", {"size": 1}) is None


def test_random_unicode_fast_equals_reference():
    rng = random.Random(11)
    alphabet = list("ابپتٹجچحخدڈرزسشعغفقکگلمنوہیے ۔،\n\t0123456789aeiou😀َ‌")
    text = "".join(rng.choice(alphabet) for _ in range(4000))
    assert train_reference(text, 80) == train_fast(count_words_text(text), 80)
