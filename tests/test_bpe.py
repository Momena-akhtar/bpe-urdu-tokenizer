import random

import pytest
import regex

from urdu_bpe.bpe import (
    EOT,
    EOT_ID,
    FIRST_MERGE_ID,
    N_MERGES,
    PRETOKENIZE_PATTERN,
    VOCAB_SIZE,
    BPETokenizer,
    merge_ids,
)
from urdu_bpe.bpe_train import count_words_text, train_fast

GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

TRICKY = [
    "",
    " ",
    "  ",
    "\n",
    "\n\n\n",
    "\t \r\n",
    "a",
    "hello world",
    "don't stop",
    "😀",
    "👍🏽 ok 🇵🇰",
    "emoji 😀😀😀!",
    "اردو",
    " اردو ",
    "أُردُو",
    "ہو\u200cگئی",
    "ہو\u200dگئی",
    "\u200eاردو\u200f",
    "٢٠٢٤ ۲۰۲۴ 2024",
    "پاکستان، لاہور۔ کیا؟ ہاں!",
    "ایک iPhone اور Google",
    "Roman Urdu: mera naam hai",
    "line1\nلائن۲\n\nline3",
    "\ud800",
    "x\udfffy",
    "\x00\x01\x7f",
    "<|endoftext|>",
    "ab<|endoftext|>cd<|endoftext|>",
    "ǅ ß ﬁ é é",
    "كتابـي",
    "a" * 5000,
    "ا" * 3000 + " " + "ب" * 3000,
]


@pytest.fixture(scope="module")
def tok(tmp_path_factory) -> BPETokenizer:
    from conftest import make_text

    merges = train_fast(count_words_text(make_text(2, 60)), 400)
    assert len(merges) == 400
    return BPETokenizer(merges)


def test_layout_constants():
    assert (VOCAB_SIZE, N_MERGES, FIRST_MERGE_ID, EOT_ID) == (50_000, 49_743, 257, 256)
    t = BPETokenizer()
    assert t.vocab_size == 257 and t.vocab[:256] == [bytes([i]) for i in range(256)]
    assert t.vocab[EOT_ID] == EOT.encode()


@pytest.mark.parametrize("text", TRICKY)
def test_lossless(tok: BPETokenizer, text: str):
    for special in (True, False):
        assert tok.decode(tok.encode(text, special=special)) == text
    assert tok.decode(BPETokenizer().encode(text)) == text  # no merges: pure bytes


def test_lossless_fuzz(tok: BPETokenizer):
    rng = random.Random(0)
    alphabet = list(
        "ابپتٹثجچحخدڈذرڑزسشصضطظعغفقکگلمنںوہھءیے ۔،؟!.,0123456789٠١٢ \n\t\u200cَّaZé😀"
    ) + ["<|endoftext|>"]
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 80)))
        assert tok.decode(tok.encode(text)) == text


def test_urdu_words_stay_whole_where_gpt2_cuts_them():
    ours = regex.compile(PRETOKENIZE_PATTERN)
    gpt2 = regex.compile(GPT2_PATTERN)
    word = "أُردُو"
    assert ours.findall(word) == [word]
    assert len(gpt2.findall(word)) > 1  # the failure the assignment describes
    assert ours.findall("ہو\u200cگئی") == ["ہو\u200cگئی"]  # ZWNJ does not split a word
    assert ours.findall("کتاب۔ اردو،") == ["کتاب", "۔", " اردو", "،"]
    assert ours.findall("a  b") == ["a", " ", " b"]  # last space attaches to the next word
    assert ours.findall("year 2024!") == ["year", " 2024", "!"]


def test_special_token_never_split_or_merged(tok: BPETokenizer):
    assert tok.encode("ab<|endoftext|>cd").count(EOT_ID) == 1
    assert tok.encode(EOT) == [EOT_ID]
    assert EOT_ID not in {x for pair in tok.merges for x in pair}
    literal = tok.encode(EOT, special=False)
    assert EOT_ID not in literal and tok.decode(literal) == EOT
    assert tok.decode([EOT_ID]) == EOT


def test_merge_ids_scans_left_to_right():
    assert merge_ids([1, 1, 1], (1, 1), 9) == [9, 1]
    assert merge_ids([1, 1, 1, 1, 1], (1, 1), 9) == [9, 9, 1]
    assert merge_ids([], (1, 2), 9) == []


def test_encoding_applies_merges_in_learned_order():
    t = BPETokenizer(train_fast(count_words_text("aaabdaaabac"), 3))
    assert t.encode("aaabdaaabac") == [259, 100, 259, 97, 99]  # aaab d aaab a c
    assert t.token_bytes(259) == b"aaab"


def test_invalid_ids_and_partial_bytes(tok: BPETokenizer):
    for bad in (-1, tok.vocab_size, 10**6):
        with pytest.raises(ValueError):
            tok.decode([bad])
    assert tok.decode([0xD8]) == "�"  # half of a two-byte letter: replaced, not an exception
    assert not tok.is_valid_utf8(0xD8) and tok.is_valid_utf8(ord("a"))


def test_bad_merge_tables_rejected():
    with pytest.raises(ValueError):
        BPETokenizer([(300, 1)])  # refers to a token that does not exist yet
    with pytest.raises(ValueError):
        BPETokenizer([(EOT_ID, 1)])  # the special token cannot take part in a merge


def test_save_load_roundtrip(tok: BPETokenizer, tmp_path):
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = BPETokenizer.load(path, expected_merges=400)
    assert (
        loaded.merges == tok.merges and loaded.vocab == tok.vocab and loaded.pattern == tok.pattern
    )
    sample = "پاکستان 2024 😀 <|endoftext|> hello"
    assert loaded.encode(sample) == tok.encode(sample)
    with pytest.raises(ValueError):
        BPETokenizer.load(path, expected_merges=49_743)
    tok.save_readable(tmp_path / "merges.txt")
    assert len((tmp_path / "merges.txt").read_text().splitlines()) == 400
