"""Byte-level BPE tokenizer for Urdu, written from scratch (standard library + ``regex``).

Vocabulary layout (exactly 50,000 tokens)::

    0 .. 255        the 256 byte values (an Urdu letter is two bytes in UTF-8)
    256             <|endoftext|>, never split and never part of a merge
    257 .. 49,999   49,743 merges, in the order they were learned

Training lives in ``bpe_train.py``; this module is the tokenizer itself: pre-tokenisation, encoding,
decoding, saving and loading. Normalisation is deliberately *not* done here (see ``normalize.py``),
so ``decode(encode(text)) == text`` for every ``str``, including emoji, English, newlines, the empty
string and even lone surrogates (bytes are produced with ``surrogatepass``).

Pre-tokenisation
----------------
BPE never merges across a pre-token boundary, so the pre-tokenizer decides what a "word" is. GPT-2's
pattern treats combining marks as punctuation (``\\p{L}+`` stops at a diacritic), so it cuts the word
أُردُو into five pieces. Ours keeps a word whole::

    (?i:'s|'t|'re|'ve|'m|'ll|'d)             English contractions, for code-mixed text
    | ?[\\p{L}\\p{M}\\u200c\\u200d]+           a word: letters, combining marks, ZWNJ, ZWJ
    | ?\\p{N}+                                a number (any script: 2024, ۲۰۲۴, ٢٠٢٤)
    | ?[^\\s\\p{L}\\p{M}\\p{N}\\u200c\\u200d]+   a run of punctuation, symbols or emoji (،  ۔  ؟  « »)
    |\\s+(?!\\S)|\\s+                          whitespace; the last space stays with the next word

* ``\\p{M}`` (Mn/Mc/Me) is what keeps zabar, zer, pesh, shadda, tanween, dagger alef and the hamza
  marks inside their word.
* ZWNJ/ZWJ are *inside* the word class, so a stray joiner can never split a word in two. Normalisation
  removes them from the training text anyway, but the tokenizer must survive un-normalised input.
* As in GPT-2 a leading space belongs to the word that follows, so " کتاب" and "کتاب" are different
  tokens and a merge never has to learn the space separately.
* Urdu punctuation (۔ ، ؟ ؛) is Unicode P/S, so it falls into the punctuation alternative and is
  separated from the preceding word without needing a space.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import regex

EOT = "<|endoftext|>"
EOT_ID = 256
N_BYTES = 256
FIRST_MERGE_ID = 257
VOCAB_SIZE = 50_000
N_MERGES = VOCAB_SIZE - FIRST_MERGE_ID  # 49,743
FORMAT = "urdu-bpe-1"

PRETOKENIZE_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"| ?[\p{L}\p{M}\u200c\u200d]+"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{M}\p{N}\u200c\u200d]+"
    r"|\s+(?!\S)|\s+"
)


def base_vocab() -> list[bytes]:
    return [bytes([i]) for i in range(N_BYTES)] + [EOT.encode("ascii")]


def merge_ids(ids: Sequence[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``ids``, scanning left to right.

    Training and encoding share this one function so both agree on overlaps such as (a, a) in "aaa".
    """
    a, b = pair
    out: list[int] = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def printable(token: bytes) -> str:
    """Readable form of a token: UTF-8 text if valid, else \\xNN escapes for the stray bytes."""
    return token.decode("utf-8", errors="backslashreplace")


class BPETokenizer:
    def __init__(
        self,
        merges: Iterable[tuple[int, int]] = (),
        pattern: str = PRETOKENIZE_PATTERN,
        *,
        cache_limit: int = 1_000_000,
    ) -> None:
        self.pattern = pattern
        self._regex = regex.compile(pattern)
        self.merges: list[tuple[int, int]] = [(int(a), int(b)) for a, b in merges]
        self.vocab: list[bytes] = base_vocab()
        for a, b in self.merges:
            limit = len(self.vocab)
            if not (0 <= a < limit and 0 <= b < limit) or EOT_ID in (a, b):
                raise ValueError(f"merge {(a, b)} refers to an id that does not exist yet")
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self._rank = {pair: FIRST_MERGE_ID + i for i, pair in enumerate(self.merges)}
        self._cache: dict[bytes, tuple[int, ...]] = {}
        self._cache_limit = cache_limit

    # properties
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def token_bytes(self, token_id: int) -> bytes:
        return self.vocab[token_id]

    def token_text(self, token_id: int) -> str:
        return printable(self.vocab[token_id])

    def is_valid_utf8(self, token_id: int) -> bool:
        try:
            self.vocab[token_id].decode("utf-8")
        except UnicodeDecodeError:
            return False
        return True

    # encoding
    def pretokenize(self, text: str) -> list[str]:
        return self._regex.findall(text)

    def _encode_word(self, word: bytes) -> tuple[int, ...]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = list(word)
        rank = self._rank
        while len(ids) > 1:
            best_rank, best_pair = None, None
            for pair in zip(ids, ids[1:], strict=False):
                r = rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None or best_rank is None:
                break
            ids = merge_ids(ids, best_pair, best_rank)  # the rank of a merge is its new token id
        result = tuple(ids)
        if len(self._cache) >= self._cache_limit:
            self._cache.clear()
        self._cache[word] = result
        return result

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode ``text`` treating ``<|endoftext|>`` as plain characters."""
        out: list[int] = []
        for piece in self._regex.findall(text):
            out.extend(self._encode_word(piece.encode("utf-8", "surrogatepass")))
        return out

    def encode(self, text: str, *, special: bool = True) -> list[int]:
        """Token ids for ``text``. With ``special`` (default) each ``<|endoftext|>`` becomes id 256."""
        if not special or EOT not in text:
            return self.encode_ordinary(text)
        out: list[int] = []
        for i, part in enumerate(text.split(EOT)):
            if i:
                out.append(EOT_ID)
            if part:
                out.extend(self.encode_ordinary(part))
        return out

    # decoding
    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        vocab, n = self.vocab, len(self.vocab)
        parts = []
        for i in ids:
            if not 0 <= i < n:
                raise ValueError(f"token id {i} is outside the vocabulary (size {n})")
            parts.append(vocab[i])
        return b"".join(parts)

    def decode(self, ids: Iterable[int]) -> str:
        data = self.decode_bytes(ids)
        try:
            return data.decode("utf-8", errors="surrogatepass")
        except UnicodeDecodeError:  # an arbitrary id list can end mid-character
            return data.decode("utf-8", errors="replace")

    # persistence
    def save(self, path: str | Path) -> None:
        """Write the tokenizer as JSON: header fields, then one ``[left, right]`` id pair per line."""
        header = {
            "format": FORMAT,
            "vocab_size": self.vocab_size,
            "special_tokens": {EOT: EOT_ID},
            "num_merges": len(self.merges),
            "pretokenize_pattern": self.pattern,
        }
        body = ",\n".join(json.dumps(list(m)) for m in self.merges)
        text = json.dumps(header, ensure_ascii=False)[:-1] + ', "merges": [\n' + body + "\n]}\n"
        Path(path).write_text(text, encoding="utf-8")

    def save_readable(self, path: str | Path) -> None:
        """One line per merge: ``rank  left + right -> merged`` as decoded text (human inspection)."""
        lines = []
        for i, (a, b) in enumerate(self.merges):
            left, right, both = (printable(self.vocab[x]) for x in (a, b, FIRST_MERGE_ID + i))
            lines.append(f"{i + 1}\t{FIRST_MERGE_ID + i}\t{left!r} + {right!r} -> {both!r}")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, expected_merges: int | None = None) -> BPETokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError(f"{path}: unknown tokenizer format {data.get('format')!r}")
        if data["special_tokens"] != {EOT: EOT_ID}:
            raise ValueError(f"{path}: unexpected special tokens {data['special_tokens']}")
        merges = [tuple(m) for m in data["merges"]]
        if len(merges) != data["num_merges"]:
            raise ValueError(
                f"{path}: header says {data['num_merges']} merges, file has {len(merges)}"
            )
        if expected_merges is not None and len(merges) != expected_merges:
            raise ValueError(f"{path}: expected {expected_merges} merges, found {len(merges)}")
        return cls(merges, data["pretokenize_pattern"])
