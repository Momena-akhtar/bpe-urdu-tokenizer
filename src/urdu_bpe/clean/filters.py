"""Line- and document-level filters. Pure functions: no I/O, so each is unit-testable.

Everything here follows the assignment's cleaning pipeline: *filter* out lines that are mostly not
Urdu (keeping natural code-mixing such as English words inside Urdu sentences) and *deduplicate*
boilerplate. Lines are the unit because the collectors store one paragraph per line.
"""

from __future__ import annotations

import hashlib
import re

import regex

from urdu_bpe.collect.textutil import looks_like_clean_urdu

EOT = "<|endoftext|>"

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_LETTER = regex.compile(r"\p{L}")
_URDU_LETTER = regex.compile(r"(?V1)[\p{L}&&\p{Script=Arabic}]")
_SPACES = re.compile(r"\s+")

# Sources whose text is OCR output: whole documents must pass the frequent-word check.
OCR_SOURCES = frozenset({"archive_org_books"})


def line_hash(line: str) -> int:
    """64-bit hash of a line (whitespace-collapsed), used for boilerplate / duplicate detection."""
    key = _SPACES.sub(" ", line).strip().encode("utf-8", "surrogatepass")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


def strip_urls_and_emails(line: str) -> str:
    return _SPACES.sub(" ", _EMAIL.sub(" ", _URL.sub(" ", line))).strip()


def urdu_share(line: str) -> tuple[int, float]:
    """(number of Arabic-script letters, their share of all letters) in a line."""
    urdu = len(_URDU_LETTER.findall(line))
    letters = len(_LETTER.findall(line))
    return urdu, (urdu / letters if letters else 0.0)


def classify_line(
    line: str, *, min_urdu_letters: int = 2, min_urdu_share: float = 0.5
) -> str | None:
    """Reason to drop a line, or None to keep it."""
    if "�" in line:
        return "replacement_character"
    urdu, share = urdu_share(line)
    if urdu == 0:
        return "no_urdu_letters"
    if urdu < min_urdu_letters:
        return "too_few_urdu_letters"
    if share < min_urdu_share:
        return "mostly_not_urdu"
    return None


def _nbytes(text: str) -> int:
    return len(text.encode("utf-8", "surrogatepass"))


def clean_lines(text: str, counts: dict[str, list[int]]) -> list[str]:
    """Apply the per-line filters. ``counts[reason] = [lines, bytes]`` accumulates what was removed."""
    kept: list[str] = []
    for line in text.split("\n"):
        if not line:
            continue
        if EOT in line:  # a literal marker would be read as a document boundary downstream
            counts.setdefault("endoftext_marker", [0, 0])[1] += len(EOT) * line.count(EOT)  # ASCII
            line = line.replace(EOT, " ")
        had_link = bool(_URL.search(line) or _EMAIL.search(line))
        stripped = strip_urls_and_emails(line)
        if had_link:
            bucket = counts.setdefault("url_or_email_removed", [0, 0])
            bucket[0] += 1
            bucket[1] += _nbytes(line) - _nbytes(stripped)
        reason = classify_line(stripped)
        if reason:
            bucket = counts.setdefault(reason, [0, 0])
            bucket[0] += 1
            bucket[1] += _nbytes(stripped)
            continue
        kept.append(stripped)
    return kept


WORD = regex.compile(r"[\p{L}\p{M}]+")


class OcrLineFilter:
    """Keeps an OCR line only if it reads like text: known words, normal word lengths, few digits.

    Tesseract on Urdu scans yields lines of broken or invented words that a whole-document check
    cannot see. Three cheap scores catch them:

    * share of words that the trusted (non-OCR) corpus also uses: invented words are unknown;
    * mean word length and share of one-letter words: shredded text ("میا ےکی و سے دی یت") consists of
      fragments that are each a valid short word, so the vocabulary check alone passes it;
    * share of digits: stray "0", "7-" and similar are scanner noise inside Urdu prose.
    """

    def __init__(
        self,
        known: frozenset[str],
        *,
        min_known_share: float = 0.85,
        min_words: int = 4,
        min_mean_word_len: float = 3.0,
        max_single_letter_share: float = 0.12,
        max_digit_share: float = 0.05,
    ) -> None:
        self.known = known
        self.min_known_share = min_known_share
        self.min_words = min_words
        self.min_mean_word_len = min_mean_word_len
        self.max_single_letter_share = max_single_letter_share
        self.max_digit_share = max_digit_share

    def reason(self, line: str) -> str | None:
        words = WORD.findall(line)
        if len(words) < self.min_words:
            return "ocr_line_too_short"
        if sum(w in self.known for w in words) / len(words) < self.min_known_share:
            return "ocr_unknown_words"
        letters = [len(w) for w in words]
        if (
            sum(letters) / len(words) < self.min_mean_word_len
            or sum(n == 1 for n in letters) / len(words) > self.max_single_letter_share
        ):
            return "ocr_fragmented_words"
        compact = "".join(line.split())
        if sum(c.isdigit() for c in compact) / max(len(compact), 1) > self.max_digit_share:
            return "ocr_digit_noise"
        return None


def is_repetitive(
    lines: list[str], *, min_lines: int = 10, max_duplicate_share: float = 0.5
) -> bool:
    """A document where more than half the lines repeat earlier lines (listings, spam)."""
    if len(lines) < min_lines:
        return False
    return 1 - len(set(lines)) / len(lines) > max_duplicate_share


def fails_ocr_check(source: str, text: str) -> bool:
    return source in OCR_SOURCES and not looks_like_clean_urdu(text)
