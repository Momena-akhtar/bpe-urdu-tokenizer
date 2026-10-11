"""Urdu text normalisation, written from scratch (standard library only).

The same Urdu word is often typed with different code points that look identical on screen.
Left alone, a tokenizer learns every variant as a separate token. ``normalize`` maps them onto one
canonical form. It is a *separate step*: the tokenizer never normalises, so
``decode(encode(text)) == text`` holds for any string and normalisation stays inspectable.

Pipeline (order matters and is idempotent: ``normalize(normalize(x)) == normalize(x)``)::

    1. presentation forms (U+FB50-FDFF, U+FE70-FEFC, fullwidth ASCII) -> base letters (NFKC, only there)
    2. NFC           compose letter+mark sequences the same way everywhere
    3. character map replace or delete code points (table below)
    4. NFC           again, because deleting a tatweel can bring a letter and a mark together
    5. collapse repeated identical diacritics (zabar zabar -> zabar)
    6. whitespace    one space between words, one newline between paragraphs

Decisions the assignment leaves open, and why:

* **Digits -> ASCII 0-9.** Modern Urdu web text mixes Arabic-Indic, Persian and ASCII digits; the
  ASCII form dominates in the news corpus and gives numbers one spelling to learn.
* **ZWNJ (U+200C) -> removed.** It is typed inconsistently, so "ہوگئی" and "ہو\u200cگئی" are the same
  word spelled two ways; deleting it merges those variants instead of doubling the vocabulary. The
  tokenizer's pre-tokenizer still tolerates ZWNJ, so un-normalised text is handled losslessly.
* **Diacritics (harakat U+064B-0652, U+0670) are kept.** They are real text in poetry and religious
  prose, and Part C tests them. Only Quranic annotation signs (U+06D6-06ED) are removed.
* **Teh marbuta ة -> ۃ, alef wasla ٱ -> ا, ae ە -> ہ** because those are Arabic/Uyghur keyboard
  artefacts of Urdu letters.
* Line breaks are paragraph boundaries in the corpus, so blank lines collapse to one newline.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

ZWNJ = "\u200c"

# the rule table
# (rule name, description, {source code point: replacement}); replacement "" deletes.


def _span(lo: int, hi: int) -> range:
    return range(lo, hi + 1)


def _map(codes: list[int] | range, target: str) -> dict[int, str]:
    return {c: target for c in codes}


_RULES: list[tuple[str, str, dict[int, str]]] = [
    (
        "arabic_yeh",
        "Arabic yeh ي (U+064A) and alef maksura ى (U+0649) -> Urdu yeh ی (U+06CC)",
        _map([0x064A, 0x0649], "ی"),
    ),
    (
        "arabic_kaf",
        "Arabic kaf ك (U+0643) and Sindhi kaf ڪ (U+06AA) -> Urdu kaf ک (U+06A9)",
        _map([0x0643, 0x06AA], "ک"),
    ),
    (
        "heh_variants",
        "Arabic heh ه (U+0647) and ae ە (U+06D5) -> gol heh ہ (U+06C1); do-chashmi ھ (U+06BE) is kept",
        _map([0x0647, 0x06D5], "ہ"),
    ),
    ("teh_marbuta", "teh marbuta ة (U+0629) -> teh marbuta goal ۃ (U+06C3)", _map([0x0629], "ۃ")),
    ("alef_wasla", "alef wasla ٱ (U+0671) -> alef ا (U+0627)", _map([0x0671], "ا")),
    ("tatweel", "tatweel ـ (U+0640) removed", _map([0x0640], "")),
    ("zwnj", "zero-width non-joiner (U+200C) removed (see module docstring)", _map([0x200C], "")),
    (
        "invisible",
        "zero-width space/joiner, direction marks, bidi controls, BOM, soft hyphen removed",
        _map(
            [
                0x200B,
                0x200D,
                0x200E,
                0x200F,
                0x061C,
                0x00AD,
                0xFEFF,
                0x2060,
                0x2061,
                0x2062,
                0x2063,
                0x2064,
                0x180E,
                *_span(0x202A, 0x202E),
                *_span(0x2066, 0x2069),
            ],
            "",
        ),
    ),
    (
        "quranic_marks",
        "Quranic annotation signs (U+06D6-06ED, minus the letters U+06E5/6) removed",
        _map([*_span(0x06D6, 0x06E4), 0x06E7, 0x06E8, *_span(0x06EA, 0x06ED)], ""),
    ),
    (
        "arabic_indic_digits",
        "Arabic-Indic digits ٠-٩ (U+0660-0669) -> ASCII 0-9",
        {0x0660 + i: str(i) for i in range(10)},
    ),
    (
        "persian_digits",
        "Extended Arabic-Indic digits ۰-۹ (U+06F0-06F9) -> ASCII 0-9",
        {0x06F0 + i: str(i) for i in range(10)},
    ),
    (
        "arabic_number_marks",
        "Arabic decimal/thousands separators and percent -> . , %",
        {0x066B: ".", 0x066C: ",", 0x066A: "%"},
    ),
    (
        "unicode_spaces",
        "no-break and typographic spaces, tab -> space",
        _map([0x00A0, 0x1680, *_span(0x2000, 0x200A), 0x202F, 0x205F, 0x3000, 0x0009], " "),
    ),
    (
        "line_breaks",
        "CR, VT, FF, NEL, LS, PS -> newline",
        _map([0x000D, 0x000B, 0x000C, 0x0085, 0x2028, 0x2029], "\n"),
    ),
    (
        "control_chars",
        "C0/C1 control characters and DEL removed",
        _map(
            [*_span(0x00, 0x08), *_span(0x0E, 0x1F), 0x7F, *_span(0x80, 0x84), *_span(0x86, 0x9F)],
            "",
        ),
    ),
]

RULES: dict[str, str] = {name: desc for name, desc, _ in _RULES}
_TABLE: dict[int, str] = {cp: to for _, _, mapping in _RULES for cp, to in mapping.items()}
_RULE_OF: dict[str, str] = {chr(cp): name for name, _, mapping in _RULES for cp in mapping}

_PRESENTATION = re.compile("[ﭐ-﷿ﹰ-ﻼ！-～]+")
_REPEATED_MARK = re.compile("([ً-ْٰ])\\1+")
_SPACES = re.compile(" {2,}")
_AROUND_NEWLINE = re.compile(" *\n *")
_NEWLINES = re.compile("\n{2,}")


def normalize(text: str) -> str:
    text = _PRESENTATION.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_TABLE)
    text = unicodedata.normalize("NFC", text)
    text = _REPEATED_MARK.sub(r"\1", text)
    text = _SPACES.sub(" ", text)
    text = _AROUND_NEWLINE.sub("\n", text)
    text = _NEWLINES.sub("\n", text)
    return text.strip()


def rule_hits(text: str) -> Counter[str]:
    """How many characters of ``text`` each rule rewrites (for the cleaning report)."""
    hits: Counter[str] = Counter()
    for char, n in Counter(text).items():
        rule = _RULE_OF.get(char)
        if rule:
            hits[rule] += n
        elif _PRESENTATION.fullmatch(char):
            hits["presentation_forms"] += n
    return hits
