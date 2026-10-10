"""HTML to plain text and the cheap language check used while collecting.

Collection only *extracts* text: it removes markup and obviously empty lines. Unicode
normalisation (yeh/kaf/heh variants, tatweel, digits, ...) belongs to ``normalize.py`` and runs
in the cleaning stage, never here, so the raw data stays an honest record of the source.
"""

from __future__ import annotations

import re

import regex
from bs4 import BeautifulSoup, Tag

_LETTER = regex.compile(r"\p{L}")
_ARABIC_LETTER = regex.compile(r"(?V1)[\p{L}&&\p{Script=Arabic}]")
_BLOCKS = [
    "p",
    "div",
    "li",
    "ul",
    "ol",
    "br",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "pre",
    "tr",
    "table",
    "section",
    "article",
    "header",
    "footer",
]
_DROP = [
    "script",
    "style",
    "noscript",
    "iframe",
    "form",
    "nav",
    "aside",
    "button",
    "svg",
    "figure",
    "figcaption",
    "template",
]
_SPACES = re.compile(r"[ \t   ]+")


def urdu_ratio(text: str) -> float:
    """Fraction of letters written in Arabic script (Urdu shares it with Arabic and Persian)."""
    letters = len(_LETTER.findall(text))
    return len(_ARABIC_LETTER.findall(text)) / letters if letters else 0.0


_WORD = regex.compile(r"[\p{L}\p{M}]+")
_SCRIPT_CHAR = regex.compile(r"(?V1)[[\p{L}&&\p{Script=Arabic}]\p{M}]")
# Very frequent Urdu function words: genuine prose and poetry hit 12-25% of tokens, OCR noise ~1%.
_FUNCTION_WORDS = frozenset(
    [
        "کی",
        "کے",
        "میں",
        "ہے",
        "اور",
        "کہ",
        "سے",
        "کو",
        "نے",
        "ہیں",
        "پر",
        "یہ",
        "کا",
        "ایک",
        "بھی",
        "تھا",
        "تھی",
        "جو",
        "ہو",
        "گیا",
    ]
)


def function_word_rate(text: str) -> float:
    words = _WORD.findall(text)
    return sum(w in _FUNCTION_WORDS for w in words) / len(words) if words else 0.0


def script_char_ratio(text: str) -> float:
    """Share of non-space characters that are Arabic-script letters or combining marks."""
    chars = "".join(text.split())
    return len(_SCRIPT_CHAR.findall(chars)) / len(chars) if chars else 0.0


def looks_like_clean_urdu(
    text: str, *, min_function_words: float = 0.10, min_script: float = 0.85
) -> bool:
    """Rejects OCR noise and mis-detected languages that a letter-script ratio alone lets through."""
    return function_word_rate(text) >= min_function_words and script_char_ratio(text) >= min_script


def tidy_lines(text: str) -> str:
    """Strip each line, drop empty ones, join with single newlines (one paragraph per line)."""
    lines = (_SPACES.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def html_to_text(html: str | Tag, *, drop_selectors: tuple[str, ...] = ()) -> str:
    """Visible text of an HTML fragment, one paragraph per line."""
    soup = BeautifulSoup(html, "lxml") if isinstance(html, str) else html
    for tag in soup.find_all(_DROP):
        tag.decompose()
    for selector in drop_selectors:
        for tag in soup.select(selector):
            tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(_BLOCKS):
        tag.insert_before("\n")
        tag.append("\n")
    return tidy_lines(soup.get_text())
