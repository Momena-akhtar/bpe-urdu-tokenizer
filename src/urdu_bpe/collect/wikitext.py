"""MediaWiki markup to plain text (Urdu Wikipedia/Wikisource/Wikibooks dumps).

Regex based, in the spirit of WikiExtractor: templates, tables, references, files, categories,
interlanguage links and HTML are removed; link labels, headings and prose are kept. Sections that
only list references or external links are cut off.
"""

from __future__ import annotations

import html
import re

_FLAGS = re.DOTALL | re.IGNORECASE
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_TABLE = re.compile(r"\{\|(?:(?!\{\|).)*?\|\}", re.DOTALL)
_BLOCK_TAGS = re.compile(
    r"<(ref|gallery|math|timeline|syntaxhighlight|source|score|imagemap|references|noinclude"
    r"|categorytree|mapframe|maplink)\b[^>]*?(?:/>|>.*?</\1\s*>)",
    _FLAGS,
)
_SELF_CLOSING_REF = re.compile(r"<ref\b[^>]*/>", _FLAGS)
_NOWIKI = re.compile(r"<nowiki>(.*?)</nowiki>", _FLAGS)
_NAMESPACE_OPEN = re.compile(
    r"\[\[\s*(?:file|image|category|media|special|template|talk|user|help|portal|wikipedia|wp"
    r"|فائل|ملف|تصویر|زمرہ|میڈیا|خصوصی|سانچہ|تبادلۂ خیال|صارف|معاونت|باب|وکیپیڈیا|ویکیپیڈیا"
    r"|[a-z]{2,3}(?:-[a-z]+)?)\s*:",
    _FLAGS,
)
_LINK = re.compile(
    r"\[\[\s*(?:file|image|category|media|special|template|talk|user|help|portal|wikipedia|wp"
    r"|فائل|تصویر|زمرہ|میڈیا|خصوصی|سانچہ|تبادلۂ خیال|صارف|معاونت|باب|وکیپیڈیا|ویکیپیڈیا"
    r"|[a-z]{2,3}(?:-[a-z]+)?)\s*:[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]",
    _FLAGS,
)
_LINK = re.compile(r"\[\[:?([^\[\]|]*)(?:\|([^\[\]]*))?\]\]")
_EXT_LINK = re.compile(r"\[(?:https?:)?//[^\s\]]+(?:\s+([^\]]*))?\]")
_BARE_URL = re.compile(r"https?://\S+")
_HEADING = re.compile(r"^(={1,6})\s*(.*?)\s*\1\s*$", re.MULTILINE)
_LIST_MARK = re.compile(r"^[*#:;]+\s*", re.MULTILINE)
_RULE = re.compile(r"^-{4,}\s*$", re.MULTILINE)
_QUOTES = re.compile(r"'{2,5}")
_MAGIC = re.compile(r"__[A-Z]+__")
_TAGS = re.compile(r"</?[a-zA-Z][^>]*>")
_BLANKS = re.compile(r"\n{2,}")

# Sections that carry no prose, in Urdu and English; everything after the first one is dropped.
_TAIL_HEADINGS = {
    "حوالہ جات",
    "حوالے",
    "بیرونی روابط",
    "بیرونی ربط",
    "مزید دیکھیے",
    "مزید دیکھیں",
    "مزید پڑھیے",
    "حواشی",
    "حاشیہ",
    "نوٹ",
    "ماخذ",
    "کتابیات",
    "references",
    "external links",
    "see also",
    "notes",
    "further reading",
    "sources",
}


def _strip_nested(text: str, pattern: re.Pattern[str], limit: int = 60) -> str:
    """Remove innermost matches repeatedly so nested {{templates}} / {| tables |} all go."""
    for _ in range(limit):
        new = pattern.sub("", text)
        if new == text:
            break
        text = new
    return text


def _drop_namespaced_links(text: str) -> str:
    """Remove [[File:..]], [[Category:..]], [[en:..]] etc. with balanced brackets, so captions that
    themselves contain links or stray brackets are removed whole."""
    out, pos = [], 0
    while (m := _NAMESPACE_OPEN.search(text, pos)) is not None:
        out.append(text[pos : m.start()])
        depth, single, i = 1, 0, m.start() + 2
        while i < len(text) and depth:
            if text.startswith("[[", i):
                depth, i = depth + 1, i + 2
            elif text.startswith("]]]", i) and single:  # "[caption]" + the closing "]]"
                single, i = single - 1, i + 1
            elif text.startswith("]]", i):
                depth, i = depth - 1, i + 2
            else:
                single += text[i] == "["
                i += 1
        pos = i
    out.append(text[pos:])
    return "".join(out)


def _link_label(match: re.Match[str]) -> str:
    target, label = match.group(1), match.group(2)
    return (label if label is not None else target.split("#", 1)[0]).strip()


def strip_wikitext(source: str) -> str:
    text = _COMMENT.sub("", source)
    text = _NOWIKI.sub(lambda m: m.group(1), text)
    text = _SELF_CLOSING_REF.sub("", text)
    text = _BLOCK_TAGS.sub("", text)
    text = _strip_nested(text, _TEMPLATE)
    text = _strip_nested(text, _TABLE)
    text = text.replace("{{", "").replace("}}", "").replace("{|", "").replace("|}", "")
    text = _drop_namespaced_links(text)
    text = _LINK.sub(_link_label, text)
    text = _EXT_LINK.sub(lambda m: m.group(1) or "", text)
    text = _BARE_URL.sub("", text)
    text = _TAGS.sub("", text)
    text = html.unescape(text)
    text = _QUOTES.sub("", text)
    text = _MAGIC.sub("", text)
    text = _RULE.sub("", text)

    kept: list[str] = []
    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            title = heading.group(2).strip()
            if title.lower() in _TAIL_HEADINGS:
                break
            line = title
        line = _LIST_MARK.sub("", line).strip()
        # table remnants such as "| align=center" or "! header" are not prose
        if not line or line[0] in "|!" or line.startswith("&"):
            continue
        kept.append(line)
    return _BLANKS.sub("\n", "\n".join(kept)).strip()
