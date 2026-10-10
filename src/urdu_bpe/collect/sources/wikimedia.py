"""Wikimedia XML dumps (Urdu Wikipedia, Wikisource, Wikibooks): stream, parse, strip wikitext."""

from __future__ import annotations

import bz2
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

from urdu_bpe.collect.base import Collector
from urdu_bpe.collect.record import Document, DocWriter, SourceSpec
from urdu_bpe.collect.textutil import urdu_ratio
from urdu_bpe.collect.wikitext import strip_wikitext

DUMP_URL = "https://dumps.wikimedia.org/{wiki}/latest/{wiki}-latest-pages-articles.xml.bz2"
SITE = {
    "urwiki": "ur.wikipedia.org",
    "urwikisource": "ur.wikisource.org",
    "urwikibooks": "ur.wikibooks.org",
}


class WikimediaDump(Collector):
    def __init__(
        self,
        spec: SourceSpec,
        root: Path,
        *,
        wiki: str,
        namespaces: frozenset[str] = frozenset({"0"}),
        min_chars: int = 300,
        **kwargs: object,
    ) -> None:
        super().__init__(spec, root, **kwargs)  # type: ignore[arg-type]
        self.wiki = wiki
        self.namespaces = namespaces
        self.min_chars = min_chars

    def collect(self, writer: DocWriter) -> None:
        url = DUMP_URL.format(wiki=self.wiki)
        cache = self.downloads / f"{self.wiki}-latest-pages-articles.xml.bz2"
        parser: ET.XMLPullParser = ET.XMLPullParser(events=("end",))
        decomp = bz2.BZ2Decompressor()
        for chunk in self.session.stream_cached(url, cache):
            data = chunk
            while data:
                parser.feed(decomp.decompress(data))
                if not decomp.eof:
                    break
                data, decomp = decomp.unused_data, bz2.BZ2Decompressor()  # next bz2 stream
            self._drain(parser, writer)
            self.progress(writer)
        self._drain(parser, writer)
        self.exhausted = True

    def _drain(self, parser: ET.XMLPullParser, writer: DocWriter) -> None:
        for _, elem in parser.read_events():  # type: ignore[misc]  # typeshed also allows ns tuples
            if not isinstance(elem, ET.Element) or not elem.tag.endswith("}page"):
                continue
            ns_uri = elem.tag[: -len("page")]
            self._page(elem, ns_uri, writer)
            elem.clear()

    def _page(self, page: ET.Element, ns: str, writer: DocWriter) -> None:
        if page.findtext(f"{ns}ns") not in self.namespaces:
            return
        if page.find(f"{ns}redirect") is not None:
            self.skip("redirect")
            return
        page_id = f"{self.wiki}-{page.findtext(f'{ns}id')}"
        if page_id in writer:
            return
        title = page.findtext(f"{ns}title") or ""
        raw = page.findtext(f"{ns}revision/{ns}text") or ""
        text = strip_wikitext(raw)
        if len(text) < self.min_chars:
            self.skip("too_short")
            return
        if urdu_ratio(text) < 0.6:
            self.skip("not_urdu")
            return
        url = f"https://{SITE[self.wiki]}/wiki/{quote(title.replace(' ', '_'))}"
        published = page.findtext(f"{ns}revision/{ns}timestamp")
        writer.write(
            Document(page_id, self.spec.name, self.spec.genre, text, url, title, published)
        )
