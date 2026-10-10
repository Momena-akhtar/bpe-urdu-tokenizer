"""Express Urdu (express.pk): HTML scraper driven by the site's own sitemaps."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from pathlib import Path

from bs4 import BeautifulSoup

from urdu_bpe.collect.base import Collector
from urdu_bpe.collect.http import RobotsDisallowed
from urdu_bpe.collect.record import Document, DocWriter, SourceSpec
from urdu_bpe.collect.textutil import html_to_text, urdu_ratio

SITEMAP_INDEX = "https://www.express.pk/sitemap.xml"
LOC = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")
STORY_ID = re.compile(r"/story/(\d+)/")
JUNK = (".ads", ".advertisement", ".related-news", ".social-share", ".share", "[class*=banner]")


class ExpressScraper(Collector):
    min_chars = 200
    min_urdu = 0.6

    def __init__(
        self,
        spec: SourceSpec,
        root: Path,
        *,
        seed: int = 0,
        per_sitemap: int = 150,
        **kwargs: object,
    ) -> None:
        super().__init__(spec, root, **kwargs)  # type: ignore[arg-type]
        self.seed = seed
        self.per_sitemap = per_sitemap

    def _locs(self, url: str) -> list[str]:
        resp = self.session.get(url)
        return LOC.findall(resp.text) if resp.status_code == 200 else []

    def story_urls(self) -> Iterator[str]:
        """Yield story URLs lazily: sitemaps in a seeded random order, and from each only a seeded
        random sample. The result spans the whole archive (years of news) and a run that stops early
        has fetched a few sitemaps instead of all of them (each is ~0.8 MB)."""
        rng = random.Random(self.seed)
        sitemaps = [u for u in self._locs(SITEMAP_INDEX) if "/posts-" in u]
        rng.shuffle(sitemaps)
        self.log.info("%d post sitemaps", len(sitemaps))
        for sitemap in sitemaps:
            urls = [u for u in self._locs(sitemap) if STORY_ID.search(u)]
            rng.shuffle(urls)
            yield from urls[: self.per_sitemap]

    def parse(self, page_html: str) -> tuple[str | None, str, str | None]:
        soup = BeautifulSoup(page_html, "lxml")
        published = None
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(tag.string or "")
            except json.JSONDecodeError:
                continue
            for item in data if isinstance(data, list) else [data]:
                if isinstance(item, dict) and item.get("datePublished"):
                    published = item["datePublished"]
        h1 = soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else None
        body = soup.select_one(".story-text")
        return title, html_to_text(body, drop_selectors=JUNK) if body else "", published

    def collect(self, writer: DocWriter) -> None:
        for i, url in enumerate(self.story_urls()):
            if writer.full:
                return
            doc_id = f"express-{STORY_ID.search(url).group(1)}"  # type: ignore[union-attr]
            if doc_id in writer:
                continue  # resume: the same seed replays the same URL order
            try:
                resp = self.session.get(url)
            except RobotsDisallowed:
                self.skip("robots")
                continue
            if resp.status_code != 200:
                self.skip(f"http_{resp.status_code}")
                continue
            title, body, published = self.parse(resp.text)
            if len(body) < self.min_chars:
                self.skip("too_short")
            elif urdu_ratio(body) < self.min_urdu:
                self.skip("not_urdu")
            else:
                writer.write(
                    Document(doc_id, self.spec.name, self.spec.genre, body, url, title, published)
                )
            if i % 25 == 0:
                self.progress(writer)
        self.exhausted = True
