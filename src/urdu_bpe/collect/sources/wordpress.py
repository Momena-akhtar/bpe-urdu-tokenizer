"""Generic collector for WordPress sites through their public REST API (``/wp-json/wp/v2``).

The REST API is the site's own machine-readable interface: 100 posts per request instead of one
HTML page per article, which is far gentler on the server. robots.txt is still checked for every
URL, the crawl delay is honoured, and sites that restrict the API to logged-in users are skipped.

Posts are walked newest to oldest with a ``before`` date cursor (stable while new posts arrive,
unlike page numbers) and the cursor is checkpointed so an interrupted run resumes.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from urdu_bpe.collect.base import Collector
from urdu_bpe.collect.http import FetchError, RobotsDisallowed
from urdu_bpe.collect.record import Document, DocWriter, SourceSpec
from urdu_bpe.collect.textutil import html_to_text, urdu_ratio

FIELDS = "id,date,link,title,content,categories"


@dataclass(frozen=True)
class Section:
    """A slice of a site with its own genre: all posts, or posts in/out of given categories."""

    name: str
    genre: str
    categories: tuple[str, ...] = ()  # slugs; empty = every post
    exclude: tuple[str, ...] = ()  # slugs of categories to leave out
    max_mb: float | None = None  # cap for this section alone


class WordPressCollector(Collector):
    min_chars = 200
    min_urdu = 0.6

    def __init__(
        self,
        spec: SourceSpec,
        root: Path,
        *,
        base_url: str,
        sections: tuple[Section, ...],
        **kwargs: object,
    ) -> None:
        super().__init__(spec, root, **kwargs)  # type: ignore[arg-type]
        self.api = base_url.rstrip("/") + "/wp-json/wp/v2"
        self.sections = sections

    def _category_ids(self, slugs: tuple[str, ...]) -> list[int]:
        if not slugs:
            return []
        data = self.session.get_json(
            f"{self.api}/categories", params={"slug": ",".join(slugs), "per_page": 100}
        )
        found = {c["slug"]: c["id"] for c in data}
        missing = [s for s in slugs if s not in found]
        if missing:
            self.log.warning("categories not found on the site: %s", missing)
        return list(found.values())

    def collect(self, writer: DocWriter) -> None:
        state = self.load_state()
        for section in self.sections:
            if writer.full:
                return
            self._section(section, writer, state)
        self.exhausted = all(state.get(f"{s.name}.done") for s in self.sections)

    def _section(self, section: Section, writer: DocWriter, state: dict) -> None:
        key = section.name
        if state.get(f"{key}.done"):
            return
        include = self._category_ids(section.categories)
        exclude = self._category_ids(section.exclude)
        if section.categories and not include:
            state[f"{key}.done"] = True
            return
        cap = int(section.max_mb * 1e6) if section.max_mb else None
        start, carried = writer.text_bytes, state.get(f"{key}.bytes", 0)  # bytes survive restarts

        def used() -> int:
            return carried + writer.text_bytes - start

        before = state.get(f"{key}.before")
        per_page = 100  # halved when the server errors on big pages (some hosts 500 at 100 posts)
        while not writer.full and (cap is None or used() < cap):
            params: dict[str, object] = {
                "per_page": per_page,
                "orderby": "date",
                "order": "desc",
                "_fields": FIELDS,
            }
            if before:
                params["before"] = before
            if include:
                params["categories"] = ",".join(map(str, include))
            if exclude:
                params["categories_exclude"] = ",".join(map(str, exclude))
            try:
                resp = self.session.get(f"{self.api}/posts", params=params, retries=3)
            except FetchError:
                if per_page > 1:
                    per_page = max(1, per_page // 4)
                    self.log.warning("server keeps failing; retrying with per_page=%d", per_page)
                elif before:
                    # a single post crashes the API: step past it by skipping one day of posts
                    skipped = datetime.fromisoformat(before) - timedelta(days=1)
                    self.log.warning("API fails on the next post; skipping to before %s", skipped)
                    self.skip("api_error_day_skipped")
                    before = skipped.isoformat()
                    per_page = 20
                else:
                    raise
                continue
            except RobotsDisallowed:
                self.log.error("robots.txt forbids %s; section %s abandoned", self.api, key)
                state[f"{key}.done"] = True
                break
            if resp.status_code in (400, 404):  # past the last page, or API closed
                state[f"{key}.done"] = True
                break
            if resp.status_code != 200:
                raise FetchError(f"{resp.url}: HTTP {resp.status_code}")
            posts = resp.json()
            if not posts:
                state[f"{key}.done"] = True
                break
            per_page = min(100, per_page * 2)  # recover after an error streak
            for post in posts:
                self._post(post, section, writer)
            oldest = min(p["date"] for p in posts)
            # `before` is exclusive; +1 s keeps same-second siblings (writer dedupes by id)
            nxt = (datetime.fromisoformat(oldest) + timedelta(seconds=1)).isoformat()
            if before and nxt >= before:  # tiny pages: the +1 s margin would not move the cursor
                nxt = oldest  # strictly earlier than `before`, so progress is guaranteed
            before = nxt
            state[f"{key}.before"] = before
            state[f"{key}.bytes"] = used()
            writer.flush()
            self.save_state(state)
            self.progress(writer)
        self.save_state(state)

    def _post(self, post: dict, section: Section, writer: DocWriter) -> None:
        doc_id = f"{self.spec.name}-{post['id']}"
        if doc_id in writer:
            return
        body = html_to_text(post["content"]["rendered"])
        if len(body) < self.min_chars:
            self.skip("too_short")
            return
        if urdu_ratio(body) < self.min_urdu:
            self.skip("not_urdu")
            return
        title = html.unescape(html_to_text(post["title"]["rendered"]))
        writer.write(
            Document(
                doc_id,
                self.spec.name,
                section.genre,
                body,
                url=post["link"],
                title=title or None,
                published=post["date"],
                extra={"section": section.name},
            )
        )
