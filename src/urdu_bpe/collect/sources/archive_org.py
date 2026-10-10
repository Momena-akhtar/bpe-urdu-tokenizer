"""Public-domain Urdu books from the Internet Archive (OCR text, search + download APIs)."""

from __future__ import annotations

from urllib.parse import quote

from urdu_bpe.collect.base import Collector
from urdu_bpe.collect.http import FetchError
from urdu_bpe.collect.record import Document, DocWriter
from urdu_bpe.collect.textutil import looks_like_clean_urdu, tidy_lines, urdu_ratio

SEARCH = "https://archive.org/advancedsearch.php"
QUERY = (
    "(language:urdu OR language:urd) AND mediatype:texts AND "
    "(licenseurl:*publicdomain* OR possible-copyright-status:NOT_IN_COPYRIGHT)"
)


class ArchiveOrgBooks(Collector):
    min_chars = 5000  # OCR of a real book, not a catalogue card
    min_urdu = 0.7  # Tesseract on Urdu scans is noisy; low-ratio items are mostly garbage
    max_file_bytes = 6_000_000  # one book must not cost minutes on a slow link

    def identifiers(self, page: int) -> list[dict]:
        data = self.session.get_json(
            SEARCH,
            params={
                "q": QUERY,
                "fl[]": ["identifier", "title", "licenseurl", "year"],
                "sort[]": "downloads desc",
                "rows": 100,
                "page": page,
                "output": "json",
            },
        )
        return data["response"]["docs"]

    def ocr_text_url(self, ident: str) -> tuple[str | None, str]:
        """The item's OCR text file (format DjVuTXT) from its metadata, or (None, reason)."""
        meta = self.session.get_json(f"https://archive.org/metadata/{quote(ident)}")
        for f in meta.get("files", []):
            if f.get("format") == "DjVuTXT":
                if int(f.get("size", 0)) > self.max_file_bytes:
                    return None, "too_large_for_slow_link"
                return f"https://archive.org/download/{quote(ident)}/{quote(f['name'])}", ""
        return None, "no_ocr_text"

    def collect(self, writer: DocWriter) -> None:
        state = self.load_state()
        rejected: set[str] = set(state.get("rejected", []))
        page = state.get("page", 1)
        while not writer.full:
            items = self.identifiers(page)
            if not items:
                self.exhausted = True
                break
            for item in items:
                if writer.full:
                    break
                ident = item["identifier"]
                doc_id = f"archive-{ident}"
                if doc_id in writer or ident in rejected:
                    continue
                try:
                    reason = self._fetch(item, doc_id, writer)
                except FetchError as exc:  # one broken item (HTTP 500) must not abort the source
                    self.log.warning("skipping %s: %s", ident, exc)
                    reason = "fetch_failed"
                if reason:
                    self.skip(reason)
                    rejected.add(ident)  # never refetch a book we already judged unusable
                writer.flush()
                self.save_state({"page": page, "rejected": sorted(rejected)})
                self.progress(writer)
            page += 1
            writer.flush()
            self.save_state({"page": page, "rejected": sorted(rejected)})

    def _fetch(self, item: dict, doc_id: str, writer: DocWriter) -> str | None:
        ident = item["identifier"]
        url, reason = self.ocr_text_url(ident)
        if url is None:
            return reason
        resp = self.session.get(url)
        if resp.status_code != 200:
            return f"http_{resp.status_code}"
        text = tidy_lines(resp.text)
        if len(text) < self.min_chars:
            return "too_short"
        if urdu_ratio(text) < self.min_urdu:
            return "low_urdu_ratio"
        if not looks_like_clean_urdu(text):
            return "ocr_noise"
        writer.write(
            Document(
                doc_id,
                self.spec.name,
                self.spec.genre,
                text,
                url=f"https://archive.org/details/{quote(ident)}",
                title=item.get("title"),
                published=str(item["year"]) if item.get("year") else None,
                extra={"licenseurl": item.get("licenseurl")},
            )
        )
        return None
