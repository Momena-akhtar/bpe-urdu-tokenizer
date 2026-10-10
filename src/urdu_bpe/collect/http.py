"""Polite HTTP client: robots.txt, per-host rate limit, retries, resumable streaming, evidence log.

Every request made through ``PoliteSession`` is appended to ``evidence/requests.jsonl`` and every
robots.txt it consulted is saved verbatim under ``evidence/robots/``; together they are the proof
of self-collection that the assignment asks for.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from urdu_bpe.collect.robots import RobotsTxt

log = logging.getLogger(__name__)

USER_AGENT = (
    "UrduBPE-CourseCrawler/1.0 (NUST CS417 Assignment 1; non-commercial coursework; "
    "obeys robots.txt)"
)
RETRY_STATUS = {429, 500, 502, 503, 504}


class RobotsDisallowed(RuntimeError):
    pass


class FetchError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class PoliteSession:
    def __init__(
        self,
        evidence_dir: Path | None = None,
        *,
        min_delay: float = 1.0,
        timeout: tuple[float, float] = (15.0, 90.0),
        max_retries: int = 6,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.evidence_dir = evidence_dir
        self.min_delay = min_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self._http = requests.Session()
        self._http.headers.update({"User-Agent": user_agent, "Accept-Language": "ur,en;q=0.5"})
        self._robots: dict[str, RobotsTxt] = {}
        self._next_slot: dict[str, float] = {}
        self._lock = threading.Lock()
        self.requests = 0
        self.wire_bytes = 0
        if evidence_dir:
            (evidence_dir / "robots").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ robots.txt
    def _origin(self, url: str) -> str:
        parts = urlsplit(url)
        return f"{parts.scheme}://{parts.netloc}"

    def robots(self, url: str) -> RobotsTxt:
        origin = self._origin(url)
        with self._lock:
            cached = self._robots.get(origin)
        if cached is not None:
            return cached
        robots_url = origin + "/robots.txt"
        status: int | None = None
        body = ""
        for attempt in range(3):
            self._throttle(urlsplit(url).netloc, self.min_delay)
            try:
                resp = self._http.get(robots_url, timeout=self.timeout)
                status, body = resp.status_code, resp.text
                if status not in RETRY_STATUS:
                    break
            except requests.RequestException as exc:
                log.warning("robots.txt fetch failed (%s): %r", robots_url, exc)
            time.sleep(2**attempt)
        parsed = RobotsTxt.from_status(status, body)
        if self.evidence_dir:
            host = urlsplit(url).netloc
            (self.evidence_dir / "robots" / f"{host}.txt").write_text(body, encoding="utf-8")
            self._append_evidence(
                {"ts": _now(), "robots_url": robots_url, "status": status, "bytes": len(body)}
            )
        log.info("robots.txt %s -> %s", robots_url, status)
        with self._lock:
            self._robots[origin] = parsed
        return parsed

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        target = parts.path or "/"
        if parts.query:
            target += "?" + parts.query
        return self.robots(url).can_fetch(self.user_agent, target)

    def delay_for(self, url: str) -> float:
        declared = self.robots(url).crawl_delay(self.user_agent) or 0.0
        return max(self.min_delay, declared)

    # ------------------------------------------------------------------ requests
    def _throttle(self, host: str, delay: float) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot.get(host, 0.0))
            self._next_slot[host] = slot + delay
        if slot > now:
            time.sleep(slot - now)

    def _append_evidence(self, record: dict) -> None:
        if not self.evidence_dir:
            return
        with self._lock, (self.evidence_dir / "requests.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get(
        self,
        url: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        stream: bool = False,
        check_robots: bool = True,
        retries: int | None = None,
    ) -> requests.Response:
        """GET with robots check, throttling and retry/backoff. Returns any non-retryable status."""
        if check_robots and not self.allowed(url):
            raise RobotsDisallowed(url)
        delay = self.delay_for(url) if check_robots else self.min_delay
        host = urlsplit(url).netloc
        last_error = "no attempt made"
        attempts = retries or self.max_retries
        for attempt in range(attempts):
            self._throttle(host, delay)
            started = time.monotonic()
            try:
                resp = self._http.get(
                    url, params=params, headers=headers, stream=stream, timeout=self.timeout
                )
                if not stream:
                    resp.content  # noqa: B018  (read the body so the byte count is final)
            except requests.RequestException as exc:
                last_error = repr(exc)
                self._append_evidence({"ts": _now(), "url": url, "error": last_error})
                self._backoff(attempt, None)
                continue
            wire = 0 if stream else self._wire_bytes(resp)
            self.requests += 1
            self.wire_bytes += wire
            self._append_evidence(
                {
                    "ts": _now(),
                    "url": resp.url,
                    "status": resp.status_code,
                    "bytes": wire,
                    "ms": int((time.monotonic() - started) * 1000),
                }
            )
            if resp.status_code in RETRY_STATUS:
                last_error = f"HTTP {resp.status_code}"
                self._backoff(attempt, resp.headers.get("Retry-After"))
                continue
            return resp
        raise FetchError(f"{url}: gave up after {attempts} attempts ({last_error})")

    @staticmethod
    def _wire_bytes(resp: requests.Response) -> int:
        declared = resp.headers.get("Content-Length")  # compressed size when gzip is used
        return int(declared) if declared and declared.isdigit() else len(resp.content)

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> None:
        wait = min(2 ** (attempt + 1), 120) * random.uniform(0.8, 1.2)
        if retry_after and retry_after.isdigit():
            wait = max(wait, min(int(retry_after), 300))
        time.sleep(wait)

    def get_json(self, url: str, **kwargs: Any) -> Any:
        resp = self.get(url, **kwargs)
        if resp.status_code != 200:
            raise FetchError(f"{url}: HTTP {resp.status_code}")
        return resp.json()

    # ------------------------------------------------------------------ big files
    def stream_cached(
        self, url: str, cache: Path, *, chunk_size: int = 1 << 18, check_robots: bool = False
    ) -> Iterator[bytes]:
        """Yield a remote file's bytes, caching them on disk so any later run resumes for free.

        Cached bytes are replayed first, then the remainder is fetched with ``Range`` requests and
        appended to the cache. The consumer may stop early (e.g. once its byte target is reached);
        the partial cache stays valid. Dropped connections are re-opened from the current offset.
        """
        cache.parent.mkdir(parents=True, exist_ok=True)
        offset = 0
        if cache.exists():
            with cache.open("rb") as fh:
                while chunk := fh.read(chunk_size):
                    offset += len(chunk)
                    yield chunk
        failures = 0
        with cache.open("ab") as out:
            while True:
                headers = {"Range": f"bytes={offset}-"} if offset else None
                resp = self.get(url, headers=headers, stream=True, check_robots=check_robots)
                try:
                    if resp.status_code == 416:  # nothing left: the cache is the whole file
                        return
                    if resp.status_code not in (200, 206):
                        raise FetchError(f"{url}: HTTP {resp.status_code}")
                    if offset and resp.status_code == 200:
                        raise FetchError(f"{url}: server ignored Range; delete {cache} and retry")
                    try:
                        for chunk in resp.iter_content(chunk_size):
                            out.write(chunk)
                            offset += len(chunk)
                            self.wire_bytes += len(chunk)
                            failures = 0
                            yield chunk
                        out.flush()
                        return
                    except requests.RequestException as exc:
                        failures += 1
                        out.flush()
                        if failures > self.max_retries:
                            raise FetchError(
                                f"{url}: stream kept failing at byte {offset}"
                            ) from exc
                        log.warning("stream dropped at byte %d (%r); reconnecting", offset, exc)
                        self._backoff(failures, None)
                finally:
                    resp.close()
