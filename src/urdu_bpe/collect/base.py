"""Collector template: output folder, resume, progress, and the ``meta.json`` the data card reads."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from urdu_bpe.collect.http import PoliteSession
from urdu_bpe.collect.record import DocWriter, SourceSpec

PROGRESS_EVERY_S = 30.0


class Collector(ABC):
    """One source. Subclasses implement ``collect``; everything else is shared.

    Layout produced under ``<root>/<name>/``::

        docs.jsonl        one extracted document per line
        meta.json         spec, licence, counts, dates, status (complete | partial)
        state.json        resume cursor (collector specific)
        evidence/         requests.jsonl + robots/<host>.txt (proof of how the data was collected)
    """

    def __init__(
        self,
        spec: SourceSpec,
        root: Path,
        *,
        target_mb: float | None = None,
        downloads: Path | None = None,
        min_delay: float = 1.0,
    ) -> None:
        self.spec = spec
        self.dir = root / spec.name
        self.target_mb = spec.default_target_mb if target_mb is None else target_mb
        self.downloads = downloads or root.parent / "downloads"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session = PoliteSession(self.dir / "evidence", min_delay=min_delay)
        self.log = logging.getLogger(f"collect.{spec.name}")
        self.skipped: dict[str, int] = {}
        self.exhausted = False  # set when the source ran out of documents before the target
        self._last_progress = 0.0

    # helpers for subclasses
    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    def load_state(self) -> dict:
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def save_state(self, state: dict) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        tmp.replace(self.state_path)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def progress(self, writer: DocWriter, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_progress < PROGRESS_EVERY_S:
            return
        self._last_progress = now
        writer.flush()
        target = f"/{self.target_mb:.0f}" if self.target_mb else ""
        self.log.info(
            "%d docs, %.1f%s MB text, %.1f MB on the wire, %d requests",
            writer.docs,
            writer.text_bytes / 1e6,
            target,
            self.session.wire_bytes / 1e6,
            self.session.requests,
        )

    # template
    @abstractmethod
    def collect(self, writer: DocWriter) -> None:
        """Write documents until ``writer.full`` or the source is exhausted."""

    def run(self, *, force: bool = False) -> dict:
        meta_path = self.dir / "meta.json"
        previous = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        target_bytes = int(self.target_mb * 1e6) if self.target_mb else None
        enough = target_bytes is None or previous.get("text_bytes", 0) >= target_bytes * 0.98
        if previous.get("status") == "complete" and not force and enough:
            self.log.info("already complete (%.1f MB), skipping", previous["text_bytes"] / 1e6)
            return previous
        started = datetime.now(UTC).isoformat(timespec="seconds")
        status = "partial"
        error: str | None = None
        with DocWriter(self.dir / "docs.jsonl", target_bytes) as writer:
            self.log.info(
                "start: target %s MB, %d docs already on disk",
                self.target_mb or "all",
                writer.docs,
            )
            try:
                self.collect(writer)
                status = "complete" if writer.full or self.exhausted else "partial"
            except KeyboardInterrupt:
                error = "interrupted"
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.log.exception("collection failed")
            finally:
                self.progress(writer, force=True)
                meta = {
                    **asdict(self.spec),
                    "status": status,
                    "error": error,
                    "target_mb": self.target_mb,
                    "docs": writer.docs,
                    "text_bytes": writer.text_bytes,
                    "wire_bytes_total": previous.get("wire_bytes_total", 0)
                    + self.session.wire_bytes,
                    "requests_total": previous.get("requests_total", 0) + self.session.requests,
                    "skipped_at_collection": self.skipped,
                    "first_collected": previous.get("first_collected", started),
                    "last_run": started,
                    "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                }
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        return meta
