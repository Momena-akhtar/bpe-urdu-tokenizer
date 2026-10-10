"""Raw-document record, append-only JSONL writer, and per-source metadata.

``docs.jsonl`` holds one JSON object per line. The writer is crash-safe: on reopen it drops a
torn trailing line and recomputes counters, so a killed collector resumes where it stopped.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    id: str
    source: str
    genre: str
    text: str
    url: str | None = None
    title: str | None = None
    published: str | None = None  # ISO date as given by the source, when it has one
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        record = {k: v for k, v in asdict(self).items() if v not in (None, {}, "")}
        return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class SourceSpec:
    """Static facts about a source: what goes into the data card's source table."""

    name: str
    title: str
    genre: str
    method: str  # dump | published-download | api | scrape
    homepage: str
    licence: str
    licence_url: str
    terms_note: str
    self_collected: bool  # counts toward the 50 MB "collect it yourself" requirement
    default_target_mb: float
    description: str = ""


class DocWriter:
    """Append documents to ``docs.jsonl`` and track counts; ``full`` once the byte target is hit."""

    def __init__(self, path: Path, target_bytes: int | None = None) -> None:
        self.path = path
        self.target_bytes = target_bytes
        self.ids: set[str] = set()
        self.docs = 0
        self.text_bytes = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._recover()
        self._fh = path.open("a", encoding="utf-8")

    def _recover(self) -> None:
        if not self.path.exists():
            return
        good = 0
        with self.path.open("rb") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    break  # torn tail from a crash: everything after it is discarded
                good += len(line)
                self.ids.add(rec["id"])
                self.docs += 1
                self.text_bytes += len(rec["text"].encode("utf-8"))
        with self.path.open("r+b") as fh:
            fh.truncate(good)

    @property
    def full(self) -> bool:
        return self.target_bytes is not None and self.text_bytes >= self.target_bytes

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.ids

    def write(self, doc: Document) -> bool:
        if doc.id in self.ids:
            return False
        self._fh.write(doc.to_json() + "\n")
        self.ids.add(doc.id)
        self.docs += 1
        self.text_bytes += len(doc.text.encode("utf-8"))
        return True

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> DocWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_documents(path: Path):
    """Yield raw records (dicts) from a ``docs.jsonl``."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)
