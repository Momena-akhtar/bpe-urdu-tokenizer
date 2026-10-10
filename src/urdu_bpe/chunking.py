"""Split a big corpus file into byte ranges that start and end on document boundaries.

Training counts words and encoding turns text into ids chunk by chunk, in parallel. A chunk must
never cut a document (or a multi-byte character) in half, so ranges end just after a
``<|endoftext|>`` marker, which is also where the tokenizer's own pre-tokenisation restarts. Hence
chunked results equal whole-file results exactly.
"""

from __future__ import annotations

from pathlib import Path

EOT_BYTES = b"<|endoftext|>"
_BLOCK = 1 << 20


def chunk_ranges(
    path: Path, target_bytes: int, *, limit: int | None = None
) -> list[tuple[int, int]]:
    """Ranges of roughly ``target_bytes``, each ending right after an end-of-text marker.

    ``limit`` caps the total bytes considered (the last range still ends on a document boundary,
    so the file's first ``limit`` bytes are rounded up to a whole document).
    """
    size = path.stat().st_size
    end_of_scope = size if limit is None else min(size, limit)
    ranges: list[tuple[int, int]] = []
    start = 0
    with path.open("rb") as fh:
        while start < end_of_scope:
            probe = min(start + target_bytes, size)
            stop = size
            if probe < size:
                stop = _next_boundary(fh, probe, size)
            ranges.append((start, stop))
            start = stop
    return ranges


def _next_boundary(fh, probe: int, size: int) -> int:
    """Offset just after the first end-of-text marker that ends at or after ``probe``."""
    pos = max(0, probe - len(EOT_BYTES) + 1)  # a marker may straddle the probe point
    tail = b""
    while pos < size:
        fh.seek(pos)
        block = fh.read(_BLOCK)
        if not block:
            break
        window = tail + block
        hit = window.find(EOT_BYTES)
        if hit >= 0:
            return pos - len(tail) + hit + len(EOT_BYTES)
        tail = window[-(len(EOT_BYTES) - 1) :]
        pos += len(block)
    return size


def read_range(path: Path, start: int, stop: int) -> str:
    with path.open("rb") as fh:
        fh.seek(start)
        return fh.read(stop - start).decode("utf-8")
