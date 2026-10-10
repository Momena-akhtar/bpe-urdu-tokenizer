"""Encode a corpus file into a flat ``uint16`` array of token ids (``train.bin`` / ``val.bin``).

The file is cut into document-aligned chunks (see ``chunking``), encoded in parallel, and the id
arrays are appended in order, so the result equals encoding the whole file in one go. Every chunk is
decoded again and compared with its source text unless verification is switched off.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from urdu_bpe.bpe import BPETokenizer
from urdu_bpe.chunking import chunk_ranges, read_range

_TOKENIZER: BPETokenizer | None = None


def _init(tokenizer_path: str) -> None:
    global _TOKENIZER
    _TOKENIZER = BPETokenizer.load(tokenizer_path)


def _encode_range(args: tuple[str, int, int, bool]) -> tuple[bytes, int, bool]:
    path, start, stop, verify = args
    assert _TOKENIZER is not None
    text = read_range(Path(path), start, stop)
    ids = _TOKENIZER.encode(text)
    if max(ids, default=0) >= 1 << 16:
        raise ValueError("token id does not fit in uint16")
    ok = (not verify) or _TOKENIZER.decode(ids) == text
    return np.asarray(ids, dtype=np.uint16).tobytes(), len(text), ok


def encode_file(
    tokenizer_path: Path,
    source: Path,
    target: Path,
    *,
    workers: int | None = None,
    chunk_bytes: int = 16 << 20,
    verify: bool = True,
    progress=None,
) -> dict:
    started = time.perf_counter()
    ranges = chunk_ranges(source, chunk_bytes)
    workers = min(workers or max(1, (os.cpu_count() or 2) - 1), max(1, len(ranges)))
    jobs = [(str(source), a, b, verify) for a, b in ranges]
    tokens = chars = 0
    all_ok = True
    target.parent.mkdir(parents=True, exist_ok=True)
    with (
        target.open("wb") as out,
        mp.get_context("fork").Pool(
            workers, initializer=_init, initargs=(str(tokenizer_path),)
        ) as pool,
    ):
        for i, (blob, n_chars, ok) in enumerate(pool.imap(_encode_range, jobs)):  # ordered
            out.write(blob)
            tokens += len(blob) // 2
            chars += n_chars
            all_ok &= ok
            if progress:
                progress(i + 1, len(jobs))
    if not all_ok:
        raise RuntimeError(f"{source}: decode(encode(chunk)) != chunk for at least one chunk")
    nbytes = source.stat().st_size
    return {
        "source": str(source),
        "output": str(target),
        "tokens": tokens,
        "source_bytes": nbytes,
        "source_chars": chars,
        "bytes_per_token": round(nbytes / tokens, 3) if tokens else None,
        "chars_per_token": round(chars / tokens, 3) if tokens else None,
        "dtype": "uint16",
        "roundtrip_verified": verify,
        "seconds": round(time.perf_counter() - started, 1),
    }


def load_ids(path: Path) -> np.ndarray:
    """Memory-map a ``.bin`` file the way the language-model assignment reads it."""
    return np.memmap(path, dtype=np.uint16, mode="r")
