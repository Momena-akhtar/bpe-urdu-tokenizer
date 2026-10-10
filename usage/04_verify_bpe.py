"""Stage 04: prove the fast trainer learns the same merges as the reference trainer.

    uv run usage/04_verify_bpe.py                      # sanity check + 1 MB sample, 1,000 merges
    uv run usage/04_verify_bpe.py --merges 3000        # longer comparison (reference is slow)

1. Sanity check: the classic "aaabdaaabac" example with hand-derived merges.
2. A ~1 MB sample of whole documents drawn at random from train.txt: both trainers run, the merge
   lists must be identical, and the wall-clock time of each is reported.
Writes data/tokenizer/verify_report.json (numbers for the report).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from urdu_bpe.bpe import EOT, FIRST_MERGE_ID, BPETokenizer
from urdu_bpe.bpe_train import count_words_text, train_fast, train_reference
from urdu_bpe.chunking import EOT_BYTES
from urdu_bpe.paths import CLEAN, TOKENIZER

SANITY_TEXT = "aaabdaaabac"
# Counts: aa=4 first; then (aa,a)=2 ties (a,b)=2 and wins as the greater pair of bytes; then (aaa,b)=2.
SANITY_MERGES = [(97, 97), (257, 97), (258, 98)]


def sample_documents(path: Path, megabytes: float, seed: int) -> str:
    """Whole documents from random places in the file until ~megabytes of text are collected."""
    rng = random.Random(seed)
    size = path.stat().st_size
    docs: list[bytes] = []
    seen: set[int] = set()
    total = 0
    with path.open("rb") as fh:
        while total < megabytes * 1e6:
            start = rng.randrange(size)
            fh.seek(start)
            tail = fh.read(1 << 20)
            first = tail.find(EOT_BYTES)
            if first < 0 or start + first in seen:
                continue
            seen.add(start + first)
            rest = tail[first + len(EOT_BYTES) :]  # a document starts right after a marker
            end = rest.find(EOT_BYTES)
            if end < 0:
                continue
            docs.append(rest[: end + len(EOT_BYTES)])
            total += end
    return b"".join(docs).decode("utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", type=Path, default=CLEAN / "train.txt")
    p.add_argument("--merges", type=int, default=1000)
    p.add_argument("--size-mb", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=TOKENIZER / "verify_report.json")
    args = p.parse_args()

    ref = train_reference(SANITY_TEXT, 3)
    fast = train_fast(count_words_text(SANITY_TEXT), 3)
    assert ref == fast == SANITY_MERGES, (ref, fast)
    print(f"sanity check passed: {SANITY_TEXT!r} -> {fast}")

    sample = sample_documents(args.input, args.size_mb, args.seed)
    print(f"sample: {len(sample.encode()) / 1e6:.2f} MB, {sample.count(EOT):,} documents")
    t0 = time.perf_counter()
    counts = count_words_text(sample)
    fast = train_fast(counts, args.merges)
    fast_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref = train_reference(sample, args.merges)
    ref_s = time.perf_counter() - t0
    identical = ref == fast
    probe = sample[:100_000]
    tok = BPETokenizer(fast)
    lossless = tok.decode(tok.encode(probe)) == probe
    report = {
        "sample_mb": round(len(sample.encode()) / 1e6, 3),
        "merges_requested": args.merges,
        "merges_reference": len(ref),
        "merges_fast": len(fast),
        "identical": identical,
        "first_difference": next(
            (i for i, (a, b) in enumerate(zip(ref, fast, strict=False)) if a != b), None
        ),
        "seconds_reference": round(ref_s, 1),
        "seconds_fast": round(fast_s, 1),
        "speedup": round(ref_s / fast_s, 1) if fast_s else None,
        "sample_roundtrip_lossless": lossless,
        "first_merges": [tok.token_text(FIRST_MERGE_ID + i) for i in range(min(10, len(fast)))],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not identical:
        raise SystemExit("MISMATCH: the fast trainer diverged from the reference")
    print("fast trainer == reference trainer on the sample")


if __name__ == "__main__":
    main()
