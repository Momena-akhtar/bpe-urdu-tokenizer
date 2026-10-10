"""Encode train.txt and val.txt with the trained tokenizer into train.bin / val.bin (deliverable).

    uv run usage/encode_corpus.py              # data/clean/{train,val}.txt -> data/tokenized/*.bin

Each .bin is a flat NumPy uint16 array (np.fromfile(path, dtype=np.uint16)); the next assignment
reads them directly. Every chunk is decoded again and compared with its source text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from urdu_bpe.encode import encode_file
from urdu_bpe.paths import CLEAN, DATA, TOKENIZER


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER / "tokenizer.json")
    p.add_argument("--splits", nargs="+", default=["train", "val"], help="split names in --clean")
    p.add_argument("--clean", type=Path, default=CLEAN)
    p.add_argument("--out-dir", type=Path, default=DATA / "tokenized")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--chunk-mb", type=int, default=16)
    p.add_argument("--no-verify", action="store_true", help="skip the decode(encode(chunk)) check")
    args = p.parse_args()

    reports = []
    for split in args.splits:
        with tqdm(desc=f"encoding {split}", unit="chunk") as bar:

            def tick(done: int, total: int) -> None:
                bar.total = total
                bar.n = done
                bar.refresh()

            reports.append(
                encode_file(
                    args.tokenizer,
                    args.clean / f"{split}.txt",
                    args.out_dir / f"{split}.bin",
                    workers=args.workers,
                    chunk_bytes=args.chunk_mb << 20,
                    verify=not args.no_verify,
                    progress=tick,
                )
            )
        r = reports[-1]
        print(
            f"{split}: {r['tokens']:,} tokens, {r['bytes_per_token']} bytes/token, {r['seconds']}s"
        )
    (args.out_dir / "encoding_report.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
