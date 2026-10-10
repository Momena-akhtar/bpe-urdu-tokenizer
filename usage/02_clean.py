"""Stage 02: normalise, filter, deduplicate and split data/raw into train/val/test files.

    uv run usage/02_clean.py                       # data/raw -> data/clean/{train,val,test}.txt
    uv run usage/02_clean.py --workers 8 --force   # redo stage 1 even if the raw files are unchanged

Outputs in data/clean/: train.txt, val.txt, test.txt (<|endoftext|> after every document),
split_manifest.json, cleaning_stats.json (what every step removed; feeds the data card).
Run it only after stage 01 has finished: it warns about sources that are not `complete`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from urdu_bpe.clean.pipeline import CleanConfig, run
from urdu_bpe.paths import CLEAN, RAW

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(funcName)s | %(message)s"


def main() -> None:
    defaults = CleanConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw", type=Path, default=RAW)
    p.add_argument("--out", type=Path, default=CLEAN)
    p.add_argument("--workers", type=int, default=4, help="sources normalised in parallel")
    p.add_argument("--force", action="store_true", help="redo stage 1 for unchanged sources")
    p.add_argument("--seed", type=int, default=defaults.seed, help="fixed seed for the split")
    p.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    p.add_argument("--test-fraction", type=float, default=defaults.test_fraction)
    p.add_argument("--min-doc-chars", type=int, default=defaults.min_doc_chars)
    p.add_argument(
        "--exclude",
        default=",".join(defaults.exclude_sources),
        help="comma list of sources to leave out of the corpus ('' keeps everything)",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.out / "clean.log", encoding="utf-8"),
        ],
    )
    for meta_path in sorted(args.raw.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta["status"] != "complete":
            logging.warning(
                "%s is '%s' (%.1f MB so far); cleaning what exists",
                meta["name"],
                meta["status"],
                meta["text_bytes"] / 1e6,
            )

    cfg = CleanConfig(
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        min_doc_chars=args.min_doc_chars,
        exclude_sources=tuple(x for x in args.exclude.split(",") if x),
    )
    report = run(args.raw, args.out, cfg, workers=args.workers, force=args.force)

    print("\n| step | docs removed | lines removed | MB removed |\n|---|---|---|---|")
    for step in report["steps"]:
        print(
            f"| {step['step']} | {step['docs']:,} | {step['lines']:,} | {step['bytes'] / 1e6:,.2f} |"
        )
    print("\n| split | docs | MB |\n|---|---|---|")
    total = sum(s["bytes"] for s in report["splits"].values())
    for part, s in report["splits"].items():
        print(
            f"| {part} | {s['docs']:,} | {s['bytes'] / 1e6:,.1f} ({100 * s['bytes'] / total:.2f}%) |"
        )
    print(f"\nwrote {args.out}/train.txt, val.txt, test.txt and cleaning_stats.json")


if __name__ == "__main__":
    main()
