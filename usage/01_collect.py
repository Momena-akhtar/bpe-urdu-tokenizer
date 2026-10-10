"""Stage 01: collect raw Urdu text into data/raw/<source>/ (resumable, polite, evidence kept).

uv run usage/01_collect.py --list
uv run usage/01_collect.py --sources self            # the scraped / API sources (>= 50 MB)
uv run usage/01_collect.py --sources all --workers 3
uv run usage/01_collect.py --sources ary_urdu --target-mb ary_urdu=1   # smoke test
uv run usage/01_collect.py --status                  # table for the data card
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from urdu_bpe.collect import registry
from urdu_bpe.paths import DOWNLOADS, RAW

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(funcName)s | %(message)s"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sources", default="all", help="all | self | published | comma list of names")
    p.add_argument(
        "--target-mb",
        action="append",
        default=[],
        metavar="NAME=MB",
        help="override a source's text target in MB (0 = everything); repeatable",
    )
    p.add_argument("--out", type=Path, default=RAW)
    p.add_argument("--downloads", type=Path, default=DOWNLOADS)
    p.add_argument("--workers", type=int, default=3, help="sources collected concurrently")
    p.add_argument("--min-delay", type=float, default=1.0, help="seconds between requests per host")
    p.add_argument("--force", action="store_true", help="re-run sources already marked complete")
    p.add_argument("--list", action="store_true", help="list sources and exit")
    p.add_argument("--status", action="store_true", help="print the collection table and exit")
    return p.parse_args()


def select(arg: str) -> list[str]:
    if arg == "all":
        return list(registry.FACTORIES)
    if arg == "self":
        return registry.SELF_COLLECTED
    if arg == "published":
        return registry.PUBLISHED
    names = [n.strip() for n in arg.split(",") if n.strip()]
    unknown = [n for n in names if n not in registry.FACTORIES]
    if unknown:
        sys.exit(f"unknown source(s): {unknown}; use --list")
    return names


def print_list() -> None:
    for name, (spec, _) in registry.FACTORIES.items():
        kind = "self-collected" if spec.self_collected else "published"
        target = f"{spec.default_target_mb:g} MB" if spec.default_target_mb else "all"
        print(
            f"{name:20s} {spec.genre:13s} {spec.method:18s} {kind:15s} {target:>8s}  {spec.title}"
        )
    print("\nrejected:")
    for name, why in registry.REJECTED.items():
        print(f"  {name}: {why}")


def print_status(root: Path) -> None:
    rows, total, own = [], 0, 0
    for name, (spec, _) in registry.FACTORIES.items():
        meta_path = root / name / "meta.json"
        if not meta_path.exists():
            rows.append(f"| {name} | {spec.genre} | - | - | - | not started |")
            continue
        meta = json.loads(meta_path.read_text())
        mb = meta["text_bytes"] / 1e6
        total += mb
        own += mb if spec.self_collected else 0
        rows.append(
            f"| {name} | {spec.genre} | {meta['docs']:,} | {mb:,.1f} | "
            f"{meta['first_collected'][:10]} | {meta['status']} |"
        )
    print(
        "| source | genre | docs | text MB | first collected | status |\n|---|---|---|---|---|---|"
    )
    print("\n".join(rows))
    print(f"\ntotal {total:,.1f} MB extracted text, of which self-collected {own:,.1f} MB")


def main() -> None:
    args = parse_args()
    if args.list:
        return print_list()
    if args.status:
        return print_status(args.out)
    overrides = {}
    for item in args.target_mb:
        name, _, mb = item.partition("=")
        overrides[name] = float(mb)
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.out / "collect.log", encoding="utf-8"),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    def run(name: str) -> tuple[str, dict]:
        collector = registry.build(
            name,
            args.out,
            target_mb=overrides.get(name),
            downloads=args.downloads,
            min_delay=args.min_delay,
        )
        return name, collector.run(force=args.force)

    names = select(args.sources)
    logging.info("collecting %s with %d workers", names, args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, meta in pool.map(run, names):
            logging.info(
                "%s: %s, %d docs, %.1f MB",
                name,
                meta["status"],
                meta["docs"],
                meta["text_bytes"] / 1e6,
            )
    print_status(args.out)


if __name__ == "__main__":
    main()
