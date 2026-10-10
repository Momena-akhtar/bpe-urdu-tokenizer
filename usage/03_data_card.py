"""Stage 03: write DATA_CARD.md from data/raw metadata, cleaning_stats.json and the split files.

    uv run usage/03_data_card.py          # -> DATA_CARD.md in the repo root

Run after stage 02. Review the generated "Known gaps and biases" section and edit it by hand if the
numbers tell a different story than the prepared text; the assignment wants the card to fit one page.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from urdu_bpe.clean.datacard import build_card
from urdu_bpe.paths import CLEAN, RAW, ROOT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--clean", type=Path, default=CLEAN)
    p.add_argument("--raw", type=Path, default=RAW)
    p.add_argument("--out", type=Path, default=ROOT / "DATA_CARD.md")
    args = p.parse_args()
    args.out.write_text(build_card(args.clean, args.raw), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
