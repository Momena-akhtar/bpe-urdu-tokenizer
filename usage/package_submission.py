"""Assemble the submission: one archive for the code and small files, a manifest for the drive link.

    uv run usage/package_submission.py --report report.pdf

Archive (submission/urdu_bpe_submission.zip): src/, usage/, tests/, pyproject.toml, README.md,
DATA_CARD.md, report.pdf, tokenizer.json (all 49,743 merges), merges.txt, training and verification
reports, and evidence/<source>/ (robots.txt snapshots, request logs, meta.json) of every self-collected
source. Large files (train/val/test.txt, train/val.bin) go on the shared drive:
submission/DRIVE_FILES.md lists them with sizes and SHA-256 so the link can be checked afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

from urdu_bpe.bpe import N_MERGES, BPETokenizer
from urdu_bpe.collect import registry
from urdu_bpe.paths import CLEAN, DATA, RAW, ROOT, TOKENIZER

CODE_DIRS = ["src", "usage", "tests"]
SKIP = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--report", type=Path, default=ROOT / "report.pdf")
    p.add_argument("--out-dir", type=Path, default=ROOT / "submission")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = TOKENIZER / "tokenizer.json"
    BPETokenizer.load(
        tokenizer, expected_merges=N_MERGES
    )  # fails unless it has exactly 49,743 merges
    required = {
        "DATA_CARD.md": ROOT / "DATA_CARD.md",
        "report.pdf": args.report,
        "README.md": ROOT / "README.md",
        "pyproject.toml": ROOT / "pyproject.toml",
        "tokenizer.json": tokenizer,
    }
    optional = {
        "merges.txt": TOKENIZER / "merges.txt",
        "training_report.json": TOKENIZER / "training_report.json",
        "verify_report.json": TOKENIZER / "verify_report.json",
        "cleaning_stats.json": CLEAN / "cleaning_stats.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise SystemExit(f"missing required deliverables: {missing}")

    archive = args.out_dir / "urdu_bpe_submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, path in {**required, **{k: v for k, v in optional.items() if v.exists()}}.items():
            zf.write(path, name)
        for source in registry.SELF_COLLECTED:  # proof of the >= 50 MB self-collected requirement
            base = RAW / source
            for path in [base / "meta.json", *sorted((base / "evidence").rglob("*"))]:
                if path.is_file():
                    zf.write(path, Path("evidence") / source / path.relative_to(base))
        for folder in CODE_DIRS:
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_file() and not SKIP & set(path.parts) and path.suffix != ".pyc":
                    zf.write(path, path.relative_to(ROOT))
    print(f"wrote {archive} ({archive.stat().st_size / 1e6:.1f} MB)")

    drive = [
        CLEAN / "train.txt",
        CLEAN / "val.txt",
        CLEAN / "test.txt",
        DATA / "tokenized" / "train.bin",
        DATA / "tokenized" / "val.bin",
    ]
    rows = ["| file | size MB | sha256 |", "|---|---|---|"]
    for path in drive:
        if path.exists():
            rows.append(f"| {path.name} | {path.stat().st_size / 1e6:,.1f} | {sha256(path)} |")
        else:
            rows.append(f"| {path.name} | MISSING | |")
    manifest = args.out_dir / "DRIVE_FILES.md"
    manifest.write_text(
        "Upload these to the shared drive and put the link in the submission:\n\n"
        + "\n".join(rows)
        + "\n"
    )
    print(manifest.read_text())


if __name__ == "__main__":
    main()
