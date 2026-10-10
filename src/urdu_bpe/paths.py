"""Filesystem layout shared by every stage. Nothing under ``data/`` is ever committed."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DOWNLOADS = DATA / "downloads"  # published corpora / dumps, cached so reruns never refetch
RAW = DATA / "raw"  # one folder per source: docs.jsonl + meta.json + evidence/
CLEAN = DATA / "clean"  # cleaned documents and train/val/test splits
TOKENIZER = DATA / "tokenizer"
SAMPLE_SETS = ROOT / "sample_sets"


def source_dir(name: str, root: Path = RAW) -> Path:
    return root / name
