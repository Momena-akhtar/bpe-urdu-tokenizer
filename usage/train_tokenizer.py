"""Train the 50,000-token byte-level BPE tokenizer on data/clean/train.txt (deliverable script).

    uv run usage/train_tokenizer.py                       # full corpus, fast trainer
    uv run usage/train_tokenizer.py --max-bytes 200000000 # trial run on the first ~200 MB of documents

Phases: (1) count each distinct word once (parallel, cached in word_counts.pkl);
(2) learn 49,743 merges with the fast trainer (checkpointed: an interrupted run resumes);
(3) save tokenizer.json (all merges, in order), merges.txt (readable) and training_report.json
with wall-clock times and the machine, which the report must state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from urdu_bpe.bpe import (
    FIRST_MERGE_ID,
    PRETOKENIZE_PATTERN,
    VOCAB_SIZE,
    BPETokenizer,
)
from urdu_bpe.bpe_train import (
    FastBPETrainer,
    count_words_file,
    load_word_counts,
    save_word_counts,
)
from urdu_bpe.paths import CLEAN, TOKENIZER
from urdu_bpe.sysinfo import machine_info, peak_rss_gb


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", type=Path, default=CLEAN / "train.txt")
    p.add_argument("--out-dir", type=Path, default=TOKENIZER)
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    p.add_argument(
        "--workers", type=int, default=None, help="processes for word counting (default: cores-1)"
    )
    p.add_argument("--chunk-mb", type=int, default=64)
    p.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="only use the first N bytes (rounded to a document)",
    )
    p.add_argument("--checkpoint-every", type=int, default=1000, help="merges between checkpoints")
    p.add_argument(
        "--fresh", action="store_true", help="ignore the cached word counts and any checkpoint"
    )
    args = p.parse_args()

    num_merges = args.vocab_size - FIRST_MERGE_ID
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stat = args.input.stat()
    signature = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "limit": args.max_bytes,
        "pattern": PRETOKENIZE_PATTERN,
    }
    counts_path = args.out_dir / "word_counts.pkl"
    counts_meta = args.out_dir / "word_counts.meta.json"
    checkpoint = args.out_dir / "merges.partial.json"

    # phase 1: word counts
    counts = None if args.fresh else load_word_counts(counts_path, signature)
    if counts is not None:
        count_seconds = json.loads(counts_meta.read_text())["seconds"]
        print(
            f"word counts loaded from cache ({len(counts):,} distinct words; counted in {count_seconds:.0f}s earlier)"
        )
    else:
        t0 = time.perf_counter()
        with tqdm(desc="counting words", unit="chunk") as bar:

            def tick(done: int, total: int) -> None:
                bar.total = total
                bar.n = done
                bar.refresh()

            counts = count_words_file(
                args.input,
                workers=args.workers,
                chunk_bytes=args.chunk_mb << 20,
                limit=args.max_bytes,
                progress=tick,
            )
        count_seconds = time.perf_counter() - t0
        save_word_counts(counts, counts_path, signature)
        counts_meta.write_text(json.dumps({"seconds": round(count_seconds, 1)}))
    word_tokens = sum(counts.values())

    # phase 2: merges (resume from checkpoint when the input is unchanged)
    forced: list[tuple[int, int]] = []
    if checkpoint.exists() and not args.fresh:
        saved = json.loads(checkpoint.read_text())
        if saved["signature"] == signature:
            forced = [tuple(m) for m in saved["merges"]]
            print(f"resuming from checkpoint with {len(forced):,} merges")
    t1 = time.perf_counter()
    trainer = FastBPETrainer(counts)
    done: list[tuple[int, int]] = []
    bar = tqdm(total=num_merges, desc="learning merges", unit="merge")

    def on_merge(step: int, pair: tuple[int, int], count: int, token: bytes) -> None:
        done.append(pair)
        bar.update(1)
        if step % 500 == 0:
            bar.set_postfix(count=count, token=token.decode("utf-8", "backslashreplace")[-12:])
        if (step + 1) % args.checkpoint_every == 0 and step + 1 > len(forced):
            checkpoint.write_text(json.dumps({"signature": signature, "merges": done}))

    merges = trainer.train(num_merges, forced=forced, on_merge=on_merge)
    bar.close()
    train_seconds = time.perf_counter() - t1
    if len(merges) < num_merges:
        sys.exit(
            f"only {len(merges)} merges were possible (needed {num_merges}): the corpus is too small"
        )

    # phase 3: save and self-check
    tokenizer = BPETokenizer(merges)
    out = args.out_dir / "tokenizer.json"
    tokenizer.save(out)
    tokenizer.save_readable(args.out_dir / "merges.txt")
    reloaded = BPETokenizer.load(out, expected_merges=num_merges)
    probe = "اُردُو زبان، 2024 میں! hello 😀\n<|endoftext|>"
    assert (
        reloaded.decode(reloaded.encode(probe)) == probe and reloaded.vocab_size == args.vocab_size
    )
    checkpoint.unlink(missing_ok=True)

    total_seconds = count_seconds + train_seconds  # cached counts keep their measured counting time
    report = {
        "finished": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "input": str(args.input),
        "input_bytes": stat.st_size if args.max_bytes is None else args.max_bytes,
        "vocab_size": tokenizer.vocab_size,
        "merges": len(merges),
        "distinct_words": len(counts),
        "word_tokens": word_tokens,
        "seconds_count_words": round(count_seconds, 1),
        "seconds_learn_merges": round(train_seconds, 1),
        "seconds_total_wall_clock": round(total_seconds, 1),
        "resumed_merges": len(forced),
        "peak_rss_gb": peak_rss_gb(),
        "machine": machine_info(),
        "pretokenize_pattern": PRETOKENIZE_PATTERN,
    }
    (args.out_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("merges", "seconds_total_wall_clock", "peak_rss_gb", "machine")
            },
            indent=2,
        )
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
