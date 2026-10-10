"""BPE training, in the two stages the assignment asks for.

``train_reference``  recounts every pair after every merge: obviously correct, far too slow for the
                     full corpus, used on a 1 MB sample as the ground truth.
``FastBPETrainer``   the same algorithm with the three optimisations:
                       1. each *distinct* word is counted once and its pairs weighted by frequency;
                       2. after a merge only words that contained the merged pair are revisited,
                          found through a pair -> words index, and pair counts are updated in place;
                       3. (at encoding time, ``BPETokenizer``) token ids of already-seen words are cached.

Both pick the pair with the highest count and break ties by the *greater pair compared as a tuple of
bytes* (``(bytes_left, bytes_right)``). Distinct token ids can share the same bytes, so the token ids
are a final tie-break; that makes the two trainers produce identical merges, which
``usage/04_verify_bpe.py`` checks.
"""

from __future__ import annotations

import heapq
import multiprocessing as mp
import os
import pickle
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import regex

from urdu_bpe.bpe import EOT, PRETOKENIZE_PATTERN, base_vocab, merge_ids
from urdu_bpe.chunking import chunk_ranges, read_range

MergeCallback = Callable[[int, tuple[int, int], int, bytes], None]  # step, pair, count, new token


# --------------------------------------------------------------------------- word counting
def count_words_text(text: str, pattern: str = PRETOKENIZE_PATTERN) -> Counter[bytes]:
    """Frequency of every pre-token (as UTF-8 bytes). ``<|endoftext|>`` is split out first so it is
    never counted and never joins a merge."""
    rx = regex.compile(pattern)
    counts: Counter[bytes] = Counter()
    for part in text.split(EOT):
        counts.update(w.encode("utf-8", "surrogatepass") for w in rx.findall(part))
    return counts


def _count_range(args: tuple[str, int, int, str]) -> Counter[bytes]:
    path, start, stop, pattern = args
    return count_words_text(read_range(Path(path), start, stop), pattern)


def count_words_file(
    path: Path,
    *,
    workers: int | None = None,
    chunk_bytes: int = 64 << 20,
    limit: int | None = None,
    pattern: str = PRETOKENIZE_PATTERN,
    progress: Callable[[int, int], None] | None = None,
) -> Counter[bytes]:
    """Word frequencies of a whole corpus file, counted in parallel over document-aligned chunks."""
    ranges = chunk_ranges(path, chunk_bytes, limit=limit)
    workers = min(workers or max(1, (os.cpu_count() or 2) - 1), len(ranges))
    total: Counter[bytes] = Counter()
    jobs = [(str(path), a, b, pattern) for a, b in ranges]
    if workers <= 1:
        for i, job in enumerate(jobs):
            total.update(_count_range(job))
            if progress:
                progress(i + 1, len(jobs))
        return total
    with mp.get_context("fork").Pool(workers) as pool:
        for i, counts in enumerate(pool.imap_unordered(_count_range, jobs)):
            total.update(counts)
            if progress:
                progress(i + 1, len(jobs))
    return total


def save_word_counts(counts: Mapping[bytes, int], path: Path, signature: dict) -> None:
    with path.open("wb") as fh:
        pickle.dump(
            {"signature": signature, "counts": dict(counts)}, fh, protocol=pickle.HIGHEST_PROTOCOL
        )


def load_word_counts(path: Path, signature: dict) -> Counter[bytes] | None:
    """Cached counts, or None when absent or computed from a different input."""
    if not path.exists():
        return None
    with path.open("rb") as fh:
        blob = pickle.load(fh)
    return Counter(blob["counts"]) if blob["signature"] == signature else None


# --------------------------------------------------------------------------- stage 1: reference
def train_reference(
    text: str,
    num_merges: int,
    pattern: str = PRETOKENIZE_PATTERN,
    on_merge: MergeCallback | None = None,
) -> list[tuple[int, int]]:
    """Textbook BPE: recount *every* pair over *every* word occurrence after *every* merge."""
    rx = regex.compile(pattern)
    words: list[list[int]] = []
    for part in text.split(EOT):
        words += [list(w.encode("utf-8", "surrogatepass")) for w in rx.findall(part)]
    vocab = base_vocab()
    merges: list[tuple[int, int]] = []
    for step in range(num_merges):
        counts: Counter[tuple[int, int]] = Counter()
        for word in words:
            counts.update(zip(word, word[1:], strict=False))
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], (vocab[p[0]], vocab[p[1]]), p))
        new_id = len(vocab)
        words = [merge_ids(w, best, new_id) for w in words]
        vocab.append(vocab[best[0]] + vocab[best[1]])
        merges.append(best)
        if on_merge:
            on_merge(step, best, counts[best], vocab[new_id])
    return merges


# --------------------------------------------------------------------------- stage 2: fast
class _PairKey:
    """Heap key that makes the *greatest* (bytes_left, bytes_right, left, right) pop first among
    equal counts (heapq is a min-heap, so the comparison is reversed)."""

    __slots__ = ("t",)

    def __init__(self, t: tuple[bytes, bytes, int, int]) -> None:
        self.t = t

    def __lt__(self, other: _PairKey) -> bool:
        return self.t > other.t

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _PairKey) and self.t == other.t


def _pack(a: int, b: int) -> int:
    return (a << 16) | b  # token ids stay below 65,536


class FastBPETrainer:
    """Weighted-word BPE with an incremental pair index and a lazy max-heap."""

    def __init__(self, word_counts: Mapping[bytes, int]) -> None:
        self.vocab: list[bytes] = base_vocab()
        self.words: list[list[int]] = []
        self.freqs: list[int] = []
        for word, freq in word_counts.items():
            if len(word) > 1:  # single bytes have no pairs
                self.words.append(list(word))
                self.freqs.append(freq)
        self.pair_counts: dict[int, int] = {}
        self.index: dict[int, list[int]] = {}  # pair -> word indices (may hold stale entries)
        for wi, ids in enumerate(self.words):
            freq, seen = self.freqs[wi], set()
            for a, b in zip(ids, ids[1:], strict=False):
                key = _pack(a, b)
                self.pair_counts[key] = self.pair_counts.get(key, 0) + freq
                if key not in seen:
                    seen.add(key)
                    self.index.setdefault(key, []).append(wi)
        self.heap: list[tuple[int, _PairKey, int]] = [
            (-count, self._key(key), key) for key, count in self.pair_counts.items()
        ]
        heapq.heapify(self.heap)

    def _key(self, packed: int) -> _PairKey:
        a, b = packed >> 16, packed & 0xFFFF
        return _PairKey((self.vocab[a], self.vocab[b], a, b))

    def best_pair(self) -> int | None:
        """Packed pair with the highest count (ties: greatest bytes), or None when no pair is left.

        The heap is lazy: counts only ever *rise* through a push, so an entry whose count no longer
        matches is stale; it is dropped and, if the pair still exists, re-queued at its real count.
        """
        heap, counts = self.heap, self.pair_counts
        while heap:
            neg, key_obj, packed = heap[0]
            actual = counts.get(packed, 0)
            if actual == -neg:
                return packed
            heapq.heappop(heap)
            if actual > 0:
                heapq.heappush(heap, (-actual, key_obj, packed))
        return None

    def merge(self, packed: int) -> int:
        """Merge a pair everywhere it occurs; returns the new token id."""
        a, b = packed >> 16, packed & 0xFFFF
        new_id = len(self.vocab)
        self.vocab.append(self.vocab[a] + self.vocab[b])
        counts, index, words, freqs = self.pair_counts, self.index, self.words, self.freqs
        risen: set[int] = set()
        for wi in set(index.pop(packed, ())):
            ids = words[wi]
            merged = merge_ids(ids, (a, b), new_id)
            if len(merged) == len(ids):  # stale index entry: the word no longer holds the pair
                continue
            freq = freqs[wi]
            old: dict[int, int] = {}
            for x, y in zip(ids, ids[1:], strict=False):
                k = _pack(x, y)
                old[k] = old.get(k, 0) + 1
            new: dict[int, int] = {}
            for x, y in zip(merged, merged[1:], strict=False):
                k = _pack(x, y)
                new[k] = new.get(k, 0) + 1
            for k in old.keys() | new.keys():
                delta = new.get(k, 0) - old.get(k, 0)
                if not delta:
                    continue
                total = counts.get(k, 0) + delta * freq
                if total:
                    counts[k] = total
                else:
                    counts.pop(k, None)
                if delta > 0:
                    risen.add(k)
                    if k not in old:
                        index.setdefault(k, []).append(wi)
            words[wi] = merged
        counts.pop(packed, None)
        for k in risen:
            if k in counts:
                heapq.heappush(self.heap, (-counts[k], self._key(k), k))
        return new_id

    def train(
        self,
        num_merges: int,
        *,
        forced: Sequence[tuple[int, int]] = (),
        on_merge: MergeCallback | None = None,
    ) -> list[tuple[int, int]]:
        """Learn up to ``num_merges`` merges. ``forced`` replays merges saved by an earlier,
        interrupted run (checkpoint resume) before choosing new ones."""
        merges: list[tuple[int, int]] = []
        for step in range(num_merges):
            if step < len(forced):
                packed = _pack(*forced[step])
            else:
                best = self.best_pair()
                if best is None:
                    break
                packed = best
            count = self.pair_counts.get(packed, 0)
            new_id = self.merge(packed)
            pair = (packed >> 16, packed & 0xFFFF)
            merges.append(pair)
            if on_merge:
                on_merge(step, pair, count, self.vocab[new_id])
        return merges


def train_fast(
    word_counts: Mapping[bytes, int],
    num_merges: int,
    *,
    forced: Sequence[tuple[int, int]] = (),
    on_merge: MergeCallback | None = None,
) -> list[tuple[int, int]]:
    return FastBPETrainer(word_counts).train(num_merges, forced=forced, on_merge=on_merge)
