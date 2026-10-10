# Urdu BPE tokenizer from scratch

CS417 Assignment 1 (NUST SEECS): collect and clean an Urdu corpus (Part A), implement byte-level
BPE with a 50,000-token vocabulary (Part B), analyse it (Part C).

## Setup

```bash
uv sync
uv run pytest -q
```

## Layout

| Path | Purpose |
|---|---|
| `src/urdu_bpe/normalize.py` | `normalize(text)`: our own Urdu normaliser (rule table documented in the module) |
| `src/urdu_bpe/bpe.py` | `BPETokenizer`: pre-tokenisation regex, encode, decode, save/load |
| `src/urdu_bpe/bpe_train.py` | reference trainer and fast trainer (weighted words, pair index, lazy heap) |
| `src/urdu_bpe/encode.py`, `chunking.py` | parallel, document-aligned corpus encoding to `uint16` |
| `src/urdu_bpe/collect/` | collection framework and one collector per source; `registry.py` lists them |
| `src/urdu_bpe/clean/` | line/document filters, dedup, split, data-card generator |
| `usage/01_collect.py` | stage 01: collect raw text into `data/raw/<source>/` |
| `usage/02_clean.py` | stage 02: normalise, filter, dedupe, split -> `data/clean/{train,val,test}.txt` |
| `usage/03_data_card.py` | stage 03: write `DATA_CARD.md` |
| `usage/04_verify_bpe.py` | stage 04: sanity check + reference-vs-fast equality on a 1 MB sample |
| `usage/train_tokenizer.py` | train to exactly 50,000 tokens -> `data/tokenizer/tokenizer.json` |
| `usage/encode_corpus.py` | encode train/val -> `data/tokenized/{train,val}.bin` |
| `usage/package_submission.py` | build the archive and the drive-file manifest |
| `data/` | never committed: `downloads/`, `raw/`, `clean/`, `tokenizer/`, `tokenized/` |

Stages are stateless drivers in `usage/` over library code in `src/`; each is resumable.

## Rebuild everything from scratch

```bash
uv sync
uv run pytest -q                                   # unit tests, no network
uv run usage/01_collect.py --sources all           # Part A: raw text (hours on a slow link; resumable)
uv run usage/02_clean.py                           # Part A: train.txt / val.txt / test.txt
uv run usage/03_data_card.py                       # Part A: DATA_CARD.md
uv run usage/04_verify_bpe.py                      # Part B: sanity check + fast == reference
uv run usage/train_tokenizer.py                    # Part B: 49,743 merges -> tokenizer.json
uv run usage/encode_corpus.py                      # Part B: train.bin / val.bin
uv run usage/package_submission.py --report report.pdf
```

**Rebuild the tokenizer from `train.txt` only** (what the grader needs, no collection):

```bash
uv sync
uv run usage/train_tokenizer.py --input train.txt --out-dir tokenizer_out
uv run usage/encode_corpus.py --tokenizer tokenizer_out/tokenizer.json --clean . --out-dir tokenized_out
```

Expected result: `tokenizer_out/tokenizer.json` with `"num_merges": 49743` and `"vocab_size": 50000`
(ids 0-255 bytes, 256 `<|endoftext|>`, 257-49,999 merges). `training_report.json` next to it holds the
wall-clock time and the machine.

## Part A, stage 01: collection

```bash
uv run usage/01_collect.py --list                    # sources, genres, targets, rejected sources
uv run usage/01_collect.py --sources all --workers 13
uv run usage/01_collect.py --status                  # table for the data card
```

Each source writes `data/raw/<source>/`: `docs.jsonl` (one extracted document per line), `meta.json`
(licence, terms, counts, dates, status) and `evidence/` (`requests.jsonl` plus the `robots.txt` of every
host, the proof of self-collection). Collection only extracts text; normalisation is `normalize.py`.

Self-collected (scraping or public APIs, robots.txt obeyed, 1 request/second/host, crawl-delay
honoured): ARY Urdu, Hum News, Daily Ummat, Daily Qudrat, Daily Ausaf (news and opinion columns),
Mashriq TV (news, editorials, blog), Aik Rozan (blog), Express Urdu (HTML scraper over sitemaps),
Internet Archive public-domain Urdu books. Published: Urdu Wikipedia, Wikisource and Wikibooks dumps
(wiki markup stripped by `collect/wikitext.py`) and CC-100 Urdu (streamed, stopped at its byte target).
Rejected sources and why: `uv run usage/01_collect.py --list`.

## Part A, stage 02: cleaning

Order, as in the assignment: extract (stage 01) -> normalise -> filter -> deduplicate -> split.

1. `normalize.py`: yeh/kaf/heh variants, tatweel, invisible characters and ZWNJ, digits to ASCII,
   whitespace, presentation forms (see the module docstring for every decision and its reason).
2. Line filters: drop lines with no or few Urdu letters, or mostly not Urdu (code-mixing survives);
   strip URLs/e-mails; drop `\ufffd` lines; remove any literal `<|endoftext|>`.
3. Deduplicate, sources in priority order (curated first, CC-100 last): exact duplicate documents,
   boilerplate lines (identical line in >= 30 documents), repeated long paragraphs; then repetitive
   and too-short documents. OCR sources must also pass a frequent-word check.
4. Split by document (not by line), stratified by source, 98/1/1 by bytes, fixed seed (`--seed`).
   Documents end with `<|endoftext|>`; the train file is shuffled so it is not grouped by source.

`data/clean/cleaning_stats.json` records what every step removed; `DATA_CARD.md` is generated from it.

## Part B: the tokenizer

* Vocabulary: 0-255 bytes, 256 `<|endoftext|>`, 257-49,999 = 49,743 merges in learned order.
* Pre-tokenisation (`bpe.PRETOKENIZE_PATTERN`) keeps Urdu words whole: `\p{M}` (diacritics) and
  ZWNJ/ZWJ belong to the word class, a leading space belongs to the next word, Urdu punctuation is
  its own piece. The module docstring explains each alternative; GPT-2's pattern cuts أُردُو apart.
* Tie-break: equal counts -> the greater pair compared as `(bytes_left, bytes_right)`.
* Reference trainer (`train_reference`) recounts every pair after every merge. The fast trainer adds
  (1) distinct words counted once and weighted, (2) a pair -> words index so only affected words are
  updated, with a lazy max-heap, (3) a word -> ids cache when encoding. `usage/04_verify_bpe.py`
  checks both learn identical merges; `tests/test_bpe_train.py` does so on synthetic corpora too.
* `decode(encode(text)) == text` for every string (bytes are produced with `surrogatepass`).
* Normalisation is never done inside the tokenizer.
