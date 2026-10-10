from pathlib import Path

import numpy as np
import pytest

from conftest import make_text
from urdu_bpe.bpe import EOT, EOT_ID, BPETokenizer
from urdu_bpe.bpe_train import count_words_text, train_fast
from urdu_bpe.chunking import EOT_BYTES, chunk_ranges
from urdu_bpe.encode import encode_file, load_ids


@pytest.fixture
def setup(tmp_path: Path):
    text = make_text(seed=8, n_docs=60)
    corpus = tmp_path / "train.txt"
    corpus.write_text(text, encoding="utf-8")
    tok = BPETokenizer(train_fast(count_words_text(text), 200))
    tok_path = tmp_path / "tokenizer.json"
    tok.save(tok_path)
    return text, corpus, tok, tok_path


def test_chunks_cover_the_file_and_end_on_document_boundaries(setup):
    _, corpus, _, _ = setup
    data = corpus.read_bytes()
    ranges = chunk_ranges(corpus, 2000)
    assert ranges[0][0] == 0 and ranges[-1][1] == len(data)
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:], strict=False))
    assert all(data[:stop].endswith(EOT_BYTES) for _, stop in ranges)
    assert len(ranges) > 3
    limited = chunk_ranges(corpus, 2000, limit=5000)
    assert 5000 <= limited[-1][1] < len(data) and data[: limited[-1][1]].endswith(EOT_BYTES)


def test_marker_straddling_the_probe_point_is_found(tmp_path: Path):
    path = tmp_path / "t.txt"
    path.write_bytes(("اردو" + EOT + "ب" * 10 + EOT).encode())
    for probe in range(1, path.stat().st_size - 1):
        ranges = chunk_ranges(path, probe)
        assert ranges[-1][1] == path.stat().st_size
        assert all(path.read_bytes()[:stop].endswith(EOT_BYTES) for _, stop in ranges)


def test_chunked_parallel_encoding_equals_whole_file(setup, tmp_path: Path):
    text, corpus, tok, tok_path = setup
    out = tmp_path / "train.bin"
    report = encode_file(tok_path, corpus, out, workers=2, chunk_bytes=2500)
    ids = np.fromfile(out, dtype=np.uint16)
    assert ids.tolist() == tok.encode(text)
    assert report["tokens"] == len(ids) and report["roundtrip_verified"] is True
    assert int((ids == EOT_ID).sum()) == text.count(EOT) == 60
    assert tok.decode(ids.tolist()) == text
    assert load_ids(out).dtype == np.uint16 and len(load_ids(out)) == len(ids)


def test_encode_file_without_verification(setup, tmp_path: Path):
    _, corpus, _, tok_path = setup
    report = encode_file(tok_path, corpus, tmp_path / "x.bin", workers=1, verify=False)
    assert report["roundtrip_verified"] is False and report["bytes_per_token"] > 1
