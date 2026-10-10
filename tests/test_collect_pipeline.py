import json
from pathlib import Path

from urdu_bpe.collect.record import Document, DocWriter, SourceSpec
from urdu_bpe.collect.sources.cc100 import blank_line_documents
from urdu_bpe.collect.sources.wordpress import Section, WordPressCollector

BODY = "<p>" + "یہ ایک اردو خبر ہے جو کافی لمبی ہے۔ " * 12 + "</p>"


def doc(i: int, text: str = "متن") -> Document:
    return Document(f"d{i}", "s", "news", text, url=f"https://x/{i}")


def test_writer_dedupes_counts_and_recovers_from_torn_line(tmp_path: Path):
    path = tmp_path / "docs.jsonl"
    with DocWriter(path, target_bytes=10) as w:
        assert w.write(doc(1)) and not w.write(doc(1))
        w.write(doc(2))
        assert w.full  # 2 docs x 6 bytes of UTF-8 text
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"id": "torn", "text": "ab')  # crash mid-write
    with DocWriter(path) as w:
        assert (w.docs, w.ids) == (2, {"d1", "d2"})
        w.write(doc(3))
    assert [json.loads(line)["id"] for line in path.read_text().splitlines()] == ["d1", "d2", "d3"]


def test_cc100_grouping_handles_chunk_boundaries():
    data = "الف\nب\n\nپ\n\n\nت".encode()
    chunks = (data[i : i + 3] for i in range(0, len(data), 3))  # splits multi-byte letters
    assert list(blank_line_documents(chunks)) == ["الف\nب", "پ", "ت"]


class FakeResponse:
    def __init__(self, status: int, payload: object):
        self.status_code, self._payload, self.url = status, payload, "fake"

    def json(self):
        return self._payload


class FakeSession:
    """Serves two pages of posts newest-first, then an empty page; records the params used."""

    def __init__(self):
        self.calls: list[dict] = []
        self.wire_bytes = self.requests = 0

    def get_json(self, url, params=None, **_):
        return [{"id": 7, "slug": "columns"}]

    def get(self, url, params=None, **_):
        self.calls.append(dict(params))
        post = lambda i, d: {  # noqa: E731
            "id": i,
            "date": d,
            "link": f"https://x/{i}",
            "title": {"rendered": "عنوان &amp; ذیلی"},
            "content": {"rendered": BODY},
        }
        if "before" not in params:
            return FakeResponse(
                200, [post(3, "2026-01-03T10:00:00"), post(2, "2026-01-02T10:00:00")]
            )
        if params["before"] == "2026-01-02T10:00:01":
            return FakeResponse(
                200, [post(2, "2026-01-02T10:00:00"), post(1, "2026-01-01T10:00:00")]
            )
        return FakeResponse(200, [])


def test_wordpress_collector_cursor_dedupe_and_resume(tmp_path: Path):
    spec = SourceSpec("fake", "Fake", "news", "api", "https://x/", "L", "u", "t", True, 0)
    c = WordPressCollector(
        spec, tmp_path, base_url="https://x", sections=(Section("cols", "opinion", ("columns",)),)
    )
    c.session = FakeSession()  # type: ignore[assignment]
    meta = c.run()
    assert meta["status"] == "complete" and meta["docs"] == 3  # id 2 seen twice, stored once
    rows = [
        json.loads(line) for line in (tmp_path / "fake" / "docs.jsonl").read_text().splitlines()
    ]
    assert {r["genre"] for r in rows} == {"opinion"} and rows[0]["title"] == "عنوان & ذیلی"
    assert c.session.calls[0]["categories"] == "7"
    assert json.loads((tmp_path / "fake" / "state.json").read_text())["cols.done"] is True
