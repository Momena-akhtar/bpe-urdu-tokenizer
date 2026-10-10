"""CC-100 Urdu (Common Crawl, CC-Net processed): streamed from the xz file, stopped at the target.

The xz file is ~884 MB compressed and several GB of text. We only need a slice, so documents are
decoded on the fly and the download stops once the byte target is reached; the partial file stays
cached and a later run with a larger target simply continues from where it stopped.
"""

from __future__ import annotations

import codecs
import lzma
from collections.abc import Iterator

from urdu_bpe.collect.base import Collector
from urdu_bpe.collect.record import Document, DocWriter
from urdu_bpe.collect.textutil import urdu_ratio

URL = "https://data.statmt.org/cc-100/ur.txt.xz"


def blank_line_documents(chunks: Iterator[bytes]) -> Iterator[str]:
    """Group decoded lines into documents separated by empty lines (the CC-100 convention)."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending, lines = "", []
    for chunk in chunks:
        pending += decoder.decode(chunk)
        *complete, pending = pending.split("\n")
        for line in complete:
            if line.strip():
                lines.append(line.strip())
            elif lines:
                yield "\n".join(lines)
                lines = []
    tail = (pending + decoder.decode(b"", final=True)).strip()
    if tail:
        lines.append(tail)
    if lines:
        yield "\n".join(lines)


class CC100(Collector):
    min_chars = 150

    def _decompressed(self) -> Iterator[bytes]:
        decomp = lzma.LZMADecompressor()
        for chunk in self.session.stream_cached(URL, self.downloads / "cc100_ur.txt.xz"):
            if decomp.eof:
                break
            yield decomp.decompress(chunk)

    def collect(self, writer: DocWriter) -> None:
        for n, text in enumerate(blank_line_documents(self._decompressed())):
            if writer.full:
                return
            doc_id = f"cc100-ur-{n}"
            if doc_id in writer:
                continue
            if len(text) < self.min_chars:
                self.skip("too_short")
            elif urdu_ratio(text) < 0.5:
                self.skip("not_urdu")
            else:
                writer.write(Document(doc_id, self.spec.name, self.spec.genre, text))
            if n % 2000 == 0:
                self.progress(writer)
        self.exhausted = True
