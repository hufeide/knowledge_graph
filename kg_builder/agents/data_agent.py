"""Agent 1: 数据采集 / 清洗 / 分块 (Data Collection Agent)."""
from __future__ import annotations

from pathlib import Path

from ..schemas import Chunk
from ..utils import clean_text, decode_text_bytes
from .base import Agent
from ..config import config

# Map of common document extensions to a simple loader.
try:  # pypdf is optional; only needed for PDF input.
    from pypdf import PdfReader  # type: ignore
    _HAS_PYPDF = True
except Exception:  # pragma: no cover
    _HAS_PYPDF = False


class DataAgent(Agent):
    name = "data"

    def __init__(self, llm=None, chunk_size: int | None = None, chunk_overlap: int | None = None):
        super().__init__(llm)
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap

    # ---- loaders -------------------------------------------------------
    def load_text_file(self, path: str | Path) -> str:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md", ".text"):
            return decode_text_bytes(path.read_bytes())
        if suffix == ".pdf":
            if not _HAS_PYPDF:
                raise RuntimeError("pypdf 未安装，无法解析 PDF。请 pip install pypdf 或改用 .txt/.md。")
            reader = PdfReader(str(path))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        # attempt plain text decode as fallback
        return decode_text_bytes(path.read_bytes())

    # ---- chunking ------------------------------------------------------
    def chunk_text(self, text: str, source: str = "") -> list[Chunk]:
        text = clean_text(text)
        if not text:
            return []
        # split into sentences on common sentence terminators (zh/en)
        sentences = [s.strip() for s in _split_sentences(text) if s.strip()]
        chunks: list[Chunk] = []
        buf: list[str] = []
        buf_len = 0
        idx = 0
        for sent in sentences:
            if buf_len + len(sent) > self.chunk_size and buf:
                chunks.append(self._make_chunk(chunks, "\n".join(buf), source))
                # overlap: keep tail
                overlap_text = "\n".join(buf)[-self.chunk_overlap :]
                buf = [overlap_text] if overlap_text.strip() else []
                buf_len = len(overlap_text)
                idx += 1
            buf.append(sent)
            buf_len += len(sent)
        if buf:
            chunks.append(self._make_chunk(chunks, "\n".join(buf), source))
        return chunks

    def _make_chunk(self, existing: list, text: str, source: str) -> Chunk:
        cid = f"{source or 'doc'}-{len(existing) + 1}"
        return Chunk(id=cid, text=text, source=source)

    # ---- entry ---------------------------------------------------------
    def run(self, text: str, source: str = "document") -> list[Chunk]:
        return self.chunk_text(text, source=source)

    def run_file(self, path: str | Path, source: str | None = None) -> list[Chunk]:
        source = source or Path(path).name
        text = self.load_text_file(path)
        return self.chunk_text(text, source=source)


def _split_sentences(text: str) -> list[str]:
    import re

    # keep delimiter with the sentence
    parts = re.split(r"(?<=[。！？!?；;\.\n])", text)
    return [p for p in parts if p.strip()]
