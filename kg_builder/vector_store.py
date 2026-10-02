"""Lightweight cosine-similarity vector store (JSON-backed, numpy)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, path: str | Path, dim: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.texts: list[str] = []
        self.metas: list[dict] = []
        self.vectors: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                d = {}
            self.texts = d.get("texts", [])
            self.metas = d.get("metas", [])
            vecs = d.get("vectors", [])
            self.vectors = np.array(vecs, dtype=float) if vecs else None

    def _save(self) -> None:
        data = {
            "texts": self.texts,
            "metas": self.metas,
            "vectors": self.vectors.tolist() if self.vectors is not None else [],
        }
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def add(self, texts: list[str], metas: list[dict], vectors: list[list[float]]) -> None:
        arr = np.array(vectors, dtype=float)
        if self.vectors is None or self.vectors.size == 0:
            self.vectors = arr
        else:
            self.vectors = np.vstack([self.vectors, arr])
        self.texts.extend(texts)
        self.metas.extend(metas)
        self._save()

    def search(self, query_vec, top_k: int = 5) -> list[tuple[str, dict, float]]:
        if self.vectors is None or self.vectors.size == 0:
            return []
        q = np.array(query_vec, dtype=float)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        norms = np.linalg.norm(self.vectors, axis=1)
        norms[norms == 0] = 1.0
        sims = (self.vectors @ q) / norms
        k = min(top_k, len(sims))
        idx = np.argsort(-sims)[:k]
        return [(self.texts[i], self.metas[i], float(sims[i])) for i in idx]

    def delete_by_source(self, source: str) -> int:
        """Remove every chunk whose stored meta ``source`` equals ``source``.

        Returns the number of chunks removed.
        """
        keep = [i for i, m in enumerate(self.metas) if m.get("source") != source]
        removed = len(self.metas) - len(keep)
        if removed == 0:
            return 0
        self.texts = [self.texts[i] for i in keep]
        self.metas = [self.metas[i] for i in keep]
        if self.vectors is not None and self.vectors.size:
            self.vectors = self.vectors[keep]
        else:
            self.vectors = None
        self._save()
        return removed

    def clear(self) -> None:
        self.texts = []
        self.metas = []
        self.vectors = None
        self._save()
