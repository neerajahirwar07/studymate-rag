"""Embedding backends.

* ``HashingEmbedder``: a deterministic hashing vectorizer (stemmed unigrams + bigrams,
  sublinear term frequency, L2-normalised). It needs no model download, so the project
  runs anywhere. It is a *lexical* embedding: it matches shared vocabulary, not meaning.
* ``SentenceTransformerEmbedder``: real semantic embeddings (optional dependency).
"""
from __future__ import annotations

import math
import zlib
from collections import Counter
from typing import Protocol, Sequence

import numpy as np

from .config import Settings
from .text import analyze


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""


class HashingEmbedder:
    def __init__(self, dim: int = 1024) -> None:
        if dim < 64:
            raise ValueError("dim must be at least 64")
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _add(self, vec: np.ndarray, feature: str, weight: float) -> None:
        h = zlib.crc32(feature.encode("utf-8"))
        sign = 1.0 if (h >> 16) & 1 else -1.0
        vec[h % self.dim] += sign * weight

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = analyze(text)
            for token, count in Counter(tokens).items():
                self._add(out[row], token, 1.0 + math.log(count))
            for left, right in zip(tokens, tokens[1:]):
                self._add(out[row], f"{left}_{right}", 0.5)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "EMBEDDING_BACKEND=sentence-transformers needs: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)  # pragma: no cover
        self.dim = int(self._model.get_sentence_embedding_dimension())  # pragma: no cover
        self.name = f"st-{model_name}"  # pragma: no cover

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder(settings.embedding_dim)
    if settings.embedding_backend in {"sentence-transformers", "st"}:
        return SentenceTransformerEmbedder(settings.embedding_model)
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {settings.embedding_backend!r}")
