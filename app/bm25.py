"""Okapi BM25 over an in-memory list of tokenised chunks."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np


class BM25:
    def __init__(self, docs_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, tokens in enumerate(docs_tokens):
            for term, freq in Counter(tokens).items():
                self.postings[term].append((i, freq))
        self.idf = {
            term: math.log(1.0 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        result = np.zeros(self.n, dtype=np.float64)
        if not self.n or self.avgdl == 0:
            return result
        for term in set(query_tokens):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for i, freq in postings:
                norm = 1.0 - self.b + self.b * self.doc_len[i] / self.avgdl
                result[i] += idf * freq * (self.k1 + 1.0) / (freq + self.k1 * norm)
        return result
