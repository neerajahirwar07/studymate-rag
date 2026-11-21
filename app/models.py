"""Plain data containers shared across modules."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    title: str
    source: str | None
    chunk_index: int
    text: str
    score: float  # the score used for ranking in the chosen mode
    dense_score: float  # cosine similarity
    bm25_score: float
    coverage: float  # fraction of query terms present in this chunk

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("score", "dense_score", "bm25_score", "coverage"):
            data[key] = round(float(data[key]), 4)
        return data


@dataclass
class AskResult:
    answer: str
    citations: list[Hit] = field(default_factory=list)
    query_id: int | None = None
    latency_ms: float = 0.0
    llm: str = ""
    grounded: bool = False

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [h.to_dict() for h in self.citations],
            "query_id": self.query_id,
            "latency_ms": round(self.latency_ms, 2),
            "llm": self.llm,
            "grounded": self.grounded,
        }
