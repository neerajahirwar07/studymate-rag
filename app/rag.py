"""Core service: ingest documents, hybrid search (dense + BM25), grounded answering."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from .bm25 import BM25
from .chunking import chunk_text
from .config import Settings
from .db import Database
from .embeddings import Embedder
from .llm import LLM, ExtractiveLLM
from .models import AskResult, Hit
from .text import analyze

log = logging.getLogger("studymate")

NOT_FOUND_MESSAGE = "I could not find this in the uploaded notes."
RRF_K = 60  # reciprocal rank fusion constant
MODES = ("dense", "bm25", "hybrid")


@dataclass
class _Index:
    chunk_ids: list[int]
    document_ids: list[int]
    titles: list[str]
    sources: list[str | None]
    chunk_indexes: list[int]
    texts: list[str]
    token_sets: list[frozenset[str]]
    matrix: np.ndarray
    bm25: BM25


class RAGService:
    def __init__(self, settings: Settings, db: Database, embedder: Embedder, llm: LLM) -> None:
        self.settings = settings
        self.db = db
        self.embedder = embedder
        self.llm = llm
        self._index: _Index | None = None
        self._lock = threading.RLock()
        self._fallback = ExtractiveLLM()

    # ------------------------------------------------------------------ setup
    def start(self) -> None:
        """Create tables and re-embed stored chunks if the embedding model changed."""
        self.db.init()
        stored = self.db.get_meta("embedder")
        if stored and stored != self.embedder.name:
            log.info("Embedder changed (%s -> %s); re-embedding chunks", stored, self.embedder.name)
            self._reembed_all()
        self.db.set_meta("embedder", self.embedder.name)

    def _reembed_all(self) -> None:
        with self.db.conn() as c:
            rows = c.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
            if not rows:
                return
            vectors = self.embedder.embed([r["text"] for r in rows])
            c.executemany(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                [(v.astype(np.float32).tobytes(), r["id"]) for v, r in zip(vectors, rows)],
            )
        self._index = None

    # -------------------------------------------------------------- documents
    def ingest(self, title: str, text: str, source: str | None = None) -> dict:
        title, text = title.strip(), text.strip()
        if not title or not text:
            raise ValueError("title and text must not be empty")
        if len(text) > self.settings.max_doc_chars:
            raise ValueError(f"text exceeds {self.settings.max_doc_chars} characters")
        chunks = chunk_text(text, self.settings.chunk_chars, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError("document produced no chunks")
        vectors = self.embedder.embed(chunks).astype(np.float32)
        with self.db.conn() as c:
            cur = c.execute(
                "INSERT INTO documents(title, source, char_count) VALUES (?, ?, ?)",
                (title, source, len(text)),
            )
            doc_id = int(cur.lastrowid)
            c.executemany(
                "INSERT INTO chunks(document_id, chunk_index, text, embedding) VALUES (?, ?, ?, ?)",
                [(doc_id, i, t, v.tobytes()) for i, (t, v) in enumerate(zip(chunks, vectors))],
            )
        with self._lock:
            self._index = None
        return {"id": doc_id, "title": title, "chunks": len(chunks), "chars": len(text)}

    def delete_document(self, doc_id: int) -> bool:
        with self.db.conn() as c:
            cur = c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            deleted = cur.rowcount > 0
        if deleted:
            with self._lock:
                self._index = None
        return deleted

    def list_documents(self) -> list[dict]:
        with self.db.conn() as c:
            rows = c.execute(
                """
                SELECT d.id, d.title, d.source, d.char_count, d.created_at,
                       COUNT(ch.id) AS chunks
                FROM documents d LEFT JOIN chunks ch ON ch.document_id = d.id
                GROUP BY d.id ORDER BY d.id
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def document_count(self) -> int:
        with self.db.conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    # ------------------------------------------------------------------ index
    def _load_index(self) -> _Index | None:
        with self._lock:
            if self._index is not None:
                return self._index
            with self.db.conn() as c:
                rows = c.execute(
                    """
                    SELECT ch.id, ch.document_id, ch.chunk_index, ch.text, ch.embedding,
                           d.title, d.source
                    FROM chunks ch JOIN documents d ON d.id = ch.document_id
                    ORDER BY ch.id
                    """
                ).fetchall()
            if not rows:
                return None
            texts = [r["text"] for r in rows]
            tokens = [analyze(t) for t in texts]
            matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            self._index = _Index(
                chunk_ids=[r["id"] for r in rows],
                document_ids=[r["document_id"] for r in rows],
                titles=[r["title"] for r in rows],
                sources=[r["source"] for r in rows],
                chunk_indexes=[r["chunk_index"] for r in rows],
                texts=texts,
                token_sets=[frozenset(t) for t in tokens],
                matrix=matrix,
                bm25=BM25(tokens),
            )
            return self._index

    # ----------------------------------------------------------------- search
    def search(self, query: str, k: int = 4, mode: str = "hybrid") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        index = self._load_index()
        query_tokens = analyze(query)
        if index is None or not query_tokens:
            return []
        dense = index.matrix @ self.embedder.embed([query])[0]
        sparse = index.bm25.scores(query_tokens)

        if mode == "dense":
            ranking_scores = dense
        elif mode == "bm25":
            ranking_scores = sparse
        else:
            ranking_scores = np.zeros(len(dense))
            for scores in (dense, sparse):
                order = np.argsort(-scores, kind="stable")
                for rank, i in enumerate(order, start=1):
                    if scores[i] > 0:
                        ranking_scores[i] += 1.0 / (RRF_K + rank)

        order = np.argsort(-ranking_scores, kind="stable")[: max(k, 1)]
        query_terms = set(query_tokens)
        hits = []
        for i in order:
            if ranking_scores[i] <= 0:
                continue
            coverage = len(query_terms & index.token_sets[i]) / len(query_terms)
            hits.append(
                Hit(
                    chunk_id=index.chunk_ids[i],
                    document_id=index.document_ids[i],
                    title=index.titles[i],
                    source=index.sources[i],
                    chunk_index=index.chunk_indexes[i],
                    text=index.texts[i],
                    score=float(ranking_scores[i]),
                    dense_score=float(dense[i]),
                    bm25_score=float(sparse[i]),
                    coverage=coverage,
                )
            )
        return hits

    def is_relevant(self, hit: Hit) -> bool:
        """A chunk counts as evidence if it shares enough of the question's terms (lexical
        match) or is strongly similar in embedding space (semantic match)."""
        return (
            hit.coverage >= self.settings.min_coverage
            or hit.dense_score >= self.settings.min_dense_score
        )

    # -------------------------------------------------------------------- ask
    def ask(self, question: str, k: int = 4) -> AskResult:
        started = time.perf_counter()
        hits = [h for h in self.search(question, k=k, mode="hybrid") if self.is_relevant(h)]
        answer, llm_name, grounded = NOT_FOUND_MESSAGE, "none", False
        if hits:
            llm_name = self.llm.name
            try:
                generated = self.llm.generate(question, hits)
            except Exception as exc:  # network error, bad key, rate limit...
                log.warning("LLM call failed (%s); using extractive fallback", exc)
                generated = self._fallback.generate(question, hits)
                llm_name = f"{self._fallback.name} (fallback)"
            if generated.strip():
                answer, grounded = generated.strip(), True
            else:
                hits = []
        latency_ms = (time.perf_counter() - started) * 1000
        top_score = hits[0].dense_score if hits else None
        with self.db.conn() as c:
            cur = c.execute(
                "INSERT INTO queries(question, answer, llm, grounded, top_score, latency_ms) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (question, answer, llm_name, int(grounded), top_score, latency_ms),
            )
            query_id = int(cur.lastrowid)
        return AskResult(
            answer=answer,
            citations=hits if grounded else [],
            query_id=query_id,
            latency_ms=latency_ms,
            llm=llm_name,
            grounded=grounded,
        )

    # --------------------------------------------------------------- feedback
    def add_feedback(self, query_id: int, helpful: bool, comment: str | None) -> bool:
        with self.db.conn() as c:
            exists = c.execute("SELECT 1 FROM queries WHERE id = ?", (query_id,)).fetchone()
            if not exists:
                return False
            c.execute(
                "INSERT INTO feedback(query_id, helpful, comment) VALUES (?, ?, ?)",
                (query_id, int(helpful), (comment or "").strip()[:500] or None),
            )
        return True

    def stats(self) -> dict:
        with self.db.conn() as c:
            totals = c.execute(
                """
                SELECT COUNT(*) AS questions,
                       COALESCE(SUM(grounded), 0) AS grounded,
                       COALESCE(AVG(latency_ms), 0) AS avg_latency_ms
                FROM queries
                """
            ).fetchone()
            fb = c.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(helpful), 0) AS helpful FROM feedback"
            ).fetchone()
            top = c.execute(
                """
                SELECT question, COUNT(*) AS times FROM queries
                GROUP BY lower(question) ORDER BY times DESC, id DESC LIMIT 5
                """
            ).fetchall()
            chunks = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {
            "documents": self.document_count(),
            "chunks": int(chunks),
            "questions": int(totals["questions"]),
            "grounded_rate": round(totals["grounded"] / totals["questions"], 3)
            if totals["questions"]
            else None,
            "avg_latency_ms": round(totals["avg_latency_ms"], 2),
            "feedback_count": int(fb["n"]),
            "helpful_rate": round(fb["helpful"] / fb["n"], 3) if fb["n"] else None,
            "top_questions": [dict(r) for r in top],
        }
