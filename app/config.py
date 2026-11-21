"""Runtime settings, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    db_path: str = str(BASE_DIR / "data" / "studymate.db")
    samples_dir: str = str(BASE_DIR / "data" / "samples")

    # Embeddings: "hashing" (no downloads, runs anywhere) or "sentence-transformers".
    embedding_backend: str = "hashing"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 1024  # only used by the hashing backend

    # LLM: "auto" picks Anthropic, then OpenAI, then the built-in extractive answerer.
    llm_backend: str = "auto"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5-5"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 30.0

    # Chunking and retrieval.
    chunk_chars: int = 500
    chunk_overlap: int = 80
    min_dense_score: float = 0.35  # semantic-match gate (OR-ed with the min_coverage gate)
    min_coverage: float = 0.5

    # Service limits.
    max_doc_chars: int = 200_000
    rate_limit_per_minute: int = 30
    seed_on_start: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        backend = os.getenv("EMBEDDING_BACKEND", d.embedding_backend).lower()
        default_min_dense = d.min_dense_score
        return cls(
            db_path=os.getenv("DB_PATH", d.db_path),
            samples_dir=os.getenv("SAMPLES_DIR", d.samples_dir),
            embedding_backend=backend,
            embedding_model=os.getenv("EMBEDDING_MODEL", d.embedding_model),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", d.embedding_dim)),
            llm_backend=os.getenv("LLM_BACKEND", d.llm_backend).lower(),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            anthropic_model=os.getenv("ANTHROPIC_MODEL", d.anthropic_model),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", d.openai_model),
            llm_timeout_s=float(os.getenv("LLM_TIMEOUT_S", d.llm_timeout_s)),
            chunk_chars=int(os.getenv("CHUNK_CHARS", d.chunk_chars)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", d.chunk_overlap)),
            min_dense_score=float(os.getenv("MIN_DENSE_SCORE", default_min_dense)),
            min_coverage=float(os.getenv("MIN_COVERAGE", d.min_coverage)),
            max_doc_chars=int(os.getenv("MAX_DOC_CHARS", d.max_doc_chars)),
            rate_limit_per_minute=int(
                os.getenv("RATE_LIMIT_PER_MINUTE", d.rate_limit_per_minute)
            ),
            seed_on_start=_env_bool("SEED_ON_START", d.seed_on_start),
        )
