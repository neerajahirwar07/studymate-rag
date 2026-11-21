"""FastAPI application."""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import Settings
from .db import Database
from .embeddings import get_embedder
from .llm import get_llm
from .rag import MODES, RAGService
from .seed import seed_samples

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("studymate")

STATIC_DIR = Path(__file__).resolve().parent / "static"


class RateLimiter:
    """Sliding-window limiter per client key, kept in memory."""

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            window = self._hits[key]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.limit:
                return False
            window.append(now)
            return True


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    source: str | None = Field(default=None, max_length=300)


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=4, ge=1, le=10)


class FeedbackIn(BaseModel):
    query_id: int
    helpful: bool
    comment: str | None = Field(default=None, max_length=500)


def build_service(settings: Settings) -> RAGService:
    service = RAGService(settings, Database(settings.db_path), get_embedder(settings), get_llm(settings))
    service.start()
    return service


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = build_service(settings)
        if settings.seed_on_start:
            added = seed_samples(app.state.service, settings.samples_dir)
            log.info("seed_on_start: added %d sample documents", added)
        log.info("ready: embedder=%s llm=%s", app.state.service.embedder.name, app.state.service.llm.name)
        yield

    app = FastAPI(title="StudyMate RAG", version="1.0.0", lifespan=lifespan)
    limiter = RateLimiter(settings.rate_limit_per_minute)

    def service() -> RAGService:
        return app.state.service

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        if request.method == "POST" and request.url.path in {"/ask", "/documents", "/feedback"}:
            # Behind a proxy (Render, nginx) the real client IP is the first X-Forwarded-For entry.
            forwarded = request.headers.get("x-forwarded-for", "")
            client = forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")
            if not limiter.allow(client):
                return JSONResponse({"detail": "Rate limit exceeded. Try again in a minute."}, status_code=429)
        return await call_next(request)

    @app.get("/health")
    def health() -> dict:
        svc = service()
        return {
            "status": "ok",
            "embedder": svc.embedder.name,
            "llm": svc.llm.name,
            "documents": svc.document_count(),
        }

    @app.get("/documents")
    def list_documents() -> list[dict]:
        return service().list_documents()

    @app.post("/documents", status_code=201)
    def add_document(body: DocumentIn) -> dict:
        try:
            return service().ingest(body.title, body.text, body.source)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: int) -> None:
        if not service().delete_document(doc_id):
            raise HTTPException(status_code=404, detail="Document not found")

    @app.get("/search")
    def search(
        q: str = Query(min_length=2, max_length=500),
        k: int = Query(default=4, ge=1, le=10),
        mode: str = Query(default="hybrid"),
    ) -> dict:
        if mode not in MODES:
            raise HTTPException(status_code=422, detail=f"mode must be one of {list(MODES)}")
        hits = service().search(q, k=k, mode=mode)
        return {"query": q, "mode": mode, "results": [h.to_dict() for h in hits]}

    @app.post("/ask")
    def ask(body: AskIn) -> dict:
        return service().ask(body.question.strip(), k=body.k).to_dict()

    @app.post("/feedback", status_code=201)
    def feedback(body: FeedbackIn) -> dict:
        if not service().add_feedback(body.query_id, body.helpful, body.comment):
            raise HTTPException(status_code=404, detail="Unknown query_id")
        return {"status": "recorded"}

    @app.get("/stats")
    def stats() -> dict:
        return service().stats()

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
