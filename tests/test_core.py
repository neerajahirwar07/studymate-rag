import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.bm25 import BM25
from app.chunking import chunk_text, split_sentences
from app.config import Settings
from app.embeddings import HashingEmbedder
from app.llm import AnthropicLLM, ExtractiveLLM, get_llm
from app.main import create_app
from app.seed import seed_samples
from app.text import analyze, stem


# ------------------------------------------------------------------ text / chunking
def test_stem_and_analyze_drop_stopwords():
    assert stem("sorting") == stem("sorted") == "sort"
    assert analyze("What is the time complexity of binary search?") == [
        "time", "complexity", "binary", "search",
    ]


def test_split_sentences():
    assert split_sentences("One. Two!  Three?\n\nFour") == ["One.", "Two!", "Three?", "Four"]


def test_chunks_respect_max_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_text(text, max_chars=120, overlap=40)
    assert len(chunks) > 3
    assert all(len(c) <= 120 for c in chunks)
    # consecutive chunks share a sentence
    assert set(split_sentences(chunks[0])) & set(split_sentences(chunks[1]))
    # nothing is lost
    joined = " ".join(chunks)
    for i in range(40):
        assert f"Sentence number {i} is here." in joined


def test_chunk_handles_long_sentence_and_validates():
    chunks = chunk_text("x" * 1200, max_chars=500, overlap=50)
    assert all(len(c) <= 500 for c in chunks) and len(chunks) >= 3
    with pytest.raises(ValueError):
        chunk_text("hello", max_chars=500, overlap=500)


# --------------------------------------------------------------- embeddings / bm25
def test_embeddings_are_normalised_and_deterministic():
    e = HashingEmbedder(256)
    a = e.embed(["binary search is fast", "binary search is fast"])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.allclose(a[0], a[1])
    assert e.embed([""])[0].sum() == 0  # empty text -> zero vector, no NaN


def test_similar_text_scores_higher():
    e = HashingEmbedder(1024)
    q, near, far = e.embed(["binary search complexity", "binary search runs in log n time",
                            "tcp handshake packets"])
    assert q @ near > q @ far


def test_bm25_ranks_matching_doc_first():
    docs = [analyze(t) for t in ["cats and dogs", "binary search tree", "search engines index pages"]]
    scores = BM25(docs).scores(analyze("binary search"))
    assert int(np.argmax(scores)) == 1


# ---------------------------------------------------------------------------- llm
def test_extractive_llm_cites_sentences():
    from app.models import Hit

    hit = Hit(1, 1, "T", None, 0, "Cats purr loudly. Binary search runs in O(log n) time.", 1, 1, 1, 1)
    out = ExtractiveLLM().generate("What is the complexity of binary search?", [hit])
    assert "O(log n)" in out and out.endswith("[1]")


def test_get_llm_selection():
    assert get_llm(Settings(llm_backend="auto")).name == "extractive"
    assert get_llm(Settings(llm_backend="auto", anthropic_api_key="k")).name.startswith("anthropic:")
    with pytest.raises(ValueError):
        get_llm(Settings(llm_backend="openai"))


def test_anthropic_client_parses_response():
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "secret"
        body = request.read().decode()
        assert "Context passages" in body
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Answer [1]"}]})

    from app.models import Hit

    llm = AnthropicLLM("secret", "m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    hit = Hit(1, 1, "T", None, 0, "text", 1, 1, 1, 1)
    assert llm.generate("q?", [hit]) == "Answer [1]"


# ------------------------------------------------------------------------ service
@pytest.fixture()
def settings(tmp_path):
    return Settings(db_path=str(tmp_path / "t.db"), rate_limit_per_minute=1000)


@pytest.fixture()
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        seed_samples(app.state.service, settings.samples_dir)
        yield c


def test_health_and_documents(client):
    assert client.get("/health").json()["documents"] == 9
    assert len(client.get("/documents").json()) == 9


def test_ask_is_grounded_and_cited(client):
    body = client.post("/ask", json={"question": "What is the time complexity of binary search?"}).json()
    assert body["grounded"] and "O(log n)" in body["answer"]
    assert body["citations"][0]["title"] == "Binary Search"


def test_ask_abstains_when_not_in_notes(client):
    body = client.post("/ask", json={"question": "How do I bake sourdough bread?"}).json()
    assert body["grounded"] is False and body["citations"] == []


def test_search_modes_and_validation(client):
    for mode in ("dense", "bm25", "hybrid"):
        r = client.get("/search", params={"q": "tcp handshake", "mode": mode}).json()
        assert r["results"][0]["title"] == "TCP vs UDP"
    assert client.get("/search", params={"q": "tcp", "mode": "nope"}).status_code == 422
    assert client.get("/search", params={"q": "x"}).status_code == 422
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_document_lifecycle_updates_search(client):
    created = client.post("/documents", json={"title": "Bloom Filters",
        "text": "A bloom filter is a probabilistic set. It can return false positives but never false negatives."})
    assert created.status_code == 201
    ans = client.post("/ask", json={"question": "Can a bloom filter give false negatives?"}).json()
    assert ans["grounded"] and ans["citations"][0]["title"] == "Bloom Filters"
    assert client.delete(f"/documents/{created.json()['id']}").status_code == 204
    again = client.post("/ask", json={"question": "Can a bloom filter give false negatives?"}).json()
    assert not any(c["title"] == "Bloom Filters" for c in again["citations"])
    assert client.delete("/documents/9999").status_code == 404


def test_empty_document_rejected(client):
    assert client.post("/documents", json={"title": "x", "text": "   "}).status_code == 422


def test_feedback_and_stats(client):
    qid = client.post("/ask", json={"question": "What is memoization?"}).json()["query_id"]
    assert client.post("/feedback", json={"query_id": qid, "helpful": True}).status_code == 201
    assert client.post("/feedback", json={"query_id": 99999, "helpful": True}).status_code == 404
    stats = client.get("/stats").json()
    assert stats["questions"] == 1 and stats["helpful_rate"] == 1.0 and stats["documents"] == 9


def test_llm_failure_falls_back_to_extractive(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        seed_samples(app.state.service, settings.samples_dir)

        class Broken:
            name = "broken"

            def generate(self, question, hits):
                raise RuntimeError("api down")

        app.state.service.llm = Broken()
        body = c.post("/ask", json={"question": "What is memoization?"}).json()
        assert body["grounded"] and "fallback" in body["llm"]


def test_rate_limit(tmp_path):
    app = create_app(Settings(db_path=str(tmp_path / "r.db"), rate_limit_per_minute=3))
    with TestClient(app) as c:
        codes = [c.post("/ask", json={"question": "what is memoization?"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]


def test_reembeds_when_embedder_changes(tmp_path):
    path = str(tmp_path / "e.db")
    s1 = Settings(db_path=path, embedding_dim=256)
    app1 = create_app(s1)
    with TestClient(app1):
        app1.state.service.ingest("Doc", "Binary search halves the range every step.")
    app2 = create_app(Settings(db_path=path, embedding_dim=512))
    with TestClient(app2) as c:
        hits = app2.state.service.search("binary search")
        assert hits and len(app2.state.service._load_index().matrix[0]) == 512
