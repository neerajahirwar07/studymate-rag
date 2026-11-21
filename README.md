# StudyMate RAG

A retrieval-augmented question-answering service for study notes. You add notes, ask a question, and get an answer that cites the passages it came from. If the notes do not contain the answer, it says so instead of guessing.

**Stack:** Python, FastAPI, SQLite (SQL), NumPy, httpx, Docker. Optional LLM: Anthropic or OpenAI over plain HTTPS.

## How it works

```
note --> sentence-aware chunks --> embeddings --> SQLite (documents, chunks, queries, feedback)

question --> dense search (cosine) ─┐
                                     ├─> reciprocal rank fusion --> relevance gate --> LLM --> answer + citations
         --> BM25 keyword search ───┘                                    │
                                                                         └─> nothing relevant: "could not find this in the notes"
```

- **Chunking:** sentence-aware, 500 chars with 80 chars overlap (`app/chunking.py`).
- **Embeddings:** by default a deterministic hashing vectorizer (stemmed unigrams and bigrams). It needs no model download, so the project runs anywhere. It is lexical, not semantic. For real semantic embeddings set `EMBEDDING_BACKEND=sentence-transformers` and `pip install sentence-transformers`. Stored chunks are re-embedded automatically when the embedder changes.
- **Hybrid search:** dense cosine and BM25 rankings are merged with reciprocal rank fusion (`app/rag.py`).
- **Grounding gate:** a chunk is only used as evidence if it covers at least 50% of the question's terms or has a high embedding similarity. If nothing passes, the service abstains.
- **Answering:** with `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` set, a hosted model answers from the numbered passages only (the prompt treats note text as data, not instructions). With no key, a built-in extractive answerer returns the best-matching sentences with citations. If the hosted call fails, the service falls back to the extractive answerer.
- **SQL:** documents, chunks, a query log and user feedback live in SQLite; `/stats` runs aggregate queries over them (grounded rate, helpful rate, latency, top questions).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
SEED_ON_START=true uvicorn app.main:app --reload
# open http://localhost:8000   (API docs at /docs)
```

Docker:

```bash
docker build -t studymate-rag .
docker run -p 8000:8000 -v studymate-data:/data studymate-rag
# or: docker compose up --build
```

Tests and evaluation:

```bash
pytest -q
python -m eval.run_eval          # writes eval/results.md
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/ask` | `{"question": "...", "k": 4}` returns answer, citations, `grounded`, latency |
| GET | `/search?q=...&mode=hybrid\|dense\|bm25` | raw retrieval results with scores |
| POST | `/documents` | add a note `{"title", "text"}` |
| GET / DELETE | `/documents`, `/documents/{id}` | list or remove notes |
| POST | `/feedback` | `{"query_id", "helpful", "comment"}` |
| GET | `/stats` | usage and quality aggregates |
| GET | `/health` | status, embedder, LLM, document count |

POST endpoints are rate limited per client IP (`RATE_LIMIT_PER_MINUTE`, default 30).

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `DB_PATH` | `data/studymate.db` | SQLite file |
| `SEED_ON_START` | `false` | load `data/samples` into an empty database |
| `LLM_BACKEND` | `auto` | `auto`, `extractive`, `anthropic`, `openai` |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | none, `claude-sonnet-5-5` | hosted LLM |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | none, `gpt-4o-mini` | hosted LLM |
| `EMBEDDING_BACKEND` | `hashing` | `hashing` or `sentence-transformers` |
| `MIN_COVERAGE`, `MIN_DENSE_SCORE` | `0.5`, `0.35` | grounding gate |
| `RATE_LIMIT_PER_MINUTE` | `30` | per-IP limit on POST endpoints |

## Evaluation

`eval/questions.json` has 27 answerable questions labelled with their source note and 7 unanswerable ones. `python -m eval.run_eval` reports retrieval quality per mode, citation correctness, abstention, and latency. Results with the default settings (hashing embedder, extractive answerer) on the 9 sample notes:

| Mode | hit@1 | hit@3 | MRR@5 |
|------|-------|-------|-------|
| dense | 0.81 | 0.85 | 0.83 |
| bm25 | 0.96 | 0.96 | 0.96 |
| hybrid | 0.93 | 0.96 | 0.94 |

End to end: 0.96 of answerable questions answered, 0.93 cited the correct note, and 0.86 of unanswerable questions were refused (6 of 7). Full output, including the failures, is in `eval/results.md`.

Read these numbers with care. The corpus is small, and the questions were written by the project author, so they share vocabulary with the notes. BM25 beats hybrid here because the hashing embedder adds no semantic signal on top of keywords; hybrid is expected to help once real sentence embeddings are used. Known failures: "time complexity of quicksort" is answered from unrelated notes because "time" and "complexity" match, and one paraphrased question ("how is durability achieved after a commit") is refused.

## Deploy (Render, Docker)

1. Push this folder to a GitHub repository.
2. On Render: New, Web Service, connect the repo, choose **Docker** as the runtime.
3. Add environment variables: `SEED_ON_START=true`, and optionally `ANTHROPIC_API_KEY`.
4. Add a persistent disk mounted at `/data` if you want notes and feedback to survive restarts (free instances lose the SQLite file on redeploy).
5. Open the service URL, check `/health`, then ask a question in the UI.

## Limitations

- In-memory index: all chunks are loaded into RAM, which is fine for thousands of chunks, not millions. For more, move vectors to pgvector or FAISS.
- SQLite plus an in-memory rate limiter assume a single instance.
- No authentication; anyone with the URL can add or delete notes. Add auth before sharing publicly.
- The Docker image was not built in the environment where this was written (no Docker daemon available); the Dockerfile follows standard practice but build it once locally to confirm.
- The sentence-transformers backend and the hosted LLM clients were not exercised against live services; the Anthropic client is covered by a mocked-transport unit test.
