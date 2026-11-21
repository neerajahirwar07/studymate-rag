# Evaluation results

- Embedder: `hashing-1024`  |  LLM: `extractive`
- Corpus: 9 notes. Questions: 27 answerable, 7 unanswerable.

## Retrieval (labelled source document in top-k chunks)

| Mode | hit@1 | hit@3 | MRR@5 |
|------|-------|-------|-------|
| dense | 0.81 | 0.85 | 0.83 |
| bm25 | 0.96 | 0.96 | 0.96 |
| hybrid | 0.93 | 0.96 | 0.94 |

## End-to-end /ask

- Answer rate on answerable questions: 0.96
- Answers citing the correct note: 0.93
- Abstention rate on unanswerable questions: 0.86
- Latency: mean 0.2 ms, p95 0.3 ms

Unanswerable questions that were wrongly answered:
- What is the time complexity of quicksort?

Answerable questions that were refused:
- How is durability achieved after a commit?

hybrid retrieval misses (labelled note not in top 3):
- Why would you denormalize a database?

