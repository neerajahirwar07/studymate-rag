"""Offline evaluation of retrieval quality and answer grounding.

Run from the project root:  python -m eval.run_eval

Reports, per retrieval mode (dense / bm25 / hybrid):
  hit@1, hit@3  - is the labelled source document among the top 1 / 3 chunks?
  MRR           - mean reciprocal rank of the first chunk from the labelled document (top 5)
For the end-to-end /ask pipeline:
  answer rate on answerable questions, correct citation rate, abstention rate on
  unanswerable questions, and latency.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.main import build_service  # noqa: E402
from app.rag import MODES  # noqa: E402
from app.seed import seed_samples  # noqa: E402


def evaluate(service, questions: dict) -> dict:
    answerable, unanswerable = questions["answerable"], questions["unanswerable"]
    retrieval = {}
    for mode in MODES:
        hit1 = hit3 = 0
        rr = []
        misses = []
        for item in answerable:
            hits = service.search(item["q"], k=5, mode=mode)
            titles = [h.title for h in hits]
            hit1 += bool(titles[:1] and titles[0] == item["doc"])
            hit3 += item["doc"] in titles[:3]
            rank = next((i for i, t in enumerate(titles, 1) if t == item["doc"]), None)
            rr.append(1.0 / rank if rank else 0.0)
            if rank is None or rank > 3:
                misses.append(item["q"])
        n = len(answerable)
        retrieval[mode] = {
            "hit@1": hit1 / n,
            "hit@3": hit3 / n,
            "mrr": sum(rr) / n,
            "misses": misses,
        }

    answered = cited_ok = 0
    latencies = []
    wrongly_answered = []
    unanswered_answerable = []
    for item in answerable:
        result = service.ask(item["q"])
        latencies.append(result.latency_ms)
        if result.grounded:
            answered += 1
            cited_ok += any(c.title == item["doc"] for c in result.citations)
        else:
            unanswered_answerable.append(item["q"])
    abstained = 0
    for item in unanswerable:
        result = service.ask(item["q"])
        latencies.append(result.latency_ms)
        if result.grounded:
            wrongly_answered.append(item["q"])
        else:
            abstained += 1
    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "retrieval": retrieval,
        "answer_rate": answered / len(answerable),
        "citation_correct": cited_ok / len(answerable),
        "abstention_rate": abstained / len(unanswerable),
        "wrongly_answered": wrongly_answered,
        "unanswered_answerable": unanswered_answerable,
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_p95": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
    }


def to_markdown(r: dict, service) -> str:
    lines = [
        "# Evaluation results",
        "",
        f"- Embedder: `{service.embedder.name}`  |  LLM: `{service.llm.name}`",
        f"- Corpus: {service.document_count()} notes. "
        f"Questions: {r['n_answerable']} answerable, {r['n_unanswerable']} unanswerable.",
        "",
        "## Retrieval (labelled source document in top-k chunks)",
        "",
        "| Mode | hit@1 | hit@3 | MRR@5 |",
        "|------|-------|-------|-------|",
    ]
    for mode, m in r["retrieval"].items():
        lines.append(f"| {mode} | {m['hit@1']:.2f} | {m['hit@3']:.2f} | {m['mrr']:.2f} |")
    lines += [
        "",
        "## End-to-end /ask",
        "",
        f"- Answer rate on answerable questions: {r['answer_rate']:.2f}",
        f"- Answers citing the correct note: {r['citation_correct']:.2f}",
        f"- Abstention rate on unanswerable questions: {r['abstention_rate']:.2f}",
        f"- Latency: mean {r['latency_ms_mean']:.1f} ms, p95 {r['latency_ms_p95']:.1f} ms",
        "",
    ]
    if r["wrongly_answered"]:
        lines += ["Unanswerable questions that were wrongly answered:"]
        lines += [f"- {q}" for q in r["wrongly_answered"]] + [""]
    if r["unanswered_answerable"]:
        lines += ["Answerable questions that were refused:"]
        lines += [f"- {q}" for q in r["unanswered_answerable"]] + [""]
    for mode in ("hybrid",):
        misses = r["retrieval"][mode]["misses"]
        if misses:
            lines += [f"{mode} retrieval misses (labelled note not in top 3):"]
            lines += [f"- {q}" for q in misses] + [""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(ROOT / "eval" / "questions.json"))
    parser.add_argument("--out", default=str(ROOT / "eval" / "results.md"))
    parser.add_argument("--min-hit3", type=float, default=0.0, help="exit 1 if hybrid hit@3 is below this")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DB_PATH"] = str(Path(tmp) / "eval.db")
        settings = Settings.from_env()
        service = build_service(settings)
        seed_samples(service, settings.samples_dir)
        questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
        results = evaluate(service, questions)
        report = to_markdown(results, service)
    Path(args.out).write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0 if results["retrieval"]["hybrid"]["hit@3"] >= args.min_hit3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
