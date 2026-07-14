"""Ragas adapter.

Uses the real `ragas` package when the `eval` extra is installed (LLM-judged
faithfulness / answer relevancy / context precision); otherwise falls back to the
deterministic offline metrics in `ragas_metrics.py`. Callers get the same dict shape
either way, so CI runs offline and a nightly job can run the LLM-judged version.
"""
from __future__ import annotations

from dataclasses import dataclass

from caregraph.eval import ragas_metrics


@dataclass
class Sample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


def evaluate(samples: list[Sample]) -> dict[str, float]:
    try:  # pragma: no cover - only with the eval extra + an API key
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        ds = Dataset.from_list([
            {"question": s.question, "answer": s.answer, "contexts": s.contexts,
             "ground_truth": s.ground_truth}
            for s in samples
        ])
        scores = ragas_evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
        return {k: float(v) for k, v in scores.items()}
    except ImportError:
        return _offline(samples)


def _offline(samples: list[Sample]) -> dict[str, float]:
    if not samples:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}
    n = len(samples)
    return {
        "faithfulness": sum(ragas_metrics.faithfulness(s.answer, s.contexts) for s in samples) / n,
        "answer_relevancy": sum(
            ragas_metrics.answer_relevancy(s.answer, s.question) for s in samples) / n,
        "context_precision": sum(
            ragas_metrics.context_precision(s.contexts, s.ground_truth or s.answer)
            for s in samples) / n,
    }
