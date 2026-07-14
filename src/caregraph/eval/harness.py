"""Evaluation harness.

Runs a labeled clinical-QA set through `ClinicalRAG` and scores each answer with the
Ragas-style metrics. Returns aggregate scores plus per-case detail. Phase 4 wires
this into CI as a regression gate (build fails if faithfulness drops below threshold)
and adds an adversarial red-team split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from caregraph.eval.ragas_metrics import answer_relevancy, context_precision, faithfulness

if TYPE_CHECKING:  # avoid an import cycle: pipeline imports this package's metrics
    from caregraph.rag.pipeline import ClinicalRAG


@dataclass
class EvalCase:
    question: str
    ground_truth: str
    must_refuse: bool = False  # adversarial / out-of-scope cases should be refused


@dataclass
class EvalResult:
    n: int
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    refusal_accuracy: float
    per_case: list[dict[str, object]] = field(default_factory=list)


def run_eval(rag: ClinicalRAG, cases: list[EvalCase]) -> EvalResult:
    faith: list[float] = []
    rel: list[float] = []
    prec: list[float] = []
    refusal_correct = 0
    per_case: list[dict[str, object]] = []

    for case in cases:
        ans = rag.answer(case.question)
        contexts = [s.document.text for s in ans.retrieved]

        if case.must_refuse:
            correct = ans.refused
            refusal_correct += int(correct)
            per_case.append(
                {"question": case.question, "refused": ans.refused, "expected_refuse": True,
                 "correct": correct}
            )
            continue

        refusal_correct += int(not ans.refused)
        f = faithfulness(ans.text, contexts)
        r = answer_relevancy(ans.text, case.question)
        p = context_precision(contexts, case.ground_truth)
        faith.append(f)
        rel.append(r)
        prec.append(p)
        per_case.append(
            {"question": case.question, "answer": ans.text, "citations": ans.citations,
             "faithfulness": round(f, 3), "answer_relevancy": round(r, 3),
             "context_precision": round(p, 3)}
        )

    return EvalResult(
        n=len(cases),
        faithfulness=_mean(faith),
        answer_relevancy=_mean(rel),
        context_precision=_mean(prec),
        refusal_accuracy=refusal_correct / len(cases) if cases else 0.0,
        per_case=per_case,
    )


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
