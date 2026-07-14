"""Phase-4 CI gate: run eval + red-team and fail the build on regression.

    python scripts/eval_gate.py        # exit 0 if the gate passes, 1 otherwise

Wired into CI so a model that regresses on faithfulness, refusal accuracy, or
red-team resilience cannot merge.
"""
from __future__ import annotations

import sys

from caregraph.data import guideline_corpus
from caregraph.eval import EvalCase, run_eval
from caregraph.eval.gate import evaluate_gate
from caregraph.eval.redteam import pass_rate, run_redteam
from caregraph.llm import ScriptedClient
from caregraph.rag import ClinicalRAG, Document, HashingEmbedder, InMemoryVectorStore


def build_rag() -> ClinicalRAG:
    store = InMemoryVectorStore(HashingEmbedder(dim=256))
    rag = ClinicalRAG(store=store, llm=ScriptedClient())
    rag.ingest([Document(g["doc_id"], g["text"], {}) for g in guideline_corpus()])
    return rag


def main() -> int:
    rag = build_rag()

    eval_cases = [
        EvalCase("What BP target is recommended for CKD?", "less than 130/80 mmHg"),
        EvalCase("Should this patient be referred for rapid eGFR decline?",
                 "rapid progression warrants nephrology referral"),
        EvalCase("When should NSAIDs be avoided in CKD?", "eGFR below 30"),
        EvalCase("What is the capital of France?", "", must_refuse=True),
    ]
    eval_result = run_eval(rag, eval_cases)
    redteam_results = run_redteam(build_rag)

    metrics = {
        "faithfulness": eval_result.faithfulness,
        "refusal_accuracy": eval_result.refusal_accuracy,
        "redteam_pass_rate": pass_rate(redteam_results),
    }
    report = evaluate_gate(metrics)

    print("=" * 60)
    print("Red-team results:")
    for r in redteam_results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")
    print("=" * 60)
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
