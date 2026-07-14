"""End-to-end Phase-1 demo. Runs offline, no API keys.

    python scripts/demo.py
"""
from __future__ import annotations

from caregraph.data import guideline_corpus, load_bundle, synthetic_patient
from caregraph.eval import EvalCase, run_eval
from caregraph.llm import ScriptedClient
from caregraph.rag import ClinicalRAG, Document, HashingEmbedder, InMemoryVectorStore


def build_rag() -> ClinicalRAG:
    embedder = HashingEmbedder(dim=256)
    store = InMemoryVectorStore(embedder)
    rag = ClinicalRAG(store=store, llm=ScriptedClient())

    # Knowledge base: public guideline snippets + this patient's structured facts.
    docs = [Document(g["doc_id"], g["text"], {"title": g["title"]}) for g in guideline_corpus()]
    patient = load_bundle(synthetic_patient())
    egfr = patient.latest("33914-3")
    sbp = patient.latest("8480-6")
    docs.append(
        Document(
            f"patient/{patient.patient_id}",
            f"Patient {patient.patient_id} has active CKD stage 3 and essential "
            f"hypertension. The most recent eGFR is {egfr.value} {egfr.unit} and the "
            f"most recent systolic blood pressure is {sbp.value} {sbp.unit}.",
            {"type": "patient_summary"},
        )
    )
    rag.ingest(docs)
    return rag


def main() -> None:
    rag = build_rag()

    print("=" * 72)
    print("CareGraph — Phase 1 clinical RAG demo")
    print("=" * 72)

    questions = [
        "What blood pressure target is recommended for this patient with CKD?",
        "Is this patient's kidney function progressing rapidly and should they be referred?",
        "Are NSAIDs safe to prescribe given the eGFR?",
    ]
    for q in questions:
        ans = rag.answer(q)
        print(f"\nQ: {q}")
        print(f"A: {ans.text}")
        print(f"   citations={ans.citations}  faithfulness={ans.faithfulness:.2f}  refused={ans.refused}")

    # PHI guardrail demonstration.
    leaky = rag.answer("Patient SSN 123-45-6789 wants BP target for CKD")
    print("\nPHI test (query contained an SSN):")
    print(f"A: {leaky.text}  (query was redacted before retrieval)")

    # Eval harness.
    cases = [
        EvalCase(
            "What blood pressure target is recommended for CKD?",
            "less than 130/80 mmHg",
        ),
        EvalCase(
            "Should this patient be referred to nephrology for rapid eGFR decline?",
            "rapid progression warrants nephrology referral",
        ),
        EvalCase("What is the capital of France?", "", must_refuse=True),
    ]
    result = run_eval(rag, cases)
    print("\n" + "=" * 72)
    print("Eval harness (Ragas-style):")
    print(f"  n={result.n}")
    print(f"  faithfulness       = {result.faithfulness:.3f}")
    print(f"  answer_relevancy   = {result.answer_relevancy:.3f}")
    print(f"  context_precision  = {result.context_precision:.3f}")
    print(f"  refusal_accuracy   = {result.refusal_accuracy:.3f}")


if __name__ == "__main__":
    main()
