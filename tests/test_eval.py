from caregraph.data import guideline_corpus
from caregraph.eval import EvalCase, faithfulness, run_eval
from caregraph.llm import ScriptedClient
from caregraph.rag import ClinicalRAG, Document, HashingEmbedder, InMemoryVectorStore


def test_faithfulness_oracle() -> None:
    ctx = ["The blood pressure target is less than 130 over 80 mmHg."]
    assert faithfulness("The target is less than 130 over 80 mmHg.", ctx) == 1.0
    assert faithfulness("The patient should take aspirin daily.", ctx) < 0.5


def test_faithfulness_is_per_document_not_union() -> None:
    # AUDIT5 A5-M1: a claim whose terms are split across two docs (neither supports it
    # alone) must NOT score as faithful. Union-based scoring returned 1.0 here.
    contexts = [
        "Aspirin reduces stroke risk in eligible patients.",
        "Warfarin requires regular monitoring of levels.",
    ]
    claim = "Aspirin warfarin combination reduces monitoring."
    assert faithfulness(claim, contexts) == 0.0


def test_harness_scores_and_refusals() -> None:
    store = InMemoryVectorStore(HashingEmbedder(dim=256))
    rag = ClinicalRAG(store=store, llm=ScriptedClient())
    rag.ingest([Document(g["doc_id"], g["text"], {}) for g in guideline_corpus()])

    cases = [
        EvalCase("What BP target is recommended for CKD?", "less than 130/80 mmHg"),
        EvalCase("What is the capital of France?", "", must_refuse=True),
    ]
    result = run_eval(rag, cases)
    assert result.n == 2
    assert result.refusal_accuracy == 1.0
    assert result.faithfulness >= 0.5
