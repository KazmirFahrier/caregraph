from caregraph.data import guideline_corpus
from caregraph.eval.redteam import pass_rate, run_redteam
from caregraph.llm import ScriptedClient
from caregraph.rag import ClinicalRAG, Document, HashingEmbedder, InMemoryVectorStore


def build_rag() -> ClinicalRAG:
    store = InMemoryVectorStore(HashingEmbedder(dim=256))
    rag = ClinicalRAG(store=store, llm=ScriptedClient())
    rag.ingest([Document(g["doc_id"], g["text"], {}) for g in guideline_corpus()])
    return rag


def test_all_redteam_cases_pass() -> None:
    results = run_redteam(build_rag)
    failed = [r.name for r in results if not r.passed]
    assert not failed, f"red-team failures: {failed}"
    assert pass_rate(results) == 1.0


def test_injection_marker_never_surfaces() -> None:
    results = {r.name: r for r in run_redteam(build_rag)}
    assert results["prompt_injection_in_context"].passed


def test_phi_in_context_is_not_leaked() -> None:
    results = {r.name: r for r in run_redteam(build_rag)}
    assert results["phi_leakage_via_context"].passed


def test_on_topic_poisoning_is_blocked() -> None:
    # AUDIT3 A3-H1: relevance-grounding alone doesn't stop on-topic poison; the
    # trust gate must. This case would fail before the trust tier existed.
    results = {r.name: r for r in run_redteam(build_rag)}
    assert "on_topic_poisoning_untrusted" in results
    assert results["on_topic_poisoning_untrusted"].passed
