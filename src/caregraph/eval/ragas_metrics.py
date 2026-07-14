"""Ragas-style RAG metrics, deterministic and offline.

These reproduce the *shape* of the Ragas metrics (faithfulness, answer relevancy,
context precision) with token-overlap heuristics so the eval harness runs in CI with
no LLM judge. Phase 4 replaces the internals with the real `ragas` package + an LLM
judge, keeping these as the reproducible oracle that must agree on known-answer cases.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "for", "with", "that"}


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Share of answer sentences supported by a *single* retrieved context.

    Support is measured per-document (max over contexts), not against the union of all
    context terms. Union-based support let an answer recombine terms from different
    documents and score as grounded even when no single source backed the claim
    (AUDIT5 A5-M1); requiring one document to carry the sentence closes that.
    """
    sents = _sentences(answer)
    if not sents:
        return 0.0
    ctx_term_sets = [_terms(c) for c in contexts]
    supported = 0
    for s in sents:
        st = _terms(s)
        if st and any(len(st & ct) / len(st) >= 0.5 for ct in ctx_term_sets):
            supported += 1
    return supported / len(sents)


def answer_relevancy(answer: str, question: str) -> float:
    """Overlap between answer and question terms (proxy for relevancy)."""
    q, a = _terms(question), _terms(answer)
    if not q:
        return 0.0
    return len(q & a) / len(q)


def context_precision(contexts: list[str], ground_truth: str) -> float:
    """Fraction of retrieved contexts that are relevant to the ground truth."""
    if not contexts:
        return 0.0
    gt = _terms(ground_truth)
    relevant = sum(1 for c in contexts if _terms(c) & gt)
    return relevant / len(contexts)
