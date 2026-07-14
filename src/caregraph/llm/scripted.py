"""Deterministic LLM used for offline demos and tests.

It performs *extractive, grounded* answering: it composes its reply strictly from
the retrieved context it is given, so the faithfulness gate can be exercised without
any network calls or API keys. This mirrors the campaign-copilot ScriptedClient
discipline — the running system is testable end to end.
"""
from __future__ import annotations

import re

from caregraph.llm.base import Message


class ScriptedClient:
    """Answers the user's question using only sentences from the provided CONTEXT.

    The pipeline passes retrieved guideline text in a CONTEXT block. This client
    selects the context sentences whose keywords best overlap the question, so the
    output is always grounded — never invented.
    """

    def complete(self, messages: list[Message]) -> str:
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        context = self._extract_context(user)
        question = self._extract_question(user)
        if not context:
            return "INSUFFICIENT_CONTEXT"
        sentences = _split_sentences(context)
        q_terms = _terms(question)
        ranked = sorted(
            sentences,
            key=lambda s: len(_terms(s) & q_terms),
            reverse=True,
        )
        best = [s for s in ranked if _terms(s) & q_terms][:2]
        if not best:
            return "INSUFFICIENT_CONTEXT"
        return " ".join(best)

    @staticmethod
    def _extract_context(prompt: str) -> str:
        m = re.search(r"CONTEXT:\n(.*?)(?:\nQUESTION:|\Z)", prompt, re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_question(prompt: str) -> str:
        m = re.search(r"QUESTION:\s*(.*)", prompt, re.DOTALL)
        return m.group(1).strip() if m else prompt


_STOP = {
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "for", "with",
    "what", "which", "how", "should", "be", "on", "at", "this", "that", "patient",
}


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]
