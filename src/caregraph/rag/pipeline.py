"""Clinical RAG pipeline (Phase 1).

Flow: redact query PHI -> retrieve from vector store -> build a grounded prompt ->
LLM answers using only CONTEXT -> faithfulness gate -> redact output PHI -> attach
citations. Mirrors a LlamaIndex query engine but with two clinical guardrails wired
in as first-class steps rather than afterthoughts:

  1. PHI redaction on both the inbound query and the outbound answer.
  2. A faithfulness gate that refuses (rather than warns) when the answer is not
     supported by retrieved context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from caregraph.config import Settings
from caregraph.eval.ragas_metrics import faithfulness
from caregraph.llm.base import LLMClient, Message
from caregraph.obs.tracing import NoopTracer, Tracer
from caregraph.phi.redactor import PHIRedactor
from caregraph.rag.vectorstore import Document, ScoredDocument, VectorStore

_REFUSAL = "INSUFFICIENT_CONTEXT"


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    retrieved: list[ScoredDocument] = field(default_factory=list)
    faithfulness: float = 0.0
    refused: bool = False


class ClinicalRAG:
    def __init__(
        self,
        store: VectorStore,
        llm: LLMClient,
        redactor: PHIRedactor | None = None,
        settings: Settings | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._store = store
        self._llm = llm
        self._redactor = redactor or PHIRedactor()
        self._settings = settings or Settings.load()
        self._tracer = tracer or NoopTracer()

    def ingest(self, documents: list[Document]) -> None:
        # Redact before the text is ever embedded or persisted. Trust tier is preserved.
        clean = [
            Document(d.doc_id, self._redactor.redact(d.text).text, d.metadata, d.trust)
            for d in documents
        ]
        self._store.add(clean)

    def answer(self, question: str, tracer: Tracer | None = None) -> Answer:
        # A per-call tracer avoids sharing mutable tracer state across requests (A3-M2).
        t = tracer or self._tracer
        with t.span("clinical_rag.answer") as root:
            safe_question = self._redactor.redact(question).text
            with t.span("retrieve", top_k=self._settings.top_k) as rspan:
                retrieved = self._store.search(safe_question, self._settings.top_k)
                rspan.set("n_retrieved", len(retrieved))
            # Trust gate (A3-H1): only trusted docs may ground / be cited. Untrusted
            # retrieved prose is kept out of the grounding context entirely, so it can
            # neither be surfaced by the answerer nor support a clinical claim.
            trusted = [s for s in retrieved if s.document.trust == "trusted"]
            untrusted_n = len(retrieved) - len(trusted)
            root.set("untrusted_filtered", untrusted_n)
            if not trusted:
                root.set("refused", True)
                return Answer(text=_REFUSAL, retrieved=retrieved, refused=True)

            context = "\n".join(f"[{s.document.doc_id}] {s.document.text}" for s in trusted)
            prompt = f"CONTEXT:\n{context}\nQUESTION: {safe_question}"
            with t.span("generate", context_chars=len(context)):
                raw = self._llm.complete([
                    Message("system",
                            "Answer only from CONTEXT. If unsupported, say INSUFFICIENT_CONTEXT."),
                    Message("user", prompt),
                ])
            return self._finish(raw, trusted, root)

    def _finish(self, raw: str, retrieved: list[ScoredDocument], root: Any) -> Answer:
        if raw.strip() == _REFUSAL:
            root.set("refused", True)
            return Answer(text=_REFUSAL, retrieved=retrieved, refused=True)

        score = faithfulness(raw, [s.document.text for s in retrieved])
        root.set("faithfulness", round(score, 3))
        # Loud failure: a low-faithfulness answer is refused, not surfaced.
        if score < 0.5:
            root.set("refused", True)
            return Answer(text=_REFUSAL, retrieved=retrieved, faithfulness=score, refused=True)

        clean_out = self._redactor.redact(raw).text
        self._redactor.assert_clean(clean_out)  # crash rather than leak PHI
        citations = [s.document.doc_id for s in retrieved if _overlaps(raw, s.document.text)]
        if self._settings.require_citations and not citations:
            root.set("refused", True)
            return Answer(text=_REFUSAL, retrieved=retrieved, faithfulness=score, refused=True)

        root.set("refused", False)
        root.set("n_citations", len(citations))
        return Answer(
            text=clean_out,
            citations=citations,
            retrieved=retrieved,
            faithfulness=score,
        )


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


def _overlaps(answer: str, context: str, threshold: float = 0.3) -> bool:
    a = _tokens(answer)
    if not a:
        return False
    return len(a & _tokens(context)) / len(a) >= threshold
