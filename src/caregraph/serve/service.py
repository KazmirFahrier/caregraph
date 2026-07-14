"""CareGraphService — the framework-free application core.

All request logic lives here so it is unit-testable without an HTTP server. `app.py`
is a thin FastAPI shell over this class. Every call is traced (retrieve/generate
spans with latency + faithfulness) and audit-logged, matching the HIPAA posture.
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from caregraph.agent import build_agent
from caregraph.config import Settings
from caregraph.data import guideline_corpus, load_bundle, synthetic_patient
from caregraph.data.fhir import Patient
from caregraph.llm import ScriptedClient
from caregraph.llm.base import LLMClient
from caregraph.obs.tracing import InMemoryTracer
from caregraph.rag import ClinicalRAG, Document, HashingEmbedder, make_vector_store
from caregraph.rag.vectorstore import VectorStore
from caregraph.serve.pending import InMemoryPendingStore, PendingStore
from caregraph.tools import ToolServer, register_clinical_tools


@dataclass
class AskResponse:
    answer: str
    citations: list[str]
    faithfulness: float
    refused: bool
    latency_ms: float
    trace: list[dict[str, Any]] = field(default_factory=list)


class CareGraphService:
    def __init__(self, store: VectorStore, llm: LLMClient, tools: ToolServer,
                 patient: Patient, pending: PendingStore | None = None) -> None:
        # No shared tracer: each request gets its own so concurrent calls can't race
        # on tracer state (A3-M2).
        self._rag = ClinicalRAG(store=store, llm=llm)
        self._agent = build_agent(store, llm, tools, patient)
        self._tools = tools
        # Bounded, thread-safe audit ring buffer (A4-L1); emit to a log sink in prod.
        self._audit: deque[dict[str, Any]] = deque(maxlen=1000)
        # Thread-safe, TTL/size-bounded store for interrupted runs (A4-L1/L2). Swap in
        # a shared (Redis) implementation for multi-replica deployments (A4-M2).
        self._pending: PendingStore = pending or InMemoryPendingStore()

    def health(self) -> dict[str, str]:
        return {"status": "ok", "service": "caregraph"}

    def list_tools(self) -> list[dict[str, Any]]:
        return self._tools.list_tools()

    def ask(self, question: str) -> AskResponse:
        tracer = InMemoryTracer()  # per-request; not shared state
        answer = self._rag.answer(question, tracer=tracer)
        root = tracer.by_name("clinical_rag.answer")
        latency = root.duration_ms if root else 0.0
        self._audit.append({"endpoint": "ask", "question_len": len(question),
                            "refused": answer.refused, "faithfulness": answer.faithfulness})
        return AskResponse(
            answer=answer.text,
            citations=answer.citations,
            faithfulness=answer.faithfulness,
            refused=answer.refused,
            latency_ms=round(latency, 2),
            trace=[asdict(s) for s in tracer.spans],
        )

    def agent_ask(self, question: str) -> dict[str, Any]:
        tracer = InMemoryTracer()  # per-request
        raw = self._agent.run_raw(question, tracer=tracer)
        result = self._agent._to_result(raw)  # never carries the draft when interrupted
        payload: dict[str, Any] = {
            "status": result.status,
            "answer": result.answer,
            "citations": result.citations,
            "tool_calls": result.tool_calls,
            "faithfulness": result.faithfulness,
            "needs_review": result.needs_review,
            "refused": result.refused,
            "trace": [asdict(s) for s in tracer.spans],
        }
        if result.status == "interrupted":
            # Park the real draft server-side; the clinician approves via /agent/approve.
            token = uuid.uuid4().hex
            self._pending.put(token, raw)
            payload["resume_token"] = token
        self._audit.append({"endpoint": "agent_ask", "needs_review": result.needs_review,
                            "refused": result.refused, "interrupted": result.status == "interrupted"})
        return payload

    def approve(self, resume_token: str, approved: bool, approver_id: str) -> dict[str, Any]:
        """Resume an interrupted run after clinician sign-off (A2-H1).

        `approver_id` is the authenticated clinician's identity — required so the gate
        has real accountability and the audit trail records *who* signed off (A5-M2).
        """
        if not approver_id or not approver_id.strip():
            raise PermissionError("approval requires an authenticated approver_id")
        raw = self._pending.pop(resume_token)
        if raw is None:
            raise KeyError("unknown or already-used resume token")
        result = self._agent.approve(raw, approved)
        self._audit.append({"endpoint": "agent_approve", "approved": approved,
                            "approver_id": approver_id, "at": time.time(),
                            "refused": result.refused})
        return {
            "status": result.status,
            "answer": result.answer,
            "citations": result.citations,
            "tool_calls": result.tool_calls,
            "faithfulness": result.faithfulness,
            "needs_review": result.needs_review,
            "refused": result.refused,
        }

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        return list(self._audit)


def build_default_service(llm: LLMClient | None = None) -> CareGraphService:
    settings = Settings.load()
    embedder = HashingEmbedder(dim=settings.embedding_dim)
    store = make_vector_store(embedder, settings.vector_backend, settings.qdrant_url)
    store.add([Document(g["doc_id"], g["text"], {"title": g["title"]}) for g in guideline_corpus()])
    patient = load_bundle(synthetic_patient())
    server = ToolServer()
    register_clinical_tools(server, {patient.patient_id: patient})
    return CareGraphService(store, llm or ScriptedClient(), server, patient)
