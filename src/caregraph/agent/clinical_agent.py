"""The CareGraph clinical agent (Phase 2).

A LangGraph-style graph:

    plan -> retrieve -> act -> draft -> verify -> [human_review] -> respond

- plan     : choose which MCP tools to call and whether the request is *actionable*
             (e.g. proposing a prescription change) and therefore needs clinician sign-off
- retrieve : pull guideline/patient context from the vector store
- act      : call the chosen tools on the MCP server (one recovery attempt each)
- draft    : compose a grounded answer from context + tool results
- verify   : faithfulness gate + PHI egress gate (refuse, don't warn)
- human_review : interrupt for clinician approval on actionable answers
- respond  : finalize

The graph pauses (`status == "interrupted"`) before `human_review` on actionable
requests; the caller approves via `resume(...)`.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from caregraph.agent.graph import END, START, CompiledGraph, StateGraph
from caregraph.config import Settings
from caregraph.data.fhir import Patient
from caregraph.eval.ragas_metrics import faithfulness
from caregraph.llm.base import LLMClient, Message
from caregraph.obs.tracing import InMemoryTracer, NoopTracer, Tracer
from caregraph.phi.redactor import PHIRedactor
from caregraph.rag.vectorstore import VectorStore
from caregraph.tools.server import ToolServer

_ACTIONABLE = ("prescribe", "start ", "adjust", "increase", "discontinue", "order ", "titrate")
_WORD = re.compile(r"[a-z0-9]+")


def _overlaps(answer: str, context: str, threshold: float = 0.3) -> bool:
    a = {w for w in _WORD.findall(answer.lower()) if len(w) > 2}
    if not a:
        return False
    c = {w for w in _WORD.findall(context.lower()) if len(w) > 2}
    return len(a & c) / len(a) >= threshold


@dataclass
class AgentResult:
    status: str                 # "complete" | "interrupted"
    answer: str
    citations: list[str]
    tool_calls: list[dict[str, Any]]
    faithfulness: float
    needs_review: bool
    refused: bool
    trace: list[dict[str, Any]] = field(default_factory=list)


def _tracer_of(state: dict[str, Any]) -> Tracer:
    t = state.get("_tracer")
    return t if t is not None else NoopTracer()


WITHHELD = "Withheld pending clinician review."


class ClinicalAgent:
    def __init__(self, store: VectorStore, llm: LLMClient, tools: ToolServer,
                 patient: Patient, redactor: PHIRedactor | None = None,
                 top_k: int | None = None, as_of: date | None = None) -> None:
        self._store = store
        self._llm = llm
        self._tools = tools
        self._patient = patient
        self._patient_id = patient.patient_id
        self._redactor = redactor or PHIRedactor()
        # Honor the shared retrieval knob (A5-L2) instead of a hardcoded value.
        self._top_k = top_k if top_k is not None else Settings.load().top_k
        # Fixed evaluation date makes age (and everything derived from it)
        # deterministic; defaults to the wall clock in production (fixes A2-M2).
        self._as_of = as_of
        self._graph = self._build()

    # ---- graph construction -------------------------------------------------
    def _build(self) -> CompiledGraph:
        g = StateGraph()
        g.add_node("plan", self._plan)
        g.add_node("retrieve", self._retrieve)
        g.add_node("act", self._act)
        g.add_node("draft", self._draft)
        g.add_node("verify", self._verify)
        g.add_node("human_review", self._human_review)
        g.add_node("respond", self._respond)

        g.add_edge(START, "plan")
        g.add_edge("plan", "retrieve")
        g.add_edge("retrieve", "act")
        g.add_edge("act", "draft")
        g.add_edge("draft", "verify")
        g.add_conditional_edges("verify", self._route_after_verify)
        g.add_edge("human_review", "respond")
        g.add_edge("respond", END)
        return g.compile(interrupt_before=["human_review"])

    # ---- nodes --------------------------------------------------------------
    def _plan(self, state: dict[str, Any]) -> dict[str, Any]:
        q = self._redactor.redact(state["question"]).text.lower()
        pid = self._patient_id
        calls: list[dict[str, Any]] = []
        if any(k in q for k in ("egfr", "kidney", "renal", "gfr")):
            calls.append({"name": "fhir_lookup", "arguments": {"patient_id": pid,
                                                               "loinc": "33914-3"}})
        if any(k in q for k in ("blood pressure", "bp ", "systolic", "hypertension")):
            calls.append({"name": "fhir_lookup", "arguments": {"patient_id": pid,
                                                               "loinc": "8480-6"}})
        # calc_egfr: compute from the patient's own serum creatinine (no fabricated args).
        if any(k in q for k in ("creatinine", "calculate egfr", "estimate egfr", "ckd-epi")):
            scr = self._patient.latest("2160-0")
            if scr is not None:
                calls.append({"name": "calc_egfr", "arguments": {
                    "creatinine_mg_dl": scr.value, "age": self._age(), "sex": self._patient.gender}})
        # Coding: the FHIR Condition.code is authoritative, so prefer it. Only fall
        # back to the icd10_lookup terminology tool for conditions recorded as free
        # text without a code (fixes A2-M1: no lossy lookup when the code is present).
        if any(k in q for k in ("code", "icd", "coding", "billing")):
            calls.append({"name": "problem_list", "arguments": {"patient_id": pid}})
            for cond in self._patient.conditions:
                if not cond.code:  # only uncoded free-text findings need the terminology tool
                    calls.append({"name": "icd10_lookup", "arguments": {"term": cond.display}})
        # calc_chadsvasc: derive every input from the record, not hardcoded constants.
        if "stroke" in q or "chadsvasc" in q or "chads" in q:
            calls.append({"name": "calc_chadsvasc", "arguments": self._chadsvasc_args()})
        needs_review = any(k in q for k in _ACTIONABLE)
        return {"planned_calls": calls, "needs_review": needs_review, "question_clean": q}

    # ---- patient-derived helpers (fixes AUDIT.md M1) ------------------------
    def _age(self) -> int:
        try:
            birth_year = int(self._patient.birth_date[:4])
        except (ValueError, IndexError):
            return 0
        as_of = self._as_of or date.today()
        return max(0, as_of.year - birth_year)

    def _chadsvasc_args(self) -> dict[str, Any]:
        codes = {c.code for c in self._patient.conditions}
        displays = " ".join(c.display.lower() for c in self._patient.conditions)

        def has(term: str) -> bool:
            # Word-boundary match so "tia" does not fire on "essenTIAl", etc.
            return re.search(rf"\b{re.escape(term)}\b", displays) is not None

        return {
            "chf": has("heart failure") or "I50" in codes,
            "hypertension": has("hypertension") or "I10" in codes,
            "age": self._age(),
            "diabetes": has("diabetes") or any(c.startswith("E11") for c in codes),
            "stroke_history": has("stroke") or has("tia"),
            "vascular_disease": has("vascular"),
            "female": self._patient.gender.lower().startswith("f"),
        }

    def _retrieve(self, state: dict[str, Any]) -> dict[str, Any]:
        with _tracer_of(state).span("retrieve", top_k=self._top_k) as sp:
            scored = self._store.search(state["question_clean"], self._top_k)
            # Same trust gate as the pipeline (A3-H1): untrusted retrieved prose is
            # dropped so it cannot ground or be cited by the agent's answer either.
            trusted = [s for s in scored if s.document.trust == "trusted"]
            sp.set("n_retrieved", len(scored))
            sp.set("untrusted_filtered", len(scored) - len(trusted))
        return {"retrieved": trusted}

    def _act(self, state: dict[str, Any]) -> dict[str, Any]:
        # A tool failure is recorded as a typed result and the agent continues; a
        # blind identical retry can never recover a deterministic failure, so we
        # don't pretend to (fixes A2-L1). Meaningful recovery would repair args.
        t = _tracer_of(state)
        results: list[dict[str, Any]] = []
        for call in state["planned_calls"]:
            with t.span(f"tool.{call['name']}") as sp:
                res = self._tools.call_tool(call["name"], call["arguments"])
                sp.set("ok", res.ok)
            # Redact tool output at the boundary (fixes A4-M1): summaries carry PHI
            # (e.g. FHIR effectiveDateTime), and they flow both into the draft context
            # and back to the caller. Redact before either happens, then assert clean.
            summary = self._redactor.redact(res.summary()).text
            self._redactor.assert_clean(summary)
            results.append({"tool": call["name"], "ok": res.ok, "summary": summary})
        return {"tool_results": results}

    def _draft(self, state: dict[str, Any]) -> dict[str, Any]:
        guideline_ctx = "\n".join(
            f"[{s.document.doc_id}] {s.document.text}" for s in state["retrieved"]
        )
        tool_ctx = "\n".join(
            f"[tool:{r['tool']}] {r['summary']}" for r in state["tool_results"] if r["ok"]
        )
        context = "\n".join(c for c in (guideline_ctx, tool_ctx) if c)
        prompt = f"CONTEXT:\n{context}\nQUESTION: {state['question_clean']}"
        raw = self._llm.complete([
            Message("system", "Answer only from CONTEXT. If unsupported, say INSUFFICIENT_CONTEXT."),
            Message("user", prompt),
        ])
        return {"draft": raw, "context_texts": [s.document.text for s in state["retrieved"]]
                + [r["summary"] for r in state["tool_results"] if r["ok"]]}

    def _verify(self, state: dict[str, Any]) -> dict[str, Any]:
        with _tracer_of(state).span("verify") as sp:
            draft = state["draft"]
            if draft.strip() == "INSUFFICIENT_CONTEXT":
                sp.set("refused", True)
                return {"refused": True, "faithfulness": 0.0, "answer": "INSUFFICIENT_CONTEXT"}
            score = faithfulness(draft, state["context_texts"])
            sp.set("faithfulness", round(score, 3))
            if score < 0.5:
                sp.set("refused", True)
                return {"refused": True, "faithfulness": score, "answer": "INSUFFICIENT_CONTEXT"}
            clean = self._redactor.redact(draft).text
            self._redactor.assert_clean(clean)  # loud PHI gate
            citations = [s.document.doc_id for s in state["retrieved"]
                         if _overlaps(draft, s.document.text)]
            sp.set("refused", False)
            return {"refused": False, "faithfulness": score, "answer": clean,
                    "citations": citations}

    def _human_review(self, state: dict[str, Any]) -> dict[str, Any]:
        # Reached only after clinician approval (resume). Attach an audit note.
        if not state.get("approved"):
            return {"answer": "Withheld pending clinician review.", "refused": True}
        return {"review_note": "clinician-approved"}

    def _respond(self, state: dict[str, Any]) -> dict[str, Any]:
        return {}

    # ---- routing ------------------------------------------------------------
    def _route_after_verify(self, state: dict[str, Any]) -> str:
        if state.get("refused"):
            return "respond"
        if state.get("needs_review"):
            return "human_review"
        return "respond"

    # ---- public API ---------------------------------------------------------
    def run(self, question: str, tracer: Tracer | None = None) -> AgentResult:
        t = tracer or NoopTracer()
        with t.span("agent.run"):
            final = self._graph.invoke({"question": question, "_tracer": t})
        return self._to_result(final, t)

    def approve(self, interrupted_state: dict[str, Any], approved: bool = True) -> AgentResult:
        final = self._graph.resume(interrupted_state, approved)
        return self._to_result(final)

    def run_raw(self, question: str, tracer: Tracer | None = None) -> dict[str, Any]:
        """Return the raw (possibly interrupted) state for approval flows."""
        t = tracer or NoopTracer()
        with t.span("agent.run"):
            return self._graph.invoke({"question": question, "_tracer": t})

    def _to_result(self, state: dict[str, Any], tracer: Tracer | None = None) -> AgentResult:
        interrupted = state.get("status") == "interrupted"
        # Defense in depth (fixes A2-H1): an interrupted, not-yet-approved answer must
        # never carry the drafted recommendation out of the agent. The real draft
        # stays in the raw state (via run_raw) for the approval flow only.
        answer = WITHHELD if interrupted else state.get("answer", "")
        trace = [asdict(s) for s in tracer.spans] if isinstance(tracer, InMemoryTracer) else []
        return AgentResult(
            status=state.get("status", "complete"),
            answer=answer,
            citations=[] if interrupted else state.get("citations", []),
            tool_calls=state.get("tool_results", []),
            faithfulness=state.get("faithfulness", 0.0),
            needs_review=bool(state.get("needs_review")),
            refused=bool(state.get("refused")) or interrupted,
            trace=trace,
        )


def build_agent(store: VectorStore, llm: LLMClient, tools: ToolServer,
                patient: Patient, as_of: date | None = None) -> ClinicalAgent:
    return ClinicalAgent(store=store, llm=llm, tools=tools, patient=patient, as_of=as_of)
