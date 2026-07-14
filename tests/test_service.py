from caregraph.serve import build_default_service


def test_health() -> None:
    assert build_default_service().health()["status"] == "ok"


def test_ask_is_grounded_and_traced() -> None:
    svc = build_default_service()
    resp = svc.ask("What blood pressure target is recommended for CKD?")
    assert not resp.refused
    assert resp.citations
    assert resp.faithfulness >= 0.5
    # Trace must contain the retrieve and generate spans with recorded latency.
    names = {s["name"] for s in resp.trace}
    assert {"clinical_rag.answer", "retrieve", "generate"} <= names
    assert all(s["duration_ms"] >= 0 for s in resp.trace)


def test_agent_endpoint_runs_tools() -> None:
    svc = build_default_service()
    out = svc.agent_ask("What is the eGFR and blood pressure for this CKD patient?")
    assert out["status"] == "complete"
    assert any(c["tool"] == "fhir_lookup" for c in out["tool_calls"])


def test_audit_log_records_requests() -> None:
    svc = build_default_service()
    svc.ask("What BP target for CKD?")
    svc.agent_ask("What is the eGFR?")
    log = svc.audit_log
    assert len(log) == 2
    assert {e["endpoint"] for e in log} == {"ask", "agent_ask"}


def test_list_tools_passthrough() -> None:
    names = {t["name"] for t in build_default_service().list_tools()}
    assert "calc_egfr" in names


# --- Regression tests for AUDIT2 A2-H1 (human-in-the-loop enforced at the API) ---

def test_actionable_request_withholds_answer_over_api() -> None:
    svc = build_default_service()
    out = svc.agent_ask("Should we prescribe an ACE inhibitor and start it today?")
    assert out["status"] == "interrupted"
    assert out["needs_review"] is True
    assert out["refused"] is True
    # The recommendation must NOT leak in the interrupted response.
    assert "ACE" not in out["answer"] and "first line" not in out["answer"]
    assert out["answer"] == "Withheld pending clinician review."
    assert "resume_token" in out


def test_clinician_approval_releases_the_answer() -> None:
    svc = build_default_service()
    out = svc.agent_ask("Should we prescribe an ACE inhibitor and start it today?")
    approved = svc.approve(out["resume_token"], approved=True, approver_id="dr_smith")
    assert approved["status"] == "complete"
    assert approved["refused"] is False
    assert approved["answer"] != "Withheld pending clinician review."


def test_clinician_rejection_keeps_answer_withheld() -> None:
    svc = build_default_service()
    out = svc.agent_ask("Please prescribe and start metformin now.")
    rejected = svc.approve(out["resume_token"], approved=False, approver_id="dr_smith")
    assert rejected["refused"] is True
    assert "Withheld" in rejected["answer"]


def test_resume_token_is_single_use() -> None:
    import pytest
    svc = build_default_service()
    out = svc.agent_ask("Should we prescribe an ACE inhibitor and start it today?")
    svc.approve(out["resume_token"], approved=True, approver_id="dr_smith")
    with pytest.raises(KeyError):
        svc.approve(out["resume_token"], approved=True, approver_id="dr_smith")


# --- Regression tests for AUDIT3 ---

def test_agent_path_is_traced() -> None:
    # A3-M1: the agent flow, not just /ask, must emit spans.
    out = build_default_service().agent_ask("What is the eGFR for this CKD patient?")
    names = {s["name"] for s in out["trace"]}
    assert "agent.run" in names
    assert any(n.startswith("tool.") for n in names)


def test_per_request_traces_do_not_bleed_under_concurrency() -> None:
    # A3-M2: a per-request tracer means concurrent /ask calls keep separate traces.
    from concurrent.futures import ThreadPoolExecutor

    svc = build_default_service()
    qs = ["What blood pressure target is recommended for CKD?"] * 16
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(svc.ask, qs))
    for r in responses:
        names = [s["name"] for s in r.trace]
        # Each response has exactly its own three spans — no cross-request bleed.
        assert names.count("clinical_rag.answer") == 1
        assert set(names) == {"clinical_rag.answer", "retrieve", "generate"}


def test_pending_runs_are_evicted() -> None:
    # A3-L1: abandoned interrupted runs must not accumulate unbounded.
    svc = build_default_service()
    svc._pending.ttl_s = 0.0  # force immediate expiry
    for _ in range(4):
        svc.agent_ask("Should we prescribe an ACE inhibitor and start it for blood pressure?")
    assert len(svc._pending) <= 1


# --- Regression tests for AUDIT4 ---

def test_tool_outputs_are_phi_redacted_before_egress() -> None:
    # A4-M1: FHIR effectiveDateTime (a HIPAA identifier) must not leak via tool_calls.
    import re
    out = build_default_service().agent_ask(
        "What is the eGFR and blood pressure for this CKD patient?")
    blob = str(out["tool_calls"])
    assert not re.search(r"\d{4}-\d{2}-\d{2}", blob), "raw ISO date leaked in tool_calls"
    assert "[DATE]" in blob  # the date was redacted, not merely dropped


def test_audit_log_is_bounded() -> None:
    # A4-L1: the in-memory audit buffer must not grow without bound.
    svc = build_default_service()
    for _ in range(1200):
        svc.ask("What blood pressure target is recommended for CKD?")
    assert len(svc.audit_log) <= 1000


def test_pending_store_is_thread_safe_under_load() -> None:
    # A4-L2: concurrent puts/pops/evicts must not raise (no dict-changed-size races).
    from concurrent.futures import ThreadPoolExecutor

    svc = build_default_service()
    svc._pending.ttl_s = 0.0  # force eviction on every access to stress the path

    def cycle(_: int) -> None:
        out = svc.agent_ask("Should we prescribe an ACE inhibitor and start it for BP?")
        if "resume_token" in out:
            try:
                svc.approve(out["resume_token"], approved=True, approver_id="dr_smith")
            except KeyError:
                pass  # evicted by TTL=0 before approval — acceptable under this stress

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(cycle, range(80)))


def test_pending_store_is_pluggable() -> None:
    # A4-M2: the store is swappable (a shared/Redis impl slots in for multi-replica).
    from caregraph.serve.pending import InMemoryPendingStore, PendingStore
    assert isinstance(InMemoryPendingStore(), PendingStore)


# --- Regression tests for AUDIT5 ---

def test_approval_requires_an_approver_identity() -> None:
    # A5-M2: an anonymous approval must be rejected — the gate needs accountability.
    import pytest
    svc = build_default_service()
    out = svc.agent_ask("Should we prescribe an ACE inhibitor and start it for BP?")
    with pytest.raises(PermissionError):
        svc.approve(out["resume_token"], approved=True, approver_id="  ")


def test_approver_identity_is_recorded_in_audit() -> None:
    svc = build_default_service()
    out = svc.agent_ask("Should we prescribe an ACE inhibitor and start it for BP?")
    svc.approve(out["resume_token"], approved=True, approver_id="dr_rivera")
    entry = next(e for e in svc.audit_log if e.get("endpoint") == "agent_approve")
    assert entry["approver_id"] == "dr_rivera"
    assert "at" in entry


def test_agent_honors_settings_top_k() -> None:
    # A5-L2: the agent must read the shared retrieval knob, not hardcode it.
    from caregraph.agent import build_agent
    from caregraph.config import Settings
    from caregraph.data import load_bundle, synthetic_patient
    from caregraph.llm import ScriptedClient
    from caregraph.rag import HashingEmbedder, InMemoryVectorStore
    from caregraph.tools import ToolServer, register_clinical_tools

    patient = load_bundle(synthetic_patient())
    server = ToolServer()
    register_clinical_tools(server, {patient.patient_id: patient})
    agent = build_agent(InMemoryVectorStore(HashingEmbedder(dim=32)),
                        ScriptedClient(), server, patient)
    assert agent._top_k == Settings.load().top_k
