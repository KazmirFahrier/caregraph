from caregraph.agent import build_agent
from caregraph.data import guideline_corpus, load_bundle, synthetic_patient
from caregraph.llm import ScriptedClient
from caregraph.rag import Document, HashingEmbedder, InMemoryVectorStore
from caregraph.tools import ToolServer, register_clinical_tools


def _agent():
    store = InMemoryVectorStore(HashingEmbedder(dim=256))
    store.add([Document(g["doc_id"], g["text"], {}) for g in guideline_corpus()])
    patient = load_bundle(synthetic_patient())
    server = ToolServer()
    register_clinical_tools(server, {patient.patient_id: patient})
    return build_agent(store, ScriptedClient(), server, patient)


def test_agent_answers_and_calls_tools() -> None:
    result = _agent().run("What is the eGFR and is kidney function concerning?")
    assert result.status == "complete"
    assert not result.refused
    assert any(tc["tool"] == "fhir_lookup" for tc in result.tool_calls)
    assert result.faithfulness >= 0.5


def test_actionable_request_pauses_for_human_review() -> None:
    agent = _agent()
    state = agent.run_raw("Should we prescribe an ACE inhibitor to lower blood pressure?")
    assert state["status"] == "interrupted"          # withheld pending sign-off
    assert state["needs_review"] is True

    approved = agent.approve(state, approved=True)
    assert approved.status == "complete"
    assert not approved.refused


def test_out_of_scope_is_refused() -> None:
    result = _agent().run("What is the weather in Paris today?")
    assert result.refused


# --- Regression tests for AUDIT.md M1/M2 ---

def test_agent_invokes_egfr_and_icd10_tools() -> None:
    # Previously registered-but-never-called (M2). Now wired into planning.
    egfr = _agent().run("Calculate the eGFR from creatinine for this patient.")
    assert any(c["tool"] == "calc_egfr" and c["ok"] for c in egfr.tool_calls)

    codes = _agent().run("What ICD code applies to this patient's conditions?")
    assert any(c["tool"] == "icd10_lookup" and c["ok"] for c in codes.tool_calls)


def test_chadsvasc_is_derived_from_patient_not_hardcoded() -> None:
    from datetime import date

    from caregraph.agent import build_agent
    from caregraph.data.fhir import Condition, Patient
    from caregraph.llm import ScriptedClient
    from caregraph.rag import HashingEmbedder, InMemoryVectorStore
    from caregraph.tools import ToolServer, register_clinical_tools

    # Fixed as_of date makes age deterministic (AUDIT2 A2-M2): no calendar time-bomb.
    as_of = date(2026, 1, 1)

    def agent_for(patient: Patient):
        store = InMemoryVectorStore(HashingEmbedder(dim=64))
        server = ToolServer()
        register_clinical_tools(server, {patient.patient_id: patient})
        return build_agent(store, ScriptedClient(), server, patient, as_of=as_of)

    young_healthy = Patient(patient_id="p1", given="A", family="B",
                            birth_date="1990-01-01", gender="male")
    older_htn = Patient(patient_id="p2", given="C", family="D", birth_date="1957-01-01",
                        gender="female", conditions=[Condition("I10", "Essential hypertension")])

    a1 = agent_for(young_healthy).run("What is the stroke risk?")
    a2 = agent_for(older_htn).run("What is the stroke risk?")
    s1 = next(c["summary"] for c in a1.tool_calls if c["tool"] == "calc_chadsvasc")
    s2 = next(c["summary"] for c in a2.tool_calls if c["tool"] == "calc_chadsvasc")
    # Young healthy male -> 0; older hypertensive female -> 3. Different => derived.
    assert "chadsvasc=0" in s1
    assert "chadsvasc=3" in s2
