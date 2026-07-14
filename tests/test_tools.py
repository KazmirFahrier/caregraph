import math

from caregraph.data import load_bundle, synthetic_patient
from caregraph.tools import ToolServer, register_clinical_tools


def _server() -> ToolServer:
    server = ToolServer()
    patient = load_bundle(synthetic_patient())
    register_clinical_tools(server, {patient.patient_id: patient})
    return server


def test_lists_tools_in_mcp_shape() -> None:
    tools = _server().list_tools()
    names = {t["name"] for t in tools}
    assert names == {"fhir_lookup", "problem_list", "icd10_lookup", "calc_egfr", "calc_chadsvasc"}
    assert all("inputSchema" in t for t in tools)


def test_icd10_resolves_display_variants() -> None:
    # Regression for AUDIT2 A2-M1: "Essential hypertension" must resolve, not fail.
    s = _server()
    assert s.call_tool("icd10_lookup", {"term": "Essential hypertension"}).data["icd10"] == "I10"
    assert s.call_tool("icd10_lookup", {"term": "Atrial fibrillation"}).data["icd10"] == "I48.91"
    assert not s.call_tool("icd10_lookup", {"term": "broken leg"}).ok  # honest miss


def test_problem_list_reads_codes_from_record() -> None:
    res = _server().call_tool("problem_list", {"patient_id": "synthetic-0001"})
    assert res.ok
    assert res.data["Essential hypertension"] == "I10"


def test_fhir_lookup_returns_latest_observation() -> None:
    res = _server().call_tool("fhir_lookup", {"patient_id": "synthetic-0001", "loinc": "33914-3"})
    assert res.ok
    assert res.data["value"] == 38.0


def test_unknown_tool_is_typed_error_not_exception() -> None:
    res = _server().call_tool("does_not_exist", {})
    assert not res.ok and "unknown tool" in (res.error or "")


def test_egfr_ckd_epi_2021_matches_reference() -> None:
    # Female, age 55, Scr 1.0 -> ~66.5 mL/min/1.73m2 by CKD-EPI 2021.
    res = _server().call_tool("calc_egfr", {"creatinine_mg_dl": 1.0, "age": 55, "sex": "female"})
    assert res.ok
    assert math.isclose(res.data["egfr"], 66.5, abs_tol=1.5)


def test_chadsvasc_score() -> None:
    # HTN(1) + age 69 (1) + female (1) = 3, anticoagulation recommended.
    res = _server().call_tool("calc_chadsvasc", {
        "chf": False, "hypertension": True, "age": 69, "diabetes": False,
        "stroke_history": False, "vascular_disease": False, "female": True})
    assert res.data["chadsvasc"] == 3
    assert res.data["anticoagulation_recommended"] is True
