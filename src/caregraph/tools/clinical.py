"""Clinical tools exposed over the MCP-style server.

Four deterministic, auditable tools:
  - fhir_lookup       : structured patient facts from the FHIR store
  - icd10_lookup      : map a clinical term to an ICD-10 code
  - calc_egfr         : CKD-EPI 2021 (race-free) creatinine eGFR
  - calc_chadsvasc    : CHA2DS2-VASc stroke-risk score

Calculators implement published equations so results are checkable, not model-guessed.
"""
from __future__ import annotations

import re
from typing import Any

from caregraph.data.fhir import Patient
from caregraph.tools.server import Tool, ToolResult, ToolServer

# Tiny illustrative ICD-10 map; a real deployment queries a terminology service.
# Keys are matched as whole-word substrings of the (normalized) term, longest first,
# so "Essential hypertension" resolves via the "hypertension" entry instead of failing.
_ICD10 = {
    "ckd stage 3": "N18.3",
    "chronic kidney disease": "N18.9",
    "type 2 diabetes": "E11.9",
    "atrial fibrillation": "I48.91",
    "heart failure": "I50.9",
    "hypertension": "I10",
    "diabetes": "E11.9",
}


def register_clinical_tools(server: ToolServer, patients: dict[str, Patient]) -> None:
    def fhir_lookup(patient_id: str, loinc: str | None = None) -> ToolResult:
        patient = patients.get(patient_id)
        if patient is None:
            return ToolResult(ok=False, error=f"no such patient: {patient_id}")
        if loinc:
            obs = patient.latest(loinc)
            if obs is None:
                return ToolResult(ok=False, error=f"no observation {loinc}")
            return ToolResult(ok=True, data={"display": obs.display, "value": obs.value,
                                             "unit": obs.unit, "effective": obs.effective})
        return ToolResult(ok=True, data={
            "conditions": [c.display for c in patient.conditions],
            "observations": sorted({o.display for o in patient.observations}),
        })

    def icd10_lookup(term: str) -> ToolResult:
        norm = term.strip().lower()
        for key, code in _ICD10.items():  # longest keys first (dict is ordered)
            if re.search(rf"\b{re.escape(key)}\b", norm):
                return ToolResult(ok=True, data={"term": term, "icd10": code})
        return ToolResult(ok=False, error=f"no ICD-10 match for '{term}'")

    def problem_list(patient_id: str) -> ToolResult:
        """Authoritative coded problem list straight from the record (no fuzzy lookup)."""
        patient = patients.get(patient_id)
        if patient is None:
            return ToolResult(ok=False, error=f"no such patient: {patient_id}")
        coded = {c.display: c.code for c in patient.conditions if c.code}
        return ToolResult(ok=True, data=coded)

    def calc_egfr(creatinine_mg_dl: float, age: int, sex: str) -> ToolResult:
        result = _ckd_epi_2021(creatinine_mg_dl, age, sex)
        return ToolResult(ok=True, data={"egfr": round(result, 1), "unit": "mL/min/1.73m2",
                                         "equation": "CKD-EPI 2021"})

    def calc_chadsvasc(chf: bool, hypertension: bool, age: int, diabetes: bool,
                       stroke_history: bool, vascular_disease: bool, female: bool) -> ToolResult:
        score = _chadsvasc(chf, hypertension, age, diabetes, stroke_history,
                           vascular_disease, female)
        return ToolResult(ok=True, data={"chadsvasc": score,
                                         "anticoagulation_recommended": score >= 2})

    server.register(Tool("fhir_lookup", "Retrieve structured facts for a patient.",
                         _schema({"patient_id": "string", "loinc": "string?"}), fhir_lookup))
    server.register(Tool("problem_list", "Authoritative coded problem list from the record.",
                         _schema({"patient_id": "string"}), problem_list))
    server.register(Tool("icd10_lookup", "Map free-text (uncoded) findings to an ICD-10 code.",
                         _schema({"term": "string"}), icd10_lookup))
    server.register(Tool("calc_egfr", "CKD-EPI 2021 eGFR from serum creatinine.",
                         _schema({"creatinine_mg_dl": "number", "age": "integer", "sex": "string"}),
                         calc_egfr))
    server.register(Tool("calc_chadsvasc", "CHA2DS2-VASc stroke risk score.",
                         _schema({"chf": "boolean", "hypertension": "boolean", "age": "integer",
                                  "diabetes": "boolean", "stroke_history": "boolean",
                                  "vascular_disease": "boolean", "female": "boolean"}),
                         calc_chadsvasc))


def _ckd_epi_2021(scr: float, age: int, sex: str) -> float:
    female = sex.lower().startswith("f")
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    ratio = scr / kappa
    egfr = (142.0 * (min(ratio, 1.0) ** alpha) * (max(ratio, 1.0) ** -1.200)
            * (0.9938 ** age))
    if female:
        egfr *= 1.012
    return float(egfr)


def _chadsvasc(chf: bool, htn: bool, age: int, dm: bool, stroke: bool,
               vasc: bool, female: bool) -> int:
    score = 0
    score += int(chf) + int(htn) + int(dm) + int(vasc) + int(female)
    score += 2 * int(stroke)
    if age >= 75:
        score += 2
    elif age >= 65:
        score += 1
    return score


def _schema(fields: dict[str, str]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    required: list[str] = []
    type_map = {"string": "string", "number": "number", "integer": "integer",
                "boolean": "boolean"}
    for name, spec in fields.items():
        optional = spec.endswith("?")
        base = spec.rstrip("?")
        props[name] = {"type": type_map[base]}
        if not optional:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}
