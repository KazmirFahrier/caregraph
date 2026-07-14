"""Minimal FHIR R4 models and a bundle loader.

Not a full FHIR implementation — a typed subset (Patient, Observation, Condition,
MedicationRequest) sufficient to demonstrate ingesting Synthea-style bundles into
CareGraph's internal schema. Real deployments would use `fhir.resources`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Observation:
    code: str          # LOINC
    display: str
    value: float
    unit: str
    effective: str     # ISO date


@dataclass
class Condition:
    code: str          # ICD-10 / SNOMED
    display: str
    clinical_status: str = "active"


@dataclass
class Patient:
    patient_id: str
    given: str
    family: str
    birth_date: str
    gender: str
    observations: list[Observation] = field(default_factory=list)
    conditions: list[Condition] = field(default_factory=list)

    def latest(self, loinc: str) -> Observation | None:
        obs = [o for o in self.observations if o.code == loinc]
        return max(obs, key=lambda o: o.effective) if obs else None


def load_bundle(bundle: dict[str, Any]) -> Patient:
    """Parse a FHIR transaction bundle into a `Patient`.

    Accepts the standard `{"resourceType": "Bundle", "entry": [...]}` shape.
    """
    patient: Patient | None = None
    observations: list[Observation] = []
    conditions: list[Condition] = []

    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        rtype = res.get("resourceType")
        if rtype == "Patient":
            name = (res.get("name") or [{}])[0]
            patient = Patient(
                patient_id=res.get("id", "unknown"),
                given=" ".join(name.get("given", [])),
                family=name.get("family", ""),
                birth_date=res.get("birthDate", ""),
                gender=res.get("gender", ""),
            )
        elif rtype == "Observation":
            coding = (res.get("code", {}).get("coding") or [{}])[0]
            qty = res.get("valueQuantity", {})
            observations.append(
                Observation(
                    code=coding.get("code", ""),
                    display=coding.get("display", ""),
                    value=float(qty.get("value", 0.0)),
                    unit=qty.get("unit", ""),
                    effective=res.get("effectiveDateTime", ""),
                )
            )
        elif rtype == "Condition":
            coding = (res.get("code", {}).get("coding") or [{}])[0]
            conditions.append(
                Condition(
                    code=coding.get("code", ""),
                    display=coding.get("display", ""),
                    clinical_status=(res.get("clinicalStatus", {}).get("coding") or [{}])[0].get(
                        "code", "active"
                    ),
                )
            )

    if patient is None:
        raise ValueError("Bundle contains no Patient resource")
    patient.observations = observations
    patient.conditions = conditions
    return patient
