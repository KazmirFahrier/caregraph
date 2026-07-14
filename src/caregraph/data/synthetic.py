"""Synthetic data generation. No real PHI, ever.

`synthetic_patient` returns a Synthea-style FHIR bundle. `guideline_corpus` returns
a small set of public clinical-guideline snippets used as the RAG knowledge base.
"""
from __future__ import annotations

from typing import Any


def synthetic_patient() -> dict[str, Any]:
    """A single synthetic CKD/hypertension patient as a FHIR bundle."""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "synthetic-0001",
                    "name": [{"given": ["Jordan"], "family": "Rivera"}],
                    "birthDate": "1957-04-12",
                    "gender": "female",
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"code": "33914-3", "display": "eGFR"}]},
                    "valueQuantity": {"value": 38.0, "unit": "mL/min/1.73m2"},
                    "effectiveDateTime": "2026-06-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"code": "33914-3", "display": "eGFR"}]},
                    "valueQuantity": {"value": 45.0, "unit": "mL/min/1.73m2"},
                    "effectiveDateTime": "2026-01-05",
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"code": "8480-6", "display": "Systolic BP"}]},
                    "valueQuantity": {"value": 152.0, "unit": "mmHg"},
                    "effectiveDateTime": "2026-06-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"code": "2160-0", "display": "Serum creatinine"}]},
                    "valueQuantity": {"value": 1.4, "unit": "mg/dL"},
                    "effectiveDateTime": "2026-06-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "code": {"coding": [{"code": "N18.3", "display": "CKD stage 3"}]},
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "code": {"coding": [{"code": "I10", "display": "Essential hypertension"}]},
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                }
            },
            {
                # An uncoded free-text finding — exercises icd10_lookup's fallback path
                # (coded conditions above are read straight from the record instead).
                "resource": {
                    "resourceType": "Condition",
                    "code": {"coding": [{"code": "", "display": "Atrial fibrillation"}]},
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                }
            },
        ],
    }


def guideline_corpus() -> list[dict[str, str]]:
    """Public clinical-guideline snippets. Each is a citable knowledge document."""
    return [
        {
            "doc_id": "KDIGO-CKD-2024",
            "title": "KDIGO CKD staging",
            "text": (
                "Chronic kidney disease is staged by estimated glomerular filtration "
                "rate (eGFR). Stage 3a is an eGFR of 45 to 59 mL/min/1.73m2 and stage 3b "
                "is 30 to 44 mL/min/1.73m2. A sustained decline in eGFR of more than 5 "
                "mL/min/1.73m2 per year is considered rapid progression and warrants "
                "nephrology referral."
            ),
        },
        {
            "doc_id": "ACC-AHA-HTN-2023",
            "title": "Hypertension targets in CKD",
            "text": (
                "For adults with chronic kidney disease and hypertension, a blood "
                "pressure target of less than 130/80 mmHg is recommended. A systolic "
                "blood pressure of 152 mmHg is above target and indicates that "
                "antihypertensive therapy should be intensified. ACE inhibitors or ARBs "
                "are first line in patients with albuminuria."
            ),
        },
        {
            "doc_id": "DDI-NSAID-CKD",
            "title": "Medication safety in CKD",
            "text": (
                "Nonsteroidal anti-inflammatory drugs should be avoided in patients with "
                "an eGFR below 30 mL/min/1.73m2 because they can worsen renal function. "
                "Metformin dosing should be reviewed when eGFR falls below 45."
            ),
        },
    ]
