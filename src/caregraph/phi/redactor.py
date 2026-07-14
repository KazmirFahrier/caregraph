"""PHI redaction guardrail.

HIPAA Safe Harbor names 18 identifier categories. This module ships a fast,
dependency-free regex baseline covering the highest-risk ones and exposes the same
interface Microsoft Presidio uses, so a Presidio analyzer can be dropped in for
production NER-based detection without changing callers.

Design: the guardrail *blocks* — `assert_clean()` raises on residual PHI rather than
logging and continuing. Loud failure over silent leakage.

Known limitation (AUDIT2 A2-L2): person names are a Safe Harbor identifier that a
regex baseline cannot reliably detect, so this module does NOT redact names. Name /
`PERSON` detection requires the NER path (`pip install caregraph[phi]`, Presidio)
behind this same interface. The synthetic pipeline avoids surfacing patient names
(it references patient ids), but production must enable the Presidio recognizer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# (label, compiled pattern) — ordered; earlier patterns win on overlap. Ordering
# matters: SSN before PHONE, and both date forms before any bare-number rule.
#
# Design note (see AUDIT.md H2): the ZIP rule is deliberately *context-gated* — a
# bare 5-digit number is far more likely to be a lab value, specimen id, or dosage
# than a postal code, so redacting every `\d{5}` corrupts clinical text. ZIP only
# fires when preceded by a "zip"/"postal" cue or a US state abbreviation.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("MRN", re.compile(r"\bMRN[:#]?\s*\d{5,10}\b", re.IGNORECASE)),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # ISO 8601 dates as used in every FHIR `effectiveDateTime` — a HIPAA identifier.
    ("DATE", re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T[\d:]+)?\b")),
    ("DATE", re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    # Cue-word ZIP (case-insensitive): "zip 02139", "postal code: 02139-1234".
    ("ZIP", re.compile(r"(?:zip|postal)(?:\s+code)?[:#]?\s*\d{5}(?:-\d{4})?\b", re.IGNORECASE)),
    # State-abbreviation ZIP: only real USPS state codes, so clinical abbreviations
    # like "IV 50000" or "PO 40000" are not mistaken for an address (A3-L2). Codes
    # that are also medical abbreviations (e.g. "CT") remain inherently ambiguous —
    # the Presidio NER path resolves those.
    ("ZIP", re.compile(
        r"\b(?:A[LKZR]|C[AOT]|D[EC]|FL|GA|HI|I[DLNA]|K[SY]|LA|M[EDAINSOT]|"
        r"N[EVHJMYCD]|O[HKR]|PA|RI|S[CD]|T[NX]|UT|V[TA]|W[AVIY])\s+\d{5}(?:-\d{4})?\b")),
]


@dataclass
class RedactionReport:
    text: str
    hits: dict[str, int] = field(default_factory=dict)

    @property
    def found_phi(self) -> bool:
        return bool(self.hits)


class PHIRedactor:
    """Replace detected identifiers with typed placeholders, e.g. ``[SSN]``."""

    def redact(self, text: str) -> RedactionReport:
        hits: dict[str, int] = {}
        redacted = text
        for label, pattern in _PATTERNS:
            redacted, n = pattern.subn(f"[{label}]", redacted)
            if n:
                hits[label] = hits.get(label, 0) + n
        return RedactionReport(text=redacted, hits=hits)

    def assert_clean(self, text: str) -> None:
        """Raise if any PHI pattern still matches. Callers use this on outbound text."""
        report = self.redact(text)
        if report.found_phi:
            raise PHILeakError(f"Residual PHI detected before egress: {report.hits}")


class PHILeakError(RuntimeError):
    """Raised when PHI would otherwise escape the trust boundary."""
