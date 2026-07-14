from caregraph.phi import PHIRedactor
from caregraph.phi.redactor import PHILeakError

import pytest


def test_redacts_common_identifiers() -> None:
    r = PHIRedactor()
    report = r.redact("SSN 123-45-6789, call 555-123-4567 or jane@x.com, MRN: 0099887")
    assert "[SSN]" in report.text
    assert "[PHONE]" in report.text
    assert "[EMAIL]" in report.text
    assert "[MRN]" in report.text
    assert report.found_phi


def test_clean_text_reports_no_phi() -> None:
    r = PHIRedactor()
    assert not r.redact("Blood pressure target is below 130/80 mmHg.").found_phi


def test_assert_clean_blocks_leakage() -> None:
    r = PHIRedactor()
    with pytest.raises(PHILeakError):
        r.assert_clean("contact 555-123-4567")


# --- Regression tests for AUDIT.md H2 (both directions) ---

def test_iso_dates_are_redacted() -> None:
    # FHIR effectiveDateTime format — a HIPAA Safe Harbor identifier that the old
    # slash-only DATE rule missed entirely.
    r = PHIRedactor()
    assert "[DATE]" in r.redact("Observed on 2026-06-01").text
    assert "[DATE]" in r.redact("At 2026-06-01T09:30:00 in clinic").text


def test_bare_five_digit_clinical_values_are_not_over_redacted() -> None:
    # Specimen ids, dosages, and lab values must survive — the old ZIP rule mangled
    # every \d{5} into [ZIP], corrupting the answer.
    r = PHIRedactor()
    out = r.redact("Specimen 45213 shows glucose 12000 units").text
    assert "45213" in out and "12000" in out
    assert "[ZIP]" not in out
    assert not r.redact("Specimen 45213").found_phi


def test_zip_still_caught_with_context() -> None:
    r = PHIRedactor()
    assert "[ZIP]" in r.redact("Patient lives in MA 02139").text
    assert "[ZIP]" in r.redact("postal code 02139-1234").text


def test_non_state_abbreviations_are_not_over_redacted() -> None:
    # AUDIT3 A3-L2: only real USPS state codes gate a ZIP; medical abbreviations
    # followed by 5-digit values must survive.
    r = PHIRedactor()
    assert "50000" in r.redact("IV 50000 units heparin").text
    assert "40000" in r.redact("PO 40000 daily").text
    assert "[ZIP]" not in r.redact("IV 50000 units").text
