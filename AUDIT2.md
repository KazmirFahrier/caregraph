# CareGraph — Second Audit

A fresh critical pass after the first round of fixes, focused on (a) code added during
those fixes and (b) claims the first audit didn't test. `ruff` and `mypy --strict` are
still clean and 44 tests pass — so, as before, none of these were caught by the suite.

Five findings: one HIGH, two MEDIUM, two LOW. The HIGH is the most important defect
found across *either* audit, because it defeats a headline safety claim at runtime.

---

## HIGH

### A2-H1 — Human-in-the-loop is not enforced at the service/API boundary
The Phase-2 headline is "the agent pauses for clinician sign-off before anything
actionable." At the HTTP boundary it does not. Reproduced against the real service:

```
POST /agent/ask  {"question": "Should we prescribe an ACE inhibitor and start it today?"}
-> status    : interrupted
   needs_review: True
   refused   : False
   answer    : "ACE inhibitors or ARBs are first line in patients with albuminuria."
```

Two compounding problems:

1. **The recommendation leaks.** The graph interrupts *before* the `human_review`
   node — but `verify` has already populated `answer` with the full clinical
   recommendation. `CareGraphService.agent_ask` calls `agent.run()` and returns that
   state verbatim, so the caller receives the actionable answer with `refused=False`.
   The "Withheld pending clinician review" text only appears if someone calls
   `approve(approved=False)`, which the service never does.
2. **There is no way to approve.** `CareGraphService` exposes `ask`, `agent_ask`,
   `health`, `list_tools` — but no `resume`/`approve`. So the interrupted state is a
   dead end over HTTP: the sign-off flow that makes the gate meaningful is unreachable.

**Impact:** the safety property demonstrated in `agent_demo.py` (which correctly uses
`run_raw` + `approve`) is absent from the deployable surface. A reviewer who exercises
the API sees an actionable clinical recommendation returned without sign-off.

**Fix:** on interrupt, blank the answer (return only a "withheld, review required"
payload + an opaque resume token) and add a `POST /agent/approve` endpoint that resumes
the graph. `agent_ask` should never return a `refused=False` actionable answer in the
`interrupted` state.

---

## MEDIUM

### A2-M1 — `icd10_lookup` fuzzy-matches display text and silently half-fails
The agent codes conditions by passing the FHIR **display string** to a fuzzy lookup:

```
icd10_lookup("CKD stage 3")          -> ok:  icd10=N18.3
icd10_lookup("Essential hypertension") -> ERROR: no ICD-10 match
```

Two problems: (1) the terminology map is keyed on `"hypertension"`, not the display
the bundle actually uses, so half the lookups fail; (2) more fundamentally, the
**authoritative ICD-10 code is already in `Condition.code`** (`I10`, `N18.3`) — doing a
lossy text→code lookup when the structured code is present is backwards and manufactures
avoidable failures.

**Fix:** read `Condition.code` directly for coding; reserve `icd10_lookup` for free-text
that has no code, and normalize/expand its map (or back it with a real terminology
service). The failure should not be silent either way.

### A2-M2 — Wall-clock-dependent age makes outputs and tests non-deterministic
`ClinicalAgent._age()` uses `date.today()`. Every calculator that consumes age
(CHA₂DS₂-VASc, eGFR indirectly) therefore changes with the calendar, and
`test_chadsvasc_is_derived_from_patient_not_hardcoded` asserts `chadsvasc=3` for a 1957
birth — true only while that patient is 65–74. In 2032 they turn 75, the score becomes
4, and the test fails for no code change. A deterministic suite must not depend on the
run date.

**Fix:** thread an `as_of: date` (default `date.today()`) through the agent/service and
pass a fixed date in tests; assert on the derived age explicitly.

---

## LOW

### A2-L1 — The "one recovery attempt per tool" is a blind identical retry
`_act` retries a failed tool call with the *same* name and *same* arguments:

```python
res = self._tools.call_tool(call["name"], call["arguments"])
if not res.ok:
    res = self._tools.call_tool(call["name"], call["arguments"])  # identical
```

For a deterministic failure (unknown patient, unmapped term) the retry returns the
identical error — it can never recover, it just doubles the call. The docstring's
"one recovery attempt per tool" oversells a no-op.

**Fix:** either make recovery meaningful (repair/relax arguments, fall back to a
structured field) or drop the retry and the claim.

### A2-L2 — Names (Safe Harbor identifier #1) are never redacted
`redact("Patient Jordan Rivera, seen today")` returns the name unchanged. This is an
acknowledged limitation of a regex baseline, but for a HIPAA-positioned repo it should
be explicit: the synthetic patient's `given`/`family` must never flow into surfaced
text (today `demo.py` uses the id, so it's safe *by accident*), and the README should
state that name/person detection requires the Presidio NER path from the `phi` extra.

**Fix:** note the limitation in `phi/redactor.py` and README; optionally add a
`[NAME]` pass wired to Presidio's `PERSON` recognizer behind the existing interface.

---

## Summary

| ID | Severity | Area | One-line | Status |
|----|----------|------|----------|--------|
| A2-H1 | HIGH | safety/API | actionable answer leaks; HITL gate unreachable over HTTP | ✅ fixed |
| A2-M1 | MED | correctness | icd10 fuzzy-matches display, ignores the code in the record | ✅ fixed |
| A2-M2 | MED | determinism | wall-clock age → non-deterministic output + time-bomb test | ✅ fixed |
| A2-L1 | LOW | agent | "recovery attempt" is a blind identical retry | ✅ fixed |
| A2-L2 | LOW | PHI | names never redacted (regex baseline; make it explicit) | ✅ fixed |

**Note on method:** A2-H1 and A2-M2 both live in code the first round *added or
touched* — closing findings can open new ones, which is exactly why a second pass on
the diff is worth doing.

---

## Resolution log

All five fixed and locked with regression tests. Post-fix: `ruff` clean,
`mypy --strict` clean, **50 tests pass** (up from 44), eval gate exits 0.

- **A2-H1** — The agent's `_to_result` now returns `"Withheld pending clinician
  review."` with `refused=True` for any interrupted (unapproved) run, so no draft
  ever leaves the agent. `CareGraphService.agent_ask` parks the real state
  server-side under a single-use `resume_token`; a new `CareGraphService.approve` +
  `POST /agent/approve` endpoint resumes the run only on clinician sign-off. Verified
  end to end: actionable request → withheld → approve → answer released.
- **A2-M1** — Added an authoritative `problem_list` tool that reads codes straight
  from `Condition.code`; the agent prefers it and only falls back to `icd10_lookup`
  for uncoded free-text findings. `icd10_lookup` now matches display variants by
  whole-word substring (so "Essential hypertension" → I10 instead of failing). A
  synthetic uncoded AF finding exercises the fallback. All 5 tools are now invoked.
- **A2-M2** — Threaded an `as_of: date` through the agent/`build_agent`; `_age` uses
  it (defaults to `date.today()` in prod). The CHA₂DS₂-VASc test pins `as_of` so it
  can never break with the calendar.
- **A2-L1** — Removed the blind identical retry; a failed tool call is recorded as a
  typed result and the agent continues (no pretense of recovery).
- **A2-L2** — Documented the name-redaction limitation in `phi/redactor.py` and the
  README, pointing to the Presidio `PERSON` recognizer behind the same interface.
