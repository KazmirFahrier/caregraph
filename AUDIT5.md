# CareGraph — Fifth Audit

Fifth pass. The code-level defect surface is largely exhausted (four passes, 23 fixes),
so this one targets the *robustness of the centerpiece* (the grounding gate) and the
*policy* around the safety gate. `ruff`/`mypy --strict` clean, 60 tests pass.

Four findings: two MEDIUM, two LOW. Still **no HIGH**. The findings are now design/policy
weaknesses rather than outright bugs — the honest signal that what remains is hardening.

---

## MEDIUM

### A5-M1 — The grounding gate is union-based → cross-document recombination scores as faithful
`faithfulness` (and the citation `_overlaps`) test an answer sentence against the
**union of all retrieved documents' terms**, not against any single document. So an
answer that recombines terms from different sources is scored as fully grounded even
though no source supports it. Reproduced at the metric level:

```
contexts = ["The blood pressure target is below 130.",
            "Systolic reached 200 mmHg during the crisis."]
faithfulness("The blood pressure target is 200 mmHg.", contexts)  -> 1.0     # nothing says this
_overlaps("The target is 200 mmHg", "Systolic reached 200 mmHg ...") -> True  # mis-citable
```

Today this is *masked* because the deterministic `ScriptedClient` only emits verbatim
context sentences. But the system is explicitly provider-agnostic (the whole `LLMClient`
seam exists so a real model plugs in), and the moment a real LLM is used, a hallucination
that stitches "target" + "200 mmHg" from two docs **passes the gate and gets a citation**.
The gate's central claim ("blocks ungrounded numbers") is therefore weaker than stated
for its intended use.

**Fix:** score support per-document — a sentence is faithful iff **one** document
supports ≥ the threshold of its terms (`max` over docs, not union); cite only documents
that individually support the sentence. Add a two-trusted-doc recombination test to the
gate so it can't regress.

### A5-M2 — Clinician sign-off has no identity or attribution
The human-in-the-loop gate's whole purpose is that a *licensed clinician* approves an
actionable recommendation. But `/agent/approve` takes only a `resume_token` + a boolean:

```
audit entry -> {"endpoint": "agent_approve", "approved": true, "refused": false}
```

No approver identity is required, authenticated, or recorded. Anyone who obtains the
(unauthenticated) token can release the recommendation, and the audit trail cannot say
*who* signed off — which defeats the accountability the gate exists to provide and is a
problem for any real HIPAA/clinical audit.

**Fix:** require an authenticated `approver_id` on approval, reject anonymous approvals,
and record the approver (and timestamp) in the audit event.

---

## LOW

### A5-L1 — No authentication on any endpoint; PHI endpoints are open
`/ask`, `/agent/ask`, and `/agent/approve` have no auth. For a system that returns
clinical data (and, in production, PHI), an open API is a deployment-blocker. The K8s
`NetworkPolicy` restricts *egress* but nothing gates *ingress*.

**Fix:** add auth (API key / OIDC bearer) middleware; document that it is mandatory
before any real PHI flows. Ties into A5-M2 (the approver identity comes from auth).

### A5-L2 — The agent ignores `Settings.top_k`
`ClinicalRAG` reads `Settings.top_k`, but `ClinicalAgent` hardcodes `top_k = 4`. Tuning
retrieval breadth via config silently doesn't affect the agent path — config drift
between two code paths that should share it.

**Fix:** default the agent's `top_k` from `Settings`, so one knob controls both.

---

## Summary

| ID | Severity | Area | One-line | Status |
|----|----------|------|----------|--------|
| A5-M1 | MED | grounding | union-based faithfulness/citations pass cross-doc recombination | ✅ fixed |
| A5-M2 | MED | safety/policy | clinician sign-off has no authenticated identity or attribution | ✅ fixed |
| A5-L1 | LOW | security | no authN on any endpoint; PHI endpoints open | ✅ fixed |
| A5-L2 | LOW | config | agent ignores `Settings.top_k` (hardcoded) | ✅ fixed |

**Severity trend across five passes:** HIGH 2 → 1 → 1 → 0 → 0. New findings have moved
from "wrong code" to "gate robustness under a real LLM" (A5-M1) and "policy/identity"
(A5-M2) — the shape of a codebase approaching done.

---

## Resolution log

All four fixed and locked with regression tests. Post-fix: `ruff` clean,
`mypy --strict` clean, **64 tests pass** (up from 60), eval gate exits 0.

- **A5-M1** — `faithfulness` now scores support **per document** (a sentence must be
  carried by a single context, `any` over docs) instead of against the union of all
  context terms. Verified: a claim recombining terms from two docs scores **0.0**,
  while genuine single-doc grounding still scores 1.0. A recombination regression test
  guards it; the eval gate still passes (real answers are single-source).
- **A5-M2** — `service.approve` now requires a non-empty `approver_id`, rejects
  anonymous approvals (`PermissionError` → HTTP 401), and records the approver id +
  timestamp in the audit event. The `Approval` request model makes `approver_id`
  mandatory.
- **A5-L1** — Every data endpoint (`/tools`, `/ask`, `/agent/ask`, `/agent/approve`)
  now depends on an API-key check; set `CAREGRAPH_API_KEY` to enforce it (required
  before real PHI). `/health` stays open for probes.
- **A5-L2** — `ClinicalAgent` defaults `top_k` from `Settings.top_k`, so one knob
  controls retrieval breadth on both the pipeline and agent paths.
