# CareGraph — Fourth Audit

Fourth pass, targeting the boundary between the agent's tool outputs and the PHI gate,
and the serving layer's shared state under the documented multi-replica deployment.
`ruff`/`mypy --strict` clean, 56 tests pass — none catch these.

Four findings: two MEDIUM, two LOW. **No HIGH this round** — the first sign across four
passes that the remaining surface is shrinking (severity trend below).

---

## MEDIUM

### A4-M1 — Tool outputs bypass the PHI gate (observation dates leak to the caller)
The `verify` node redacts the *answer* text, but the agent's **tool-call summaries are
never redacted** — and they carry PHI. Reproduced:

```
agent_ask("What is the eGFR and blood pressure for this CKD patient?")
tool_calls:
  "display=eGFR; value=38.0; unit=mL/min/1.73m2; effective=2026-06-01"
  "display=Systolic BP; value=152.0; unit=mmHg; effective=2026-06-01"
```

`2026-06-01` is a FHIR `effectiveDateTime` — a HIPAA Safe Harbor identifier — and it is
returned verbatim in the API response's `tool_calls`. Worse, the same unredacted
summaries are concatenated into the **draft context** (`_draft`'s `tool_ctx`), so a
question like "when was the eGFR measured?" could surface the date in the answer too.
The PHI gate covers only the final answer string, not the tool data that feeds it or
the tool metadata shipped alongside it.

**Impact:** a real PHI egress on the agent path, in a project whose headline is PHI
safety. The `/ask` path is unaffected (no patient data in the guideline KB), which is
why earlier passes missed it — the leak is specific to tool results.

**Fix:** redact tool result summaries at the boundary — before they enter the draft
context and before they are returned in `tool_calls` — and run `assert_clean` over the
serialized tool output, not just the answer.

### A4-M2 — Human-in-the-loop resume state is per-process, but the deployment runs 2 replicas
`_pending` (the interrupted-run store keyed by `resume_token`) lives in a single
process's memory. But `deploy/k8s/deployment.yaml` sets `replicas: 2` behind a
`ClusterIP` Service with no session affinity. So:

- `POST /agent/ask` on replica A issues a token and stores state in A's memory.
- `POST /agent/approve` load-balances to replica B → `_pending` miss → **404**.

The clinician sign-off flow — the safety feature fixed in AUDIT2 — is **broken under
the very deployment the repo ships**. It also means an approval is lost on any pod
restart/rollout.

**Fix:** move `_pending` to a shared store (Redis/DB) with a TTL, or set session
affinity / a single-replica annotation and say so. At minimum the manifest and the
in-memory design must be made consistent.

---

## LOW

### A4-L1 — `_audit` log grows unbounded
Every request appends to `self._audit` with no cap or rotation (50 asks → 50 entries,
forever). A slow memory leak on a long-lived process, and unbounded growth of a
structure that should be emitted to a log sink, not retained in RAM.

**Fix:** cap with a ring buffer (`collections.deque(maxlen=...)`) or write audit
events to a logger/OTLP and keep only a recent window in memory.

### A4-L2 — `_pending` / `_audit` are unguarded shared mutable state
`agent_ask`, `approve`, and `_evict_pending` mutate `_pending` (and append to
`_audit`) with no lock. `_evict_pending` iterates `self._pending.items()` to build the
expired list while other threads may insert — a latent `RuntimeError: dictionary
changed size during iteration` under concurrent load (not reliably reproduced in a
64-thread run, but the race is real). A3-M2 fixed the tracer race; these two shared
dicts/lists were not covered.

**Fix:** guard the pending/audit mutations with a `threading.Lock` (or use structures
that tolerate concurrent access), consistent with the per-request-tracer fix.

---

## Summary

| ID | Severity | Area | One-line | Status |
|----|----------|------|----------|--------|
| A4-M1 | MED | PHI | tool-call summaries (and draft context) skip the PHI gate → date leaks | ✅ fixed |
| A4-M2 | MED | architecture | in-memory resume state breaks under the shipped `replicas: 2` | ✅ fixed |
| A4-L1 | LOW | serving | `_audit` grows unbounded | ✅ fixed |
| A4-L2 | LOW | concurrency | `_pending`/`_audit` mutated without a lock | ✅ fixed |

**Severity trend across four passes:** HIGH×2, HIGH×1, HIGH×1, **HIGH×0**. The audits
are now surfacing MED/LOW hardening items rather than headline-breaking defects.

---

## Resolution log

All four fixed and locked with regression tests. Post-fix: `ruff` clean,
`mypy --strict` clean, **60 tests pass** (up from 56), eval gate exits 0.

- **A4-M1** — `_act` now redacts each tool-result summary at the boundary and runs
  `assert_clean` over it, so PHI (e.g. `effectiveDateTime`) is `[DATE]`-masked before
  it enters the draft context or the returned `tool_calls`. Verified: no raw ISO date
  appears in the agent response.
- **A4-M2** — Made the manifest consistent with the design: `replicas: 1` with a
  comment that the in-process resume state requires a shared `PendingStore` (Redis) to
  scale past one pod. The store is now a swappable `PendingStore` protocol so that
  implementation drops in without touching callers.
- **A4-L1** — `_audit` is a `deque(maxlen=1000)` — bounded, and meant to be drained to
  a log sink in production.
- **A4-L2** — Pending mutations live in `InMemoryPendingStore`, guarded by a
  `threading.Lock` with snapshot-based eviction (no mutate-while-iterating). A
  80-iteration/8-thread stress test with `ttl=0` exercises concurrent put/pop/evict
  without raising.
