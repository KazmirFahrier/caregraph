# CareGraph — Pre-portfolio Audit

A deliberate defect-finding pass before treating the repo as portfolio-ready — the
same discipline applied to campaign-copilot. Every item below is reproduced from the
current code, not hypothetical. Nine findings: two HIGH, three MEDIUM, four LOW.

Verification environment: `ruff 0.x`, `mypy --strict`, `pytest` (36 passing) in the
sandbox. Tests passing did **not** catch any of these — that is the point.

---

## HIGH

### H1 — CI is red: the "green CI / mypy --strict" claim is false
`ruff check` reports 1 error (unused `field` import in `agent/graph.py:13`).
`mypy --strict src/caregraph` reports **21 errors** in the exact environment CI uses
(`pip install -e ".[dev]"`, i.e. optional extras absent):

- 12 × `import-not-found` for guarded optional deps (`torch`, `transformers`, `peft`,
  `trl`, `datasets`, `ragas`, `qdrant_client`, `opentelemetry*`, `fastapi`,
  `pydantic`). These are never silenced, so `mypy src` fails on a clean dev install.
- 4 real typing errors (see L9).
- 5 `untyped-decorator` errors in `serve/app.py` because FastAPI resolves to `Any`.

**Impact:** the CI workflow as committed would fail on the first push — the opposite
of the "production CI from the start" story the README tells. A reviewer who runs the
pipeline sees red.

**Fix:** add `[[tool.mypy.overrides]]` with `ignore_missing_imports = true` for the
optional modules (or install the extras in CI), fix the 4 real errors, remove the
unused import. Then make CI actually run `ruff` + `mypy` (it lists them but they've
never passed).

### H2 — PHI redactor is both over- and under-inclusive (HIPAA correctness)
`phi/redactor.py` fails in both directions, demonstrated:

```
redact("Visit on 2026-06-01 with eGFR 38")  -> "Visit on 2026-06-01 ..."   # date NOT redacted
redact("Specimen 45213 shows glucose 12000") -> "Specimen [ZIP] ... [ZIP]"  # clinical values MANGLED
```

- **Under-redaction:** ISO dates (`YYYY-MM-DD`) are the format FHIR uses for every
  `effectiveDateTime`, and dates are a HIPAA Safe Harbor identifier. The `DATE`
  pattern only matches slashed US dates, so real dates flow straight through — an
  actual PHI leak in a project whose headline claim is PHI safety.
- **Over-redaction:** the `ZIP` pattern `\b\d{5}\b` rewrites *any* 5-digit number as
  `[ZIP]`, corrupting specimen IDs, dosages, and lab values in the answer text.

**Impact:** this is the single most important bug given the project's positioning.
The red-team "PHI leakage" test passes only because it happens to use a slashed SSN
and a phone number, not a date.

**Fix:** add an ISO-date pattern; constrain `ZIP` to contexts (preceded by a state or
"zip"/"postal") or drop it in favor of Presidio's NER for numbers. Add a red-team
case with an ISO date and a 5-digit lab value so the gate protects both directions.

---

## MEDIUM

### M1 — Agent fabricates CHA₂DS₂-VASc inputs instead of reading the patient
`agent/clinical_agent.py:98-100` hardcodes the calculator arguments:

```python
calls.append({"name": "calc_chadsvasc", "arguments": {
    "chf": False, "hypertension": True, "age": 69, "diabetes": False,
    "stroke_history": False, "vascular_disease": False, "female": True}})
```

Age, sex, and comorbidities are constants, not derived from the patient's FHIR
record. The stroke-risk score is therefore identical for every patient — a grounding
failure that directly contradicts the "answers are grounded in the patient" thesis.
`calc_egfr` has the same latent problem (it's never wired to real creatinine).

**Fix:** derive the arguments from `fhir_lookup` / the loaded `Patient` (birth date →
age, `gender`, `conditions`) before calling the calculator.

### M2 — Two of four MCP tools are never invoked by the agent (test-only paths)
Registered tools: `fhir_lookup`, `icd10_lookup`, `calc_egfr`, `calc_chadsvasc`.
The agent's `_plan` only ever emits `fhir_lookup` and `calc_chadsvasc`. `icd10_lookup`
and `calc_egfr` are reachable only from unit tests / direct calls — exactly the
"test-only call path creates false confidence" defect found in campaign-copilot
(`compress()`/`pin()`).

**Fix:** either route questions to these tools in `_plan` (e.g. "what's the ICD-10
code for…" → `icd10_lookup`; a creatinine value → `calc_egfr`) or remove them so the
tool surface reflects what the system actually uses.

### M3 — The managed-vector-DB backend switch is advertised but inert
- `Settings.vector_backend` and `Settings.qdrant_url` are read by **no** code path.
- `build_default_service` always constructs `InMemoryVectorStore`, regardless of env.
- `QdrantVectorStore.__init__` raises `NotImplementedError`, so even manual selection
  fails.
- Yet `docker-compose.yml` and `deploy/k8s/deployment.yaml` set
  `CAREGRAPH_VECTOR_BACKEND=qdrant` and `QDRANT_URL`, implying a working switch.

**Impact:** a reviewer who reads the compose file and then the service sees that the
"managed vector DB" is interface-only with no factory honoring it.

**Fix:** add a `make_vector_store(settings)` factory that reads `vector_backend` and
either implement the Qdrant path or clearly label it "Phase-5 stub, not wired" in the
README status table rather than in deploy manifests.

---

## LOW

### L1 — Divergent `faithfulness` definitions
`pipeline._faithfulness` (per-sentence overlap, threshold 0.3) gates whether an answer
ships; `eval/ragas_metrics.faithfulness` (0.5 term-overlap) is what the CI gate
measures. The metric guarding production is not the metric being evaluated, so they
can disagree. Unify on one definition (have the pipeline import the eval metric).

### L2 — No `py.typed` marker
The package advertises strict typing but ships no `py.typed`, so downstream consumers
get no types. Add `src/caregraph/py.typed` and include it in packaging.

### L3 — Fine-tuning ablation scores a suspicious perfect 1.000
`LogisticExtractor` hits F1 = 1.000 because every paraphrase in the test set also
appears in training (closed vocabulary). It's honest but invites a "is this leaking?"
question in an interview. Hold out a few paraphrases from training so the number lands
believably below 1.0 and the story ("generalizes to unseen phrasings") is stronger.

### L4 — Four real type errors hidden behind the import noise
`tools/clinical.py:86` and `agent/clinical_agent.py:174` return `Any` where a concrete
type is declared; `rag/vectorstore.py:78` has an empty-body method with a non-None
return type. These are genuine (not dependency) issues and should be fixed regardless
of H1's import handling.

---

## Summary table

| ID | Severity | Area | One-line | Status |
|----|----------|------|----------|--------|
| H1 | HIGH | CI/quality | ruff + 21 mypy-strict errors; CI is red, not green | ✅ fixed |
| H2 | HIGH | PHI/HIPAA | misses ISO dates, mangles 5-digit clinical values | ✅ fixed |
| M1 | MED | grounding | CHA₂DS₂-VASc inputs hardcoded, not from patient | ✅ fixed |
| M2 | MED | agent | 2/4 MCP tools never invoked (test-only paths) | ✅ fixed |
| M3 | MED | infra | vector-backend switch advertised but inert | ✅ fixed |
| L1 | LOW | eval | two divergent faithfulness definitions | ✅ fixed |
| L2 | LOW | packaging | missing `py.typed` | ✅ fixed |
| L3 | LOW | eval | fine-tune ablation perfect score invites doubt | ✅ fixed |
| L4 | LOW | typing | 4 real type errors behind the import noise | ✅ fixed |

**Recommended order:** H1 and H2 before the repo goes public (they undercut the two
loudest claims — production CI and PHI safety). M1–M3 next (they undercut the grounding
thesis). L1–L4 are polish.

---

## Resolution log

All nine findings fixed and locked with regression tests. Post-fix state:
`ruff` clean, `mypy --strict` clean (36 files), **44 tests pass** (up from 36), eval
gate exits 0.

- **H1** — Added `[[tool.mypy.overrides]]` silencing missing stubs for the guarded
  optional deps; relaxed decorator/subclass checks for `serve.app` only (FastAPI is
  dynamically typed); fixed the 4 real type errors (L4); removed the unused import; CI
  now runs `ruff` + `mypy --strict` and both pass.
- **H2** — Added ISO-8601 date detection (with optional time); replaced the blanket
  `\d{5}` ZIP rule with two context-gated rules (a `zip`/`postal` cue, or a
  case-sensitive state-abbreviation prefix). New unit tests + a red-team ISO-date case.
- **M1** — `calc_chadsvasc` inputs are now derived from the FHIR record (age from
  birth date, sex from gender, comorbidities from conditions via word-boundary
  matching — which caught a live "tia" ⊂ "essential" false positive during testing).
- **M2** — `_plan` now routes creatinine questions to `calc_egfr` (using the patient's
  own serum creatinine, newly added to the synthetic bundle) and coding questions to
  `icd10_lookup`. All four MCP tools are exercised from the running system.
- **M3** — Added `make_vector_store(embedder, backend, url)` factory honoring
  `Settings.vector_backend`; `QdrantVectorStore` is now a complete (guarded)
  implementation, not a stub. `build_default_service` uses the factory.
- **L1** — The pipeline now gates on the same `ragas_metrics.faithfulness` the eval
  measures (broke the resulting import cycle with a `TYPE_CHECKING` guard in harness).
- **L2** — Added `src/caregraph/py.typed` and packaging config.
- **L3** — New `make_split` holds one paraphrase per label out of training; fine-tuned
  F1 is now a believable **0.860** (vs keyword 0.462), reflecting generalization.
- **L4** — `float(egfr)` cast, typed `CompiledGraph` return, real Qdrant method bodies.
