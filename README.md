# CareGraph

A HIPAA-aware clinical intelligence agent built on a modern healthcare-AI stack:
**LlamaIndex-style RAG over a managed vector DB, LangGraph
agent orchestration over MCP tools, LoRA/QLoRA fine-tuning, Ragas evaluation, and
HIPAA-grade PHI handling.**

All patient data is **synthetic** (Synthea-style). No real PHI ever enters the repo.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Synthetic FHIR data + guideline corpus | ✅ implemented |
| 1 | Clinical RAG + managed-vector-DB interface + PHI redaction + eval stub | ✅ implemented |
| 2 | LangGraph-style agent + MCP tool server + human-in-the-loop | ✅ implemented |
| 3 | LoRA/QLoRA fine-tuning + base-vs-prompt-vs-tuned ablation | ✅ implemented |
| 4 | Ragas adapter + adversarial red-team + CI regression gate | ✅ implemented |
| 5 | FastAPI serving + Docker/K8s + OTel/Phoenix tracing + audit log | ✅ implemented |
| 6 | HIPAA hardening (deeper access control, threat model) | 🔜 planned |

Five dedicated defect-finding passes (`AUDIT.md` … `AUDIT5.md`) surfaced 27 issues — a
red CI, a PHI redactor wrong in both directions, hardcoded tool inputs, dead tool
paths, an actionable recommendation that leaked past the human-in-the-loop gate, on-
topic retrieved content that poisoned cited answers, tool outputs that leaked dates
past the PHI gate, a union-based grounding metric that would pass a real-LLM
recombination, and a sign-off gate with no clinician identity — all now fixed and
locked with regression tests. Each pass re-audited the code the previous one changed;
the HIGH-severity count fell to zero by the fourth pass (trend: HIGH 2 → 1 → 1 → 0 → 0),
and later findings shifted from bugs to design/policy hardening.

## Design principle (carried over from campaign-copilot)

**Claims must be checkable; failures must be loud.** The PHI redactor blocks rather
than logs. The RAG answer contract requires citations or refuses. The eval oracle
must score perfectly on a known-answer set or the harness is measuring itself.

## Quickstart

```bash
python -m pip install -e .          # optional; core slice runs on the stdlib
python scripts/demo.py              # Phase-1 RAG demo, no API keys needed
python scripts/agent_demo.py        # Phase-2 agent + MCP tools + human-in-the-loop
python scripts/finetune_demo.py     # Phase-3 fine-tuning ablation (offline, no GPU)
python scripts/eval_gate.py         # Phase-4 eval + red-team CI gate (exit 1 on regression)
python scripts/serve_demo.py        # Phase-5 service core + trace spans (offline)
uvicorn caregraph.serve.app:app     # Phase-5 real API (needs: pip install caregraph[serve])
python -m pytest -q                 # 64 tests
ruff check src tests && mypy --strict src/caregraph   # both clean
```

The core slice runs with **zero external services and no API keys** via a
deterministic scripted LLM and a hashing embedder. Real backends (Qdrant,
Pinecone, OpenAI/HF embeddings, Presidio) are wired behind the same interfaces and
activate automatically when their packages/credentials are present.

## Layout

```
src/caregraph/
  llm/          provider-agnostic LLM client + deterministic ScriptedClient
  phi/          PHI redaction guardrail (regex now, Presidio-ready)
  data/         FHIR models + synthetic patient/guideline generation
  rag/          embeddings, vector store (InMemory + Qdrant adapter), pipeline
  tools/        MCP-style tool server + clinical tools (FHIR, ICD-10, eGFR, CHA2DS2-VASc)
  agent/        LangGraph-style state machine + clinical agent (human-in-the-loop)
  finetune/     synthetic dataset, trainable extractor + baselines, real QLoRA script
  eval/         Ragas metrics + adapter, red-team suite, CI regression gate
  obs/          OpenTelemetry/Phoenix-style tracer (retrieve/generate spans)
  serve/        framework-free service core + FastAPI shell + audit log

Deployment: Dockerfile, docker-compose.yml (api + Qdrant + Phoenix),
deploy/k8s/ (Deployment, Service, NetworkPolicy with least-privilege egress).
```
