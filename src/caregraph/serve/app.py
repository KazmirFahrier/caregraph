"""FastAPI shell over `CareGraphService`.

Import-guarded so the core installs/tests without FastAPI. Run the service with:

    pip install "caregraph[serve]"
    uvicorn caregraph.serve.app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health         liveness/readiness probe (used by the K8s manifests)
    GET  /tools          MCP tools/list passthrough
    POST /ask            grounded clinical RAG answer + trace
    POST /agent/ask      full agent run; actionable requests return a resume_token
                         and a withheld answer (no recommendation leaks)
    POST /agent/approve  clinician sign-off that resumes an interrupted run
"""
from __future__ import annotations

from typing import Any

from caregraph.serve.service import build_default_service


def create_app() -> Any:
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover - only without the serve extra
        raise RuntimeError("The API needs pip install 'caregraph[serve]'") from exc

    import os

    app = FastAPI(title="CareGraph", version="0.1.0")
    service = build_default_service()
    api_key = os.getenv("CAREGRAPH_API_KEY")

    def require_auth(x_api_key: str | None = Header(default=None)) -> str:
        # A5-L1: gate every data endpoint. If CAREGRAPH_API_KEY is unset the caller
        # identity is "anonymous" (dev only) — set the key before any real PHI flows.
        if api_key is not None and x_api_key != api_key:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return x_api_key or "anonymous"

    class Query(BaseModel):
        question: str

    class Approval(BaseModel):
        resume_token: str
        approved: bool = True
        approver_id: str  # authenticated clinician identity (A5-M2), required

    @app.get("/health")
    def health() -> dict[str, str]:
        return service.health()

    @app.get("/tools")
    def tools(_: str = Depends(require_auth)) -> list[dict[str, Any]]:
        return service.list_tools()

    @app.post("/ask")
    def ask(q: Query, _: str = Depends(require_auth)) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(service.ask(q.question))

    @app.post("/agent/ask")
    def agent_ask(q: Query, _: str = Depends(require_auth)) -> dict[str, Any]:
        return service.agent_ask(q.question)

    @app.post("/agent/approve")
    def agent_approve(a: Approval, _: str = Depends(require_auth)) -> dict[str, Any]:
        try:
            return service.approve(a.resume_token, a.approved, a.approver_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    return app


# Module-level ASGI app for `uvicorn caregraph.serve.app:app`.
try:  # pragma: no cover - exercised only with the serve extra installed
    app = create_app()
except RuntimeError:
    app = None
