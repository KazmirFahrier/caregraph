"""Phase-2 demo: the LangGraph-style clinical agent + MCP tools. Runs offline.

    python scripts/agent_demo.py
"""
from __future__ import annotations

from caregraph.agent import build_agent
from caregraph.data import guideline_corpus, load_bundle, synthetic_patient
from caregraph.llm import ScriptedClient
from caregraph.rag import Document, HashingEmbedder, InMemoryVectorStore
from caregraph.tools import ToolServer, register_clinical_tools


def build() -> tuple[object, ToolServer]:
    store = InMemoryVectorStore(HashingEmbedder(dim=256))
    store.add([Document(g["doc_id"], g["text"], {}) for g in guideline_corpus()])
    patient = load_bundle(synthetic_patient())
    server = ToolServer()
    register_clinical_tools(server, {patient.patient_id: patient})
    return build_agent(store, ScriptedClient(), server, patient), server


def main() -> None:
    agent, server = build()

    print("=" * 72)
    print("CareGraph — Phase 2 agent + MCP tools")
    print("=" * 72)
    print("\nMCP tools/list:")
    for t in server.list_tools():
        print(f"  - {t['name']}: {t['description']}")

    print("\n--- Informational question (no sign-off needed) ---")
    r = agent.run("What is the eGFR and what blood pressure target applies for this CKD patient?")
    print(f"answer     : {r.answer}")
    print(f"tool_calls : {[c['tool'] + '->' + c['summary'] for c in r.tool_calls]}")
    print(f"citations  : {r.citations}  faithfulness={r.faithfulness:.2f}")

    print("\n--- Actionable request (human-in-the-loop interrupt) ---")
    state = agent.run_raw("Should we prescribe an ACE inhibitor to lower the blood pressure?")
    print(f"status     : {state['status']}  (paused before human_review)")
    print(f"draft held : {state.get('answer')[:90]}...")
    approved = agent.approve(state, approved=True)
    print(f"after clinician approval -> status={approved.status}, refused={approved.refused}")
    print(f"answer     : {approved.answer}")

    print("\n--- Out-of-scope question (refused) ---")
    r = agent.run("What is the weather in Paris today?")
    print(f"refused    : {r.refused}  answer={r.answer}")


if __name__ == "__main__":
    main()
