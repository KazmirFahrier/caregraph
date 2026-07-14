"""Phase-5 demo: the service core + tracing. Runs offline, no server needed.

    python scripts/serve_demo.py

Shows the same code path the FastAPI app calls, plus the recorded trace (retrieve /
generate spans with latency + faithfulness) that would export to Arize Phoenix.
"""
from __future__ import annotations

from caregraph.serve import build_default_service


def main() -> None:
    service = build_default_service()

    print("=" * 72)
    print("CareGraph — Phase 5 service + tracing")
    print("=" * 72)
    print(f"health: {service.health()}")

    resp = service.ask("What blood pressure target is recommended for this CKD patient?")
    print("\nPOST /ask")
    print(f"  answer      : {resp.answer[:80]}...")
    print(f"  citations   : {resp.citations}")
    print(f"  faithfulness: {resp.faithfulness:.2f}   latency: {resp.latency_ms} ms   "
          f"refused: {resp.refused}")
    print("  trace (spans exported to Phoenix / OTLP):")
    for span in resp.trace:
        attrs = {k: v for k, v in span["attributes"].items()}
        print(f"    - {span['name']:<22} {span['duration_ms']:.2f} ms  {attrs}")

    agent = service.agent_ask("What is the eGFR and does kidney function need referral?")
    print("\nPOST /agent/ask")
    print(f"  status      : {agent['status']}   refused: {agent['refused']}")
    print(f"  tool_calls  : {[c['tool'] for c in agent['tool_calls']]}")

    print("\nAudit log (HIPAA):")
    for entry in service.audit_log:
        print(f"  {entry}")


if __name__ == "__main__":
    main()
