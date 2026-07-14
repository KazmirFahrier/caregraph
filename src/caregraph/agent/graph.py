"""A minimal LangGraph-compatible state machine.

Mirrors the LangGraph API surface — `add_node`, `add_edge`,
`add_conditional_edges`, `START`/`END`, and `interrupt_before` for human-in-the-loop
— so the clinical agent can be ported to `langgraph.graph.StateGraph` by changing
imports only. Kept dependency-free so the agent runs and is unit-tested in CI.

State is a plain dict; each node returns a partial dict that is merged in (the
LangGraph reducer convention).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

START = "__start__"
END = "__end__"

Node = Callable[[dict[str, Any]], dict[str, Any]]
Router = Callable[[dict[str, Any]], str]


@dataclass
class CompiledGraph:
    nodes: dict[str, Node]
    edges: dict[str, str]
    conditional: dict[str, Router]
    entry: str
    interrupt_before: set[str]

    def invoke(self, state: dict[str, Any], _resume: bool = False) -> dict[str, Any]:
        state = dict(state)
        current = state.pop("__resume_at__") if _resume and "__resume_at__" in state else self.entry
        while current != END:
            if current in self.interrupt_before and not state.get("__approved__"):
                # Pause for human review; caller inspects state and calls resume().
                state["status"] = "interrupted"
                state["__resume_at__"] = current
                return state
            update = self.nodes[current](state)
            state.update(update)
            current = self._next(current, state)
        state["status"] = "complete"
        state.pop("__approved__", None)
        state.pop("__resume_at__", None)
        return state

    def resume(self, state: dict[str, Any], approved: bool) -> dict[str, Any]:
        state = dict(state)
        state["__approved__"] = approved
        state["approved"] = approved
        return self.invoke(state, _resume=True)

    def _next(self, current: str, state: dict[str, Any]) -> str:
        if current in self.conditional:
            return self.conditional[current](state)
        return self.edges.get(current, END)


class StateGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, str] = {}
        self._conditional: dict[str, Router] = {}
        self._entry: str = START
        self._interrupt_before: set[str] = set()

    def add_node(self, name: str, fn: Node) -> None:
        self._nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        if src == START:
            self._entry = dst
        else:
            self._edges[src] = dst

    def add_conditional_edges(self, src: str, router: Router) -> None:
        self._conditional[src] = router

    def set_entry_point(self, name: str) -> None:
        self._entry = name

    def compile(self, interrupt_before: list[str] | None = None) -> CompiledGraph:
        return CompiledGraph(
            nodes=self._nodes,
            edges=self._edges,
            conditional=self._conditional,
            entry=self._entry,
            interrupt_before=set(interrupt_before or []),
        )
