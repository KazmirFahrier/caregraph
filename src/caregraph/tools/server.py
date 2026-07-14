"""MCP-style tool server.

Mirrors the Model Context Protocol surface (`list_tools` == MCP `tools/list`,
`call_tool` == MCP `tools/call`) with JSON-schema'd inputs and structured results.
The in-process `ToolServer` here is what a real `mcp.server.Server` would wrap: the
same `Tool` handlers are exposed over stdio/SSE in Phase 5 without changing callers.

Discipline carried from campaign-copilot: a tool that fails returns a typed error
result the agent can react to — it never raises into the orchestration loop, and the
agent gets exactly one recovery attempt per tool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def summary(self) -> str:
        if not self.ok:
            return f"ERROR: {self.error}"
        return "; ".join(f"{k}={v}" for k, v in self.data.items())


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., ToolResult]


class ToolServer:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def list_tools(self) -> list[dict[str, Any]]:
        """MCP tools/list shape."""
        return [
            {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
            for t in self._tools.values()
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """MCP tools/call shape. Unknown tools and handler crashes become typed errors."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"unknown tool: {name}")
        try:
            return tool.handler(**arguments)
        except Exception as exc:  # noqa: BLE001 - surfaced to agent as recoverable
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
