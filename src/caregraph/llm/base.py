"""Provider-agnostic LLM interface.

Real providers (OpenAI, Anthropic, HF endpoints, a fine-tuned local model) implement
the same `LLMClient` protocol as the deterministic `ScriptedClient` used in tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, messages: list[Message]) -> str:
        """Return the assistant completion for a chat transcript."""
        ...
