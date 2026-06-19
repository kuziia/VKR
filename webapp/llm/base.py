"""LLM provider Protocol — see plan §2.6.4.

Allows swapping Claude OAuth (subscription) → Anthropic API → local models
without touching call sites in search/agents.
"""
from __future__ import annotations

from typing import AsyncIterator, Protocol


class LLMNotConfigured(RuntimeError):
    """Raised when no usable LLM provider is wired up (no token / no SDK)."""


class LLMRateLimited(RuntimeError):
    """Raised on subscription / API rate-limit."""


class LLMClient(Protocol):
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str: ...

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]: ...

    async def complete_json(
        self,
        prompt: str,
        *,
        schema: dict,
        model: str = "claude-sonnet-4-6",
    ) -> dict: ...


def get_llm() -> LLMClient:
    """Resolve the configured LLM provider.

    Raises `LLMNotConfigured` if the SDK is missing or no token in env.
    Lazy import: dashboard/article endpoints don't depend on this.
    """
    from .claude import ClaudeClient

    return ClaudeClient.from_env()
