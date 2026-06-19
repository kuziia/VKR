"""ClaudeClient — wraps `claude-agent-sdk` with OAuth token from env.

Lazy SDK import so the rest of the app runs without the SDK installed.
Fails clearly via `LLMNotConfigured` if SDK absent or token missing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from typing import AsyncIterator

from ..settings import settings
from .base import LLMClient, LLMNotConfigured, LLMRateLimited

log = logging.getLogger(__name__)

_SEM = asyncio.Semaphore(8)  # plan §2.6.3: 8-way max for subscription

_RATE_LIMIT_MARKERS = ("rate_limit", "rate limit", "429", "usage_limit", "5h_limit")


def _looks_rate_limited(msg: str) -> bool:
    low = msg.lower()
    return any(m in low for m in _RATE_LIMIT_MARKERS)


class ClaudeClient(LLMClient):
    def __init__(self, oauth_token: str) -> None:
        if not oauth_token:
            raise LLMNotConfigured(
                "ANTHROPIC_OAUTH_TOKEN is empty. Run `claude setup-token` "
                "on a non-RU server and put the token into .env"
            )
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as e:
            raise LLMNotConfigured(
                "claude-agent-sdk is not installed. "
                "Run: pip install claude-agent-sdk"
            ) from e
        self._token = oauth_token

    @classmethod
    def from_env(cls) -> "ClaudeClient":
        return cls(settings.anthropic_oauth_token)

    async def _query(
        self,
        prompt: str,
        *,
        system: str | None,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield text chunks from claude-agent-sdk `query()`."""
        from claude_agent_sdk import (  # type: ignore
            ClaudeAgentOptions,
            query,
        )

        opts_kwargs: dict = {
            "model": model,
            "max_turns": 1,
            "permission_mode": "bypassPermissions",
        }
        if system:
            opts_kwargs["system_prompt"] = system

        # The bundled Claude Code CLI (used by claude-agent-sdk) authenticates
        # via CLAUDE_CODE_OAUTH_TOKEN, not ANTHROPIC_OAUTH_TOKEN.
        env = {"CLAUDE_CODE_OAUTH_TOKEN": self._token}
        opts_kwargs["env"] = env

        # The SDK's bundled CLI is a Bun binary requiring AVX; on CPUs without
        # AVX it crashes (SIGILL). Use the Node.js npm CLI instead when found.
        cli_path = settings.claude_cli_path or shutil.which("claude")
        if cli_path:
            opts_kwargs["cli_path"] = cli_path

        try:
            options = ClaudeAgentOptions(**opts_kwargs)
        except TypeError:
            valid = {k: v for k, v in opts_kwargs.items() if k in ("model", "system_prompt")}
            options = ClaudeAgentOptions(**valid)

        try:
            async for msg in query(prompt=prompt, options=options):
                # SDK message types vary; coerce to text best-effort.
                content = getattr(msg, "content", None)
                if content is None and isinstance(msg, dict):
                    content = msg.get("content")
                if content is None:
                    continue
                if isinstance(content, str):
                    yield content
                    continue
                # content blocks
                if isinstance(content, list):
                    for block in content:
                        text = getattr(block, "text", None)
                        if text is None and isinstance(block, dict):
                            text = block.get("text")
                        if text:
                            yield text
        except Exception as e:
            msg = str(e)
            if _looks_rate_limited(msg):
                raise LLMRateLimited(f"Claude rate-limited: {msg}") from e
            raise

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        async with _SEM:
            chunks: list[str] = []
            async for c in self._query(
                prompt,
                system=system,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                chunks.append(c)
            return "".join(chunks).strip()

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        async with _SEM:
            async for c in self._query(
                prompt,
                system=system,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield c

    async def complete_json(
        self,
        prompt: str,
        *,
        schema: dict,
        model: str = "claude-sonnet-4-6",
    ) -> dict:
        sys_msg = (
            "Return ONLY valid JSON matching the requested schema, no prose, "
            "no markdown fences."
        )
        text = await self.complete(prompt, system=sys_msg, model=model, temperature=0.0)
        return _parse_json_loose(text)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_json_loose(text: str) -> dict:
    s = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        i = s.find("{")
        j = s.rfind("}")
        if 0 <= i < j:
            return json.loads(s[i : j + 1])
        raise
