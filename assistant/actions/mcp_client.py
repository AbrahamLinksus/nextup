"""A thin MCP stdio client, shared by every MCP-backed executor.

Destinations are reached through MCP servers rather than bespoke API clients so
the calendar, and later Gmail and Slack, reuse standardized integrations and
their existing auth instead of one hand-rolled SDK wrapper each.

The one thing this adds beyond `mcp`'s own session is **tool-name resolution**.
Different calendar MCP servers name the same operation `create_event`,
`create-event`, or `calendar_create_event`, and the schema is only knowable at
runtime. Rather than pin one name and fail opaquely against any other server,
this asks the server what it exposes and matches against candidates -- and when
nothing matches, the error names every tool the server actually has, which is
the information needed to fix the config.
"""

from __future__ import annotations

import json
import shlex
from contextlib import AsyncExitStack
from typing import Any, Self

import structlog

log = structlog.get_logger(__name__)


class MCPToolUnavailable(RuntimeError):
    """The configured server exposes nothing matching the operation we need."""


class MCPClient:
    def __init__(self, command: str, args: list[str] | None = None) -> None:
        self.command = command
        self.args = args or []
        self._stack: AsyncExitStack | None = None
        self._session: Any = None
        self._tool_names: list[str] = []

    @classmethod
    def from_settings(cls) -> Self | None:
        """Build from config, or None when no server is configured.

        None is the default, and it means every action stays a dry run. A fresh
        clone should not be able to write to a real calendar because someone ran
        the pipeline before reading the README.
        """
        from assistant.config import get_settings

        settings = get_settings()
        if not settings.calendar_mcp_command.strip():
            return None
        return cls(settings.calendar_mcp_command, shlex.split(settings.calendar_mcp_args))

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def connect(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(
            stdio_client(StdioServerParameters(command=self.command, args=self.args))
        )
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

        listed = await self._session.list_tools()
        self._tool_names = [tool.name for tool in listed.tools]
        log.info("mcp.connected", command=self.command, tools=self._tool_names)

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack, self._session = None, None

    @property
    def tool_names(self) -> list[str]:
        return list(self._tool_names)

    def resolve(self, candidates: tuple[str, ...]) -> str:
        """First candidate the server actually exposes, matched loosely.

        Loose matching (case- and separator-insensitive) because `create_event`
        and `create-event` are the same operation, and being strict here means
        an otherwise-working server is unusable over punctuation.
        """
        normalized = {_normalize(name): name for name in self._tool_names}
        for candidate in candidates:
            found = normalized.get(_normalize(candidate))
            if found:
                return found
        raise MCPToolUnavailable(
            f"none of {candidates} are exposed by {self.command!r}; "
            f"it offers: {self._tool_names}"
        )

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("MCPClient.call before connect()")

        result = await self._session.call_tool(tool, arguments)
        if getattr(result, "isError", False):
            raise RuntimeError(f"MCP tool {tool!r} failed: {_text_of(result)}")

        payload = _text_of(result)
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            # Many servers return prose. Keep it rather than fail: the caller
            # only needs an id when there is one, and the text is still the
            # honest record of what happened.
            return {"text": payload}
        return parsed if isinstance(parsed, dict) else {"result": parsed}


def _normalize(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(".", "")


def _text_of(result: Any) -> str:
    parts = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)
