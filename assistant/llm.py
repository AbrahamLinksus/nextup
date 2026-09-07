"""Reasoning-model access, behind one interface with two implementations.

The design left the model choice open -- self-hosted (no credentials, no
per-call cost) versus a hosted API (markedly better structured tool-calling at
this volume). Rather than settle it by fiat, both live behind `LLMProvider` and
LLM_PROVIDER picks. `ollama` is the default so a fresh clone runs with no
credentials at all.

Three capabilities, because the system asks the model for exactly three things:

* `structured` -- one call, one JSON object matching a schema. The classifier's
  entire surface. No tool involved: classification picks a label, it does not
  pick an action.
* `choose_tool` -- one call, zero or one tool proposal. The operations agent's
  entire surface. Returning *nothing* is a first-class outcome: "no registered
  tool fits" needs no separate reasoning step to detect, it is the absence of a
  call.
* `converse` -- genuine multi-turn tool-calling for the user-facing agent. This
  is the only place the model gets real freedom, and the only place it should.

The asymmetry is the design: the background pipeline is deterministic and uses
the two narrow calls; the chat layer is agentic and uses the third.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Self

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from assistant import metrics
from assistant.config import get_settings

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any]
    """JSON Schema for the arguments object."""

    def as_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def as_ollama(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class AssistantTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# Provider-neutral conversation history. Providers translate on the way out;
# nothing above this module ever sees a provider's message shape.
#   {"role": "user",      "content": str}
#   {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
#   {"role": "tool",      "call_id": str, "name": str, "content": str}
Message = dict[str, Any]


class LLMProvider(ABC):
    """One reasoning model, three shapes of request."""

    name: str

    @abstractmethod
    async def structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Return one JSON object conforming to `schema`.

        `temperature` defaults to 0 -- a triage decision that varies run to run
        is not debuggable. It is exposed only because the self-consistency
        classifier needs sampling variance to have something to measure
        agreement across.
        """

    @abstractmethod
    async def choose_tool(
        self, *, system: str, prompt: str, tools: list[ToolSchema], max_tokens: int = 2048
    ) -> ToolCall | None:
        """Propose at most one tool call. None means nothing registered fits."""

    @abstractmethod
    async def converse(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
        max_tokens: int = 4096,
    ) -> AssistantTurn:
        """One assistant turn, possibly requesting tools."""

    async def close(self) -> None:  # pragma: no cover - trivial
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """Local models via Ollama's /api/chat.

    `structured` uses Ollama's `format` parameter with the JSON schema rather
    than a forced tool call. For a 7B model that difference is large: schema-
    constrained decoding cannot emit a malformed label, whereas a tool call it
    was merely asked to make is something it can decline to make.
    """

    name = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_reasoning_model
        self.num_ctx = settings.reasoning_num_ctx
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    async def _chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json()

    async def _measured_chat(self, payload: dict[str, Any], operation: str) -> dict[str, Any]:
        """`_chat` with the clock and the token counters around it.

        Wrapping rather than instrumenting `_chat` itself so the measurement
        covers retries: three attempts at 4s each is a 12s wait for the item,
        and reporting only the successful attempt would describe a system that
        is faster than the one anybody is using.
        """
        with metrics.timed() as elapsed:
            data = await self._chat(payload)
        metrics.record_llm_call(
            provider=self.name,
            model=self.model,
            operation=operation,
            milliseconds=elapsed[0],
            # Ollama reports tokens as evaluation counts: prompt_eval_count is
            # what it read, eval_count is what it generated.
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
        )
        return data

    def _options(self) -> dict[str, Any]:
        # temperature 0: triage decisions should not vary run to run, and a
        # reproducible classification is a debuggable one.
        return {"temperature": 0.0, "num_ctx": self.num_ctx}

    async def structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        data = await self._measured_chat(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "format": schema,
                "options": {
                    **self._options(),
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            "structured",
        )
        content = data["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                f"{self.model} returned non-JSON under a schema constraint: {content[:200]!r}"
            ) from exc

    async def choose_tool(
        self, *, system: str, prompt: str, tools: list[ToolSchema], max_tokens: int = 2048
    ) -> ToolCall | None:
        data = await self._measured_chat(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "tools": [tool.as_ollama() for tool in tools],
                "options": {**self._options(), "num_predict": max_tokens},
            },
            "choose_tool",
        )
        calls = _ollama_tool_calls(data.get("message", {}))
        return calls[0] if calls else None

    async def converse(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
        max_tokens: int = 4096,
    ) -> AssistantTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *_to_ollama(messages)],
            "stream": False,
            "options": {**self._options(), "num_predict": max_tokens},
        }
        if tools:
            payload["tools"] = [tool.as_ollama() for tool in tools]

        message = (await self._measured_chat(payload, "converse")).get("message", {})
        return AssistantTurn(
            text=message.get("content", "") or "",
            tool_calls=_ollama_tool_calls(message),
        )


def _ollama_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function", {})
        arguments = function.get("arguments", {})
        # Ollama usually hands back a dict, but some models emit a JSON string.
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                log.warning("ollama.tool_arguments_unparseable", raw=arguments[:200])
                continue
        calls.append(ToolCall(name=function.get("name", ""), arguments=arguments))
    return calls


def _to_ollama(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            out.append(
                {
                    "role": "tool",
                    "content": message["content"],
                    "tool_name": message.get("name", ""),
                }
            )
            continue
        entry: dict[str, Any] = {"role": role, "content": message.get("content", "")}
        if message.get("tool_calls"):
            entry["tool_calls"] = [
                {"function": {"name": call.name, "arguments": call.arguments}}
                for call in message["tool_calls"]
            ]
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Claude via the official SDK.

    `structured` uses `output_config.format`, and tools are declared `strict` so
    a proposal that reaches the gate is at least guaranteed to be shaped like the
    action it claims to be. That leaves the gate free to judge the *content* of
    a proposal rather than spend its budget on whether the JSON parsed.
    """

    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        import anthropic

        settings = get_settings()
        self.model = model or settings.anthropic_model
        self._client = anthropic.AsyncAnthropic()

    async def close(self) -> None:
        await self._client.close()

    async def _measured(self, operation: str, **kwargs: Any) -> Any:
        """One `messages.create`, timed, with the usage block reported.

        Anthropic returns exact token counts rather than the evaluation counts
        Ollama reports, so this is the provider whose numbers can be turned into
        a currency figure. Both land in the same counters -- the point of the
        provider abstraction is that the cost question is asked once.
        """
        with metrics.timed() as elapsed:
            response = await self._client.messages.create(**kwargs)
        usage = getattr(response, "usage", None)
        metrics.record_llm_call(
            provider=self.name,
            model=self.model,
            operation=operation,
            milliseconds=elapsed[0],
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )
        return response

    async def structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        # Sampling parameters are rejected on current Claude models; variance for
        # the self-consistency strategy comes from the model's own adaptive
        # thinking rather than from a temperature knob that no longer exists.
        response = await self._measured(
            "structured",
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        _guard_refusal(response)
        text = next((block.text for block in response.content if block.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                f"{self.model} returned non-JSON under output_config: {text[:200]!r}"
            ) from exc

    async def choose_tool(
        self, *, system: str, prompt: str, tools: list[ToolSchema], max_tokens: int = 2048
    ) -> ToolCall | None:
        response = await self._measured(
            "choose_tool",
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            tools=[{**tool.as_anthropic(), "strict": True} for tool in tools],
            # Auto, never forced. A forced call would make "nothing here fits"
            # unrepresentable, and the model would invent an action to satisfy
            # the requirement -- exactly the failure the queue exists to avoid.
            tool_choice={"type": "auto"},
        )
        _guard_refusal(response)
        for block in response.content:
            if block.type == "tool_use":
                return ToolCall(name=block.name, arguments=dict(block.input), call_id=block.id)
        return None

    async def converse(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
        max_tokens: int = 4096,
    ) -> AssistantTurn:
        response = await self._measured(
            "converse",
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=_to_anthropic(messages),
            tools=[tool.as_anthropic() for tool in tools] if tools else [],
        )
        _guard_refusal(response)

        turn = AssistantTurn()
        for block in response.content:
            if block.type == "text":
                turn.text += block.text
            elif block.type == "tool_use":
                turn.tool_calls.append(
                    ToolCall(name=block.name, arguments=dict(block.input), call_id=block.id)
                )
        return turn


def _guard_refusal(response: Any) -> None:
    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        raise LLMRefusalError(getattr(details, "explanation", "model declined the request"))


def _to_anthropic(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate neutral history into Anthropic content blocks.

    Consecutive `tool` messages collapse into a single user message. Splitting
    parallel tool results across separate messages is accepted by the API but
    teaches the model to stop making parallel calls, so the grouping is load-
    bearing rather than cosmetic.
    """
    out: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush_results() -> None:
        if pending_results:
            out.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for message in messages:
        role = message["role"]
        if role == "tool":
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message["call_id"],
                    "content": message["content"],
                }
            )
            continue

        flush_results()
        if role == "user":
            out.append({"role": "user", "content": message["content"]})
            continue

        blocks: list[dict[str, Any]] = []
        if message.get("content"):
            blocks.append({"type": "text", "text": message["content"]})
        for call in message.get("tool_calls", []):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.call_id,
                    "name": call.name,
                    "input": call.arguments,
                }
            )
        out.append({"role": "assistant", "content": blocks})

    flush_results()
    return out


# ---------------------------------------------------------------------------
# Errors and construction
# ---------------------------------------------------------------------------

class LLMResponseError(RuntimeError):
    """The model returned something the caller cannot use."""


class LLMRefusalError(RuntimeError):
    """The model declined. Treated as a queue-worthy failure, never as a label."""


def make_provider(provider: str | None = None) -> LLMProvider:
    choice = (provider or get_settings().llm_provider).lower()
    if choice == "ollama":
        return OllamaProvider()
    if choice == "anthropic":
        return AnthropicProvider()
    raise ValueError(f"unknown LLM_PROVIDER={choice!r} (expected 'ollama' or 'anthropic')")
