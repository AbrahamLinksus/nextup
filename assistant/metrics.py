"""What the system actually cost, measured rather than estimated.

Two layers, because two different questions get asked and neither answers the
other:

**Live counters** (`REGISTRY`) are process-wide totals -- model calls, tokens,
stage latencies, gate decisions -- exposed as JSON and in Prometheus exposition
format. They answer "what is happening right now", and they reset when the
process does, which is the correct lifetime for that question.

**Per-item rows** are one durable record per processed item, written inside the
same transaction as everything else that item produced. They answer "what does
this cost per item, over a week of real mail" -- and that is a question no
in-memory counter can answer, because the interesting version of it is asked
after a restart.

The per-item collector is a `ContextVar`, so nothing in the call path has to
thread a metrics object through it. The one place that matters is the pipeline's
parallel embed leg: `asyncio.create_task` copies the current context, so the task
sees the *same* `ItemMetrics` object rather than a fresh one, and its timings
land on the item they belong to.

Instrumentation is deliberately confined to the boundaries where time is
actually spent -- the model, the embedder, and the pipeline's own stages.
Instrumenting further in would produce more numbers without producing more
insight, and every one of them would be a line to maintain.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "REGISTRY",
    "ItemMetrics",
    "collect_item",
    "current_item",
    "record_embedding",
    "record_gate_decision",
    "record_llm_call",
    "stage",
]

# Latency buckets in milliseconds. Chosen for what is being measured: a local
# 7B model answers in hundreds of milliseconds to tens of seconds, so buckets
# that stop at one second would put every real call in +Inf and measure nothing.
DEFAULT_BUCKETS = (50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0)

Labels = tuple[tuple[str, str], ...]


def _labels(mapping: dict[str, Any]) -> Labels:
    """Label sets are sorted so the same labels always key the same series."""
    return tuple(sorted((str(k), str(v)) for k, v in mapping.items()))


@dataclass
class Histogram:
    """Bucketed durations. Bucketed rather than a sample reservoir because the
    memory has to be bounded: this runs for weeks and nobody empties it."""

    buckets: tuple[float, ...] = DEFAULT_BUCKETS
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    observations: int = 0
    minimum: float = float("inf")
    maximum: float = 0.0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * (len(self.buckets) + 1)

    def observe(self, value: float) -> None:
        self.total += value
        self.observations += 1
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        for index, edge in enumerate(self.buckets):
            if value <= edge:
                self.counts[index] += 1
                return
        self.counts[-1] += 1

    @property
    def mean(self) -> float:
        return self.total / self.observations if self.observations else 0.0

    def quantile(self, q: float) -> float:
        """Bucket-interpolated quantile.

        Approximate by construction -- the true value is somewhere inside a
        bucket and this returns the bucket's upper edge. Good enough to answer
        "is p95 seconds or minutes", which is the only question asked of it
        here; the exact figures come from the per-item rows in Postgres.
        """
        if not self.observations:
            return 0.0
        target = q * self.observations
        seen = 0
        for index, count in enumerate(self.counts):
            seen += count
            if seen >= target:
                return self.buckets[index] if index < len(self.buckets) else self.maximum
        return self.maximum

    def snapshot(self) -> dict[str, Any]:
        return {
            "count": self.observations,
            "total_ms": round(self.total, 2),
            "mean_ms": round(self.mean, 2),
            "min_ms": round(self.minimum, 2) if self.observations else 0.0,
            "max_ms": round(self.maximum, 2),
            "p50_ms": round(self.quantile(0.50), 2),
            "p95_ms": round(self.quantile(0.95), 2),
        }


class Registry:
    """Process-wide counters and histograms.

    No locking: everything that writes here runs on one event loop, and the
    operations are single bytecode-level mutations of a dict entry. A lock would
    be honest about intent and dishonest about need.
    """

    def __init__(self) -> None:
        self.counters: dict[tuple[str, Labels], float] = {}
        self.histograms: dict[tuple[str, Labels], Histogram] = {}

    def incr(self, name: str, value: float = 1.0, **labels: Any) -> None:
        key = (name, _labels(labels))
        self.counters[key] = self.counters.get(key, 0.0) + value

    def observe(self, name: str, milliseconds: float, **labels: Any) -> None:
        key = (name, _labels(labels))
        histogram = self.histograms.get(key)
        if histogram is None:
            histogram = self.histograms[key] = Histogram()
        histogram.observe(milliseconds)

    def reset(self) -> None:
        self.counters.clear()
        self.histograms.clear()

    # -- readouts --------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """JSON-shaped, for `/metrics` and the dashboard."""
        counters: dict[str, Any] = {}
        for (name, labels), value in sorted(self.counters.items()):
            counters.setdefault(name, {})[_render(labels)] = value

        histograms: dict[str, Any] = {}
        for (name, labels), histogram in sorted(self.histograms.items()):
            histograms.setdefault(name, {})[_render(labels)] = histogram.snapshot()

        return {"counters": counters, "histograms": histograms}

    def prometheus(self) -> str:
        """Text exposition format.

        Written by hand rather than pulled in with a client library: it is a
        documented line format, this exports nine series, and a dependency whose
        job is string formatting is a dependency to keep up to date forever.
        """
        lines: list[str] = []
        for (name, labels), value in sorted(self.counters.items()):
            metric = f"assistant_{name}_total"
            lines.append(f"# TYPE {metric} counter")
            lines.append(f"{metric}{_prom_labels(labels)} {value:g}")

        for (name, labels), histogram in sorted(self.histograms.items()):
            metric = f"assistant_{name}_milliseconds"
            lines.append(f"# TYPE {metric} histogram")
            cumulative = 0
            for edge, count in zip(histogram.buckets, histogram.counts, strict=False):
                cumulative += count
                lines.append(f"{metric}_bucket{_prom_labels(labels, le=str(edge))} {cumulative}")
            lines.append(
                f"{metric}_bucket{_prom_labels(labels, le='+Inf')} {histogram.observations}"
            )
            lines.append(f"{metric}_sum{_prom_labels(labels)} {histogram.total:g}")
            lines.append(f"{metric}_count{_prom_labels(labels)} {histogram.observations}")

        return "\n".join(lines) + "\n"


def _render(labels: Labels) -> str:
    return ",".join(f"{k}={v}" for k, v in labels) or "-"


def _prom_labels(labels: Labels, **extra: str) -> str:
    pairs = [*labels, *sorted(extra.items())]
    if not pairs:
        return ""
    body = ",".join(f'{k}="{_escape(v)}"' for k, v in pairs)
    return "{" + body + "}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


REGISTRY = Registry()


# ---------------------------------------------------------------------------
# Per-item collection
# ---------------------------------------------------------------------------

@dataclass
class ItemMetrics:
    """Everything one item cost, accumulated as it moves through the pipeline."""

    llm_calls: int = 0
    llm_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    embed_calls: int = 0
    embed_ms: float = 0.0
    embedded_texts: int = 0
    stages: dict[str, float] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_row(self) -> dict[str, Any]:
        """The shape `repository.record_pipeline_metrics` stores."""
        return {
            "llm_calls": self.llm_calls,
            "llm_ms": int(self.llm_ms),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "embed_calls": self.embed_calls,
            "embed_ms": int(self.embed_ms),
            "embedded_texts": self.embedded_texts,
            "stages": {name: round(ms, 1) for name, ms in self.stages.items()},
        }


_CURRENT: ContextVar[ItemMetrics | None] = ContextVar("assistant_item_metrics", default=None)


def current_item() -> ItemMetrics | None:
    return _CURRENT.get()


@contextmanager
def collect_item() -> Iterator[ItemMetrics]:
    """Attribute everything measured inside this block to one item.

    Tasks started within the block inherit the context by copy, so the pipeline's
    concurrent embed leg reports into this same object -- which is the point,
    since its latency is part of the item's cost even though it runs beside the
    classifier rather than after it.
    """
    item = ItemMetrics()
    token = _CURRENT.set(item)
    try:
        yield item
    finally:
        _CURRENT.reset(token)


# ---------------------------------------------------------------------------
# Instrumentation helpers
# ---------------------------------------------------------------------------

@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time one named pipeline stage.

    A wall-clock measurement, deliberately: what matters is how long an item
    waits, and for a stage that is mostly a model call, wall clock *is* the cost.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - started) * 1000
        REGISTRY.observe("stage_duration", elapsed, stage=name)
        item = _CURRENT.get()
        if item is not None:
            item.stages[name] = item.stages.get(name, 0.0) + elapsed


def record_llm_call(
    *,
    provider: str,
    model: str,
    operation: str,
    milliseconds: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """One completed model call.

    `operation` is the shape of the request -- structured, choose_tool, converse
    -- not the caller. Three shapes have three different cost profiles, and
    knowing that classification is cheap while conversation is not is the whole
    reason to label at all.
    """
    REGISTRY.incr("llm_calls", provider=provider, model=model, operation=operation)
    REGISTRY.observe("llm_duration", milliseconds, provider=provider, operation=operation)
    if input_tokens:
        REGISTRY.incr("llm_input_tokens", input_tokens, provider=provider, model=model)
    if output_tokens:
        REGISTRY.incr("llm_output_tokens", output_tokens, provider=provider, model=model)

    item = _CURRENT.get()
    if item is not None:
        item.llm_calls += 1
        item.llm_ms += milliseconds
        item.input_tokens += input_tokens
        item.output_tokens += output_tokens


def record_embedding(*, model: str, milliseconds: float, texts: int) -> None:
    REGISTRY.incr("embed_calls", model=model)
    REGISTRY.incr("embedded_texts", texts, model=model)
    REGISTRY.observe("embed_duration", milliseconds, model=model)

    item = _CURRENT.get()
    if item is not None:
        item.embed_calls += 1
        item.embed_ms += milliseconds
        item.embedded_texts += texts


def record_gate_decision(*, allowed: bool, reason: str) -> None:
    """The gate's verdict, counted by reason.

    The most useful single number this system produces: the ratio of automatic
    actions to queued ones, broken down by *why* something queued. A confidence
    threshold that is too high shows up here as a pile of `low_confidence`
    entries long before it shows up as a complaint.
    """
    REGISTRY.incr("gate_decisions", outcome="allowed" if allowed else "queued", reason=reason)


def record_item_processed(*, source_type: str, outcome: str, milliseconds: float) -> None:
    REGISTRY.incr("items_processed", source_type=source_type, outcome=outcome)
    REGISTRY.observe("item_duration", milliseconds, source_type=source_type)


@contextmanager
def timed() -> Iterator[list[float]]:
    """Measure a block without naming a metric, for callers that report the
    duration themselves. The list holds one element once the block exits."""
    holder: list[float] = []
    started = time.perf_counter()
    try:
        yield holder
    finally:
        holder.append((time.perf_counter() - started) * 1000)
