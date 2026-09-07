"""Measurement.

Two things are worth testing here and they are not the arithmetic. The first is
attribution: the pipeline's embed leg runs in a task started beside the
classifier, and if a `ContextVar` did not follow it there, every item would
under-report the half of its latency that was overlapped. The second is that the
per-item row lands in the same transaction as the item -- metrics describing
work that got rolled back would be a record of something that never happened.

The Prometheus formatting is tested for shape rather than for bytes: what breaks
in exposition format is a missing `+Inf` bucket or a non-cumulative count, not a
space in the wrong place.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from assistant import metrics
from assistant import repository as repo
from assistant.metrics import Histogram, Registry


@pytest.fixture(autouse=True)
def clean_registry():
    """The registry is process-global, so a test that reads it needs it empty."""
    metrics.REGISTRY.reset()
    yield
    metrics.REGISTRY.reset()


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

def test_counters_are_keyed_by_their_labels():
    registry = Registry()
    registry.incr("llm_calls", provider="ollama", operation="structured")
    registry.incr("llm_calls", provider="ollama", operation="structured")
    registry.incr("llm_calls", provider="ollama", operation="converse")

    snapshot = registry.snapshot()["counters"]["llm_calls"]

    assert snapshot["operation=structured,provider=ollama"] == 2
    assert snapshot["operation=converse,provider=ollama"] == 1


def test_label_order_does_not_create_a_second_series():
    """Otherwise the same metric splits in two depending on keyword order."""
    registry = Registry()
    registry.incr("x", provider="a", operation="b")
    registry.incr("x", operation="b", provider="a")

    assert list(registry.snapshot()["counters"]["x"].values()) == [2]


def test_a_histogram_reports_the_shape_of_what_it_saw():
    histogram = Histogram()
    for value in (10, 20, 30, 40, 5000):
        histogram.observe(value)

    snapshot = histogram.snapshot()

    assert snapshot["count"] == 5
    assert snapshot["min_ms"] == 10
    assert snapshot["max_ms"] == 5000
    # p50 lands in a small bucket even though the mean is dragged up by the tail
    assert snapshot["p50_ms"] <= 100
    assert snapshot["mean_ms"] > 1000


def test_an_empty_histogram_reports_zero_rather_than_dividing_by_zero():
    assert Histogram().snapshot()["mean_ms"] == 0.0
    assert Histogram().quantile(0.95) == 0.0


def test_prometheus_buckets_are_cumulative_and_end_at_inf():
    registry = Registry()
    for value in (10, 400, 9000):
        registry.observe("item_duration", value, source_type="notion")

    text = registry.prometheus()
    lines = [line for line in text.splitlines() if "_bucket" in line]
    counts = [int(line.rsplit(" ", 1)[1]) for line in lines]

    assert counts == sorted(counts), "buckets must be cumulative"
    assert counts[-1] == 3
    assert 'le="+Inf"' in lines[-1]
    assert "# TYPE assistant_item_duration_milliseconds histogram" in text


def test_label_values_are_escaped_in_exposition_format():
    registry = Registry()
    registry.incr("gate_decisions", reason='a "quoted" reason')

    assert '\\"quoted\\"' in registry.prometheus()


# ---------------------------------------------------------------------------
# Per-item attribution
# ---------------------------------------------------------------------------

def test_work_inside_the_block_is_attributed_to_the_item():
    with metrics.collect_item() as item:
        metrics.record_llm_call(
            provider="fake", model="m", operation="structured",
            milliseconds=120, input_tokens=300, output_tokens=40,
        )
        metrics.record_embedding(model="e", milliseconds=80, texts=6)

    assert item.llm_calls == 1
    assert item.llm_ms == 120
    assert item.total_tokens == 340
    assert item.embedded_texts == 6


def test_work_outside_any_block_still_reaches_the_registry():
    """The conversational agent has no item; its calls are process-level only."""
    metrics.record_llm_call(provider="fake", model="m", operation="converse", milliseconds=10)

    assert metrics.current_item() is None
    assert metrics.REGISTRY.snapshot()["counters"]["llm_calls"]


async def test_a_task_started_inside_the_block_reports_to_the_same_item():
    """The pipeline's embed leg runs beside the classifier, not after it.

    `asyncio.create_task` copies the context, so the task sees this same
    ItemMetrics object. Without that, overlapped work would vanish from the
    item's cost and every item would look cheaper than it was.
    """
    async def embed_leg():
        await asyncio.sleep(0)
        metrics.record_embedding(model="e", milliseconds=50, texts=3)

    with metrics.collect_item() as item:
        task = asyncio.create_task(embed_leg())
        metrics.record_llm_call(
            provider="fake", model="m", operation="structured", milliseconds=70
        )
        await task

    assert item.embed_calls == 1
    assert item.embedded_texts == 3
    assert item.llm_calls == 1


def test_nested_collection_does_not_leak_into_the_outer_item():
    with metrics.collect_item() as outer:
        with metrics.collect_item() as inner:
            metrics.record_embedding(model="e", milliseconds=10, texts=1)
        metrics.record_embedding(model="e", milliseconds=10, texts=2)

    assert inner.embedded_texts == 1
    assert outer.embedded_texts == 2


def test_stages_accumulate_by_name():
    with metrics.collect_item() as item:
        with metrics.stage("classify"):
            pass
        with metrics.stage("classify"):
            pass

    assert list(item.stages) == ["classify"]
    histograms = metrics.REGISTRY.snapshot()["histograms"]
    assert histograms["stage_duration"]["stage=classify"]["count"] == 2


def test_the_row_shape_is_what_the_repository_stores():
    with metrics.collect_item() as item:
        metrics.record_llm_call(
            provider="p", model="m", operation="structured",
            milliseconds=1500.7, input_tokens=10, output_tokens=2,
        )

    row = item.as_row()

    assert row["llm_ms"] == 1500  # integer milliseconds, as the column is
    assert row["input_tokens"] == 10
    assert isinstance(row["stages"], dict)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

@pytest.mark.db
async def test_a_processed_item_leaves_one_measurement_row(db):
    from assistant.actions import ExecutorPool
    from assistant.classify import SingleCallClassifier
    from assistant.pipeline import Pipeline
    from tests.conftest import FakeEmbedder, FakeProvider, make_item

    pipeline = Pipeline(
        db,
        classifier=SingleCallClassifier(
            FakeProvider([{"label": "informational", "confidence": 0.9, "rationale": "notes"}])
        ),
        provider=FakeProvider(),
        embedder=FakeEmbedder(),
        executors=ExecutorPool(),
    )
    await pipeline.process(make_item("# Notes\n\nReference material, nothing to do here."))

    rows = await (await db.execute("SELECT * FROM pipeline_metrics")).fetchall()

    assert len(rows) == 1
    assert rows[0]["outcome"] == "indexed"
    assert rows[0]["duration_ms"] >= 0
    assert rows[0]["llm_calls"] == 0, "the fake provider is not instrumented; the row still lands"
    assert "chunk" in rows[0]["stages"]


@pytest.mark.db
async def test_the_summary_aggregates_a_window(db):
    now = datetime.now(UTC)
    for index, (outcome, duration) in enumerate(
        [("acted", 1000), ("acted", 3000), ("queued", 2000), ("skipped", 10)]
    ):
        await repo.record_pipeline_metrics(
            db,
            item_key=f"notion:page-{index}",
            source_type="notion",
            outcome=outcome,
            label="actionable-mandatory",
            duration_ms=duration,
            detail={
                "llm_calls": 2, "llm_ms": duration // 2,
                "input_tokens": 100, "output_tokens": 20,
                "embed_calls": 1, "embed_ms": 5, "embedded_texts": 3,
                "stages": {"classify": duration / 2},
            },
        )

    summary = await repo.metrics_summary(db, since=now - timedelta(days=1))

    assert summary["totals"]["items"] == 4
    assert summary["totals"]["llm_calls"] == 8
    assert summary["totals"]["input_tokens"] == 400
    assert summary["totals"]["p50_ms"] > 0
    assert {row["outcome"] for row in summary["by_outcome"]} == {"acted", "queued", "skipped"}
    assert summary["stages"][0]["stage"] == "classify"


@pytest.mark.db
async def test_model_wait_never_exceeds_the_wall_clock(db):
    """Embedding overlaps classification, so the two cannot be added.

    An earlier version summed them and reported 102% of wall clock -- the
    overlapped second counted twice. Reasoning calls within one item are
    sequential, so that leg alone is a real fraction of the elapsed time.
    """
    await repo.record_pipeline_metrics(
        db,
        item_key="notion:page-1",
        source_type="notion",
        outcome="acted",
        label=None,
        duration_ms=5000,
        # Both legs nearly fill the item's wall clock because they ran at once.
        detail={"llm_calls": 3, "llm_ms": 4900, "embed_ms": 1600},
    )

    summary = await repo.metrics_summary(db, since=datetime.now(UTC) - timedelta(days=1))

    assert summary["totals"]["model_wait_pct"] <= 100
    assert summary["totals"]["embed_ms"] == 1600


@pytest.mark.db
async def test_the_summary_ignores_rows_outside_the_window(db):
    """The readout is "the last N days", so an old row must not be averaged in."""
    await repo.record_pipeline_metrics(
        db, item_key="notion:old", source_type="notion", outcome="acted",
        label=None, duration_ms=99999, detail={},
    )
    await db.execute("UPDATE pipeline_metrics SET created_at = now() - interval '30 days'")

    summary = await repo.metrics_summary(db, since=datetime.now(UTC) - timedelta(days=7))

    assert summary["totals"]["items"] == 0
