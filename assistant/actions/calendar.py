"""Calendar execution: the v1 action type.

Two executors implement the same interface. `CalendarMCPActionExecutor` talks to
a real calendar through an MCP server. `DryRunCalendarExecutor` is what runs
when no server is configured -- it records exactly what it would have done and
hands back a `dryrun:`-prefixed id.

The dry-run path is the default rather than a testing convenience. Everything
upstream of here -- the confidence gate, the boundary bias, the injection check
-- exists because an automatic calendar write has no human in front of it. It
would be inconsistent to then have an unconfigured clone write to a real
calendar on first run.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import structlog

from assistant.actions.base import ActionExecutor, ActionResult
from assistant.config import get_settings
from assistant.dates import parse_iso_local
from assistant.models import ActionSpec

log = structlog.get_logger(__name__)

ACTION_TYPE = "create_calendar_event"

# Ordered by how commonly calendar MCP servers name the operation.
_CREATE_CANDIDATES = ("create_event", "create-event", "calendar_create_event", "add_event")
_UPDATE_CANDIDATES = ("update_event", "update-event", "calendar_update_event", "edit_event")
_DELETE_CANDIDATES = ("delete_event", "delete-event", "calendar_delete_event", "remove_event")

DEFAULT_DURATION = timedelta(hours=1)


def build_event(spec: ActionSpec) -> dict[str, Any]:
    """Map an ActionSpec onto a calendar event payload.

    Every datetime crosses into the configured timezone here rather than being
    passed through as written. A resolved date with no timezone is still
    ambiguous by the time a calendar API sees it, and 00:00 UTC is the previous
    evening in the only timezone this system runs in.
    """
    settings = get_settings()
    start = parse_iso_local(spec.value("start"))
    end = parse_iso_local(spec.value("end"))
    all_day = bool(spec.value("all_day", False))

    if start is None:
        raise ValueError("calendar action has no resolvable start")
    if end is None:
        end = start + DEFAULT_DURATION

    event: dict[str, Any] = {
        "calendarId": settings.calendar_id,
        "summary": spec.value("title") or "(untitled)",
    }
    if all_day:
        event["start"] = {"date": start.date().isoformat()}
        event["end"] = {"date": (end.date() or start.date()).isoformat()}
    else:
        event["start"] = {"dateTime": start.isoformat(), "timeZone": settings.timezone}
        event["end"] = {"dateTime": end.isoformat(), "timeZone": settings.timezone}

    if spec.value("location"):
        event["location"] = spec.value("location")
    if spec.value("description"):
        event["description"] = spec.value("description")
    return event


def summarize(spec: ActionSpec) -> str:
    start = parse_iso_local(spec.value("start"))
    when = start.strftime("%Y-%m-%d %H:%M %Z") if start else "unknown time"
    return f"calendar event: {spec.value('title') or '(untitled)'} at {when}"


class CalendarMCPActionExecutor(ActionExecutor):
    action_type = ACTION_TYPE

    def __init__(self, client: Any) -> None:
        self.client = client

    async def create(self, spec: ActionSpec) -> ActionResult:
        payload = await self.client.call(
            self.client.resolve(_CREATE_CANDIDATES), build_event(spec)
        )
        action_id = _extract_id(payload)
        log.info("calendar.created", action_id=action_id, spec=spec.action_type)
        return ActionResult(action_id=action_id, summary=summarize(spec), detail=payload)

    async def update(self, action_id: str, spec: ActionSpec) -> ActionResult:
        payload = await self.client.call(
            self.client.resolve(_UPDATE_CANDIDATES),
            {**build_event(spec), "eventId": action_id},
        )
        log.info("calendar.updated", action_id=action_id)
        return ActionResult(
            action_id=action_id, summary=f"updated {summarize(spec)}", detail=payload
        )

    async def delete(self, action_id: str) -> None:
        await self.client.call(
            self.client.resolve(_DELETE_CANDIDATES),
            {"calendarId": get_settings().calendar_id, "eventId": action_id},
        )
        log.info("calendar.deleted", action_id=action_id)


class DryRunCalendarExecutor(ActionExecutor):
    """Records intent without writing anywhere. The default.

    The `dryrun:` id prefix is deliberate: it flows into `item_lifecycle.action_id`
    and the audit log, so a later reconciliation that tries to update a dry-run
    event is obvious in the data rather than a silent no-op nobody notices.
    """

    action_type = ACTION_TYPE

    def __init__(self) -> None:
        self.performed: list[tuple[str, str, dict[str, Any]]] = []

    async def create(self, spec: ActionSpec) -> ActionResult:
        import uuid

        event = build_event(spec)
        action_id = f"dryrun:{uuid.uuid4().hex[:12]}"
        self.performed.append(("create", action_id, event))
        log.info("calendar.dry_run_create", action_id=action_id, payload=event)
        return ActionResult(
            action_id=action_id, summary=summarize(spec), detail=event, dry_run=True
        )

    async def update(self, action_id: str, spec: ActionSpec) -> ActionResult:
        event = build_event(spec)
        self.performed.append(("update", action_id, event))
        log.info("calendar.dry_run_update", action_id=action_id, payload=event)
        return ActionResult(
            action_id=action_id,
            summary=f"updated {summarize(spec)}",
            detail=event,
            dry_run=True,
        )

    async def delete(self, action_id: str) -> None:
        self.performed.append(("delete", action_id, {}))
        log.info("calendar.dry_run_delete", action_id=action_id)


def _extract_id(payload: dict[str, Any]) -> str:
    for key in ("id", "eventId", "event_id"):
        if payload.get(key):
            return str(payload[key])
    nested = payload.get("event")
    if isinstance(nested, dict):
        return _extract_id(nested)
    # An executor that cannot report an id has broken reconciliation: without
    # one, a later edit cannot update or delete what it created. Loud is right.
    raise RuntimeError(f"calendar MCP response carried no event id: {payload!r}")
