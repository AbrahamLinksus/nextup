"""Source connectors and the trigger registry.

Registering a source is one call. Everything derived from the registry -- the
poll schedule, the agent's fetch tools, on-demand routing -- follows without a
second edit anywhere.
"""

from assistant.config import get_settings
from assistant.connectors.base import ConnectorRegistry, SourceConnector, TriggerConfig
from assistant.connectors.gmail import GmailConnector
from assistant.connectors.manual import ManualConnector, item_from_file, make_item
from assistant.connectors.notion import NotionConnector


def default_registry() -> ConnectorRegistry:
    """The live sources for v1.

    Notion polls every 12 hours; Gmail is push-driven and declares no interval.
    The asymmetry is the decision, not an oversight: self-authored notes tolerate
    a staleness window that inbound deadlines do not.
    """
    settings = get_settings()
    registry = ConnectorRegistry()
    registry.register(NotionConnector(), poll_interval_hours=settings.notion_poll_interval_hours)
    registry.register(GmailConnector(), poll_interval_hours=None)
    return registry


__all__ = [
    "ConnectorRegistry",
    "GmailConnector",
    "ManualConnector",
    "NotionConnector",
    "SourceConnector",
    "TriggerConfig",
    "default_registry",
    "item_from_file",
    "make_item",
]
