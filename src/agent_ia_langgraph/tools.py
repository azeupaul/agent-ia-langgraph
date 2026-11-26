"""Tools the model can ask the agent to run."""

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import BaseTool, tool


@tool(parse_docstring=True)
def get_current_time(timezone: str) -> str:
    """Get the current date and time in a given timezone.

    Use it whenever the user asks for the time or date somewhere, since you
    cannot know the current time otherwise.

    Args:
        timezone: IANA timezone name, e.g. "Africa/Douala". Use the timezone
            of the main city if the user gives a country.

    Returns:
        The current date and time in ISO 8601 format, with the UTC offset.
    """
    return datetime.now(ZoneInfo(timezone)).isoformat()


TOOLS: tuple[BaseTool, ...] = (get_current_time,)
"""The tools the agent gives the model, in the order the model sees them."""
