from datetime import datetime, timedelta

import pytest
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from agent_ia_langgraph.tools import TOOLS, get_current_time


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.name)
def test_every_parameter_of_every_tool_is_described_to_the_model(
    tool: BaseTool,
) -> None:
    # The model only sees this schema: an undescribed parameter leaves it guessing.
    parameters = convert_to_openai_tool(tool)["function"]["parameters"]

    for name, schema in parameters["properties"].items():
        assert schema.get("description"), f"{tool.name}: {name} has no description"


def test_get_current_time_gives_the_time_in_the_requested_timezone() -> None:
    # Tokyo has no daylight saving time, so its offset is always UTC+9.
    result = get_current_time.invoke({"timezone": "Asia/Tokyo"})

    assert datetime.fromisoformat(result).utcoffset() == timedelta(hours=9)
