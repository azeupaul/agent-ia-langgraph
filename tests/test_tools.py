from datetime import datetime, timedelta

from agent_ia_langgraph.tools import get_current_time


def test_get_current_time_gives_the_time_in_the_requested_timezone() -> None:
    # Tokyo has no daylight saving time, so its offset is always UTC+9.
    result = get_current_time.invoke({"timezone": "Asia/Tokyo"})

    assert datetime.fromisoformat(result).utcoffset() == timedelta(hours=9)
