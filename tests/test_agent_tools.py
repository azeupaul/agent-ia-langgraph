import pytest
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from agent_ia_langgraph.agent import Agent
from tests.fakes import RecordingChatModel


@tool
def get_weather(city: str) -> str:
    """Get the current weather in a city."""
    if city == "Atlantis":
        raise ValueError(f"Unknown city: {city}")
    return f"Sunny in {city}"


def ask_for_weather(city: str, call_id: str) -> AIMessage:
    """A model reply that asks for get_weather, as Gemini would."""
    call = ToolCall(name="get_weather", args={"city": city}, id=call_id)
    return AIMessage(content="", tool_calls=[call])


def test_a_requested_tool_runs_and_its_result_goes_back_to_the_model() -> None:
    model = RecordingChatModel(
        messages=iter([ask_for_weather("Douala", "call_1"), "Sunny in Douala."])
    )
    agent = Agent(model, tools=[get_weather])

    reply = agent.chat("What is the weather in Douala?")

    assert reply == "Sunny in Douala."
    second_call = model.received[1]
    assert [(message.type, message.text) for message in second_call] == [
        ("human", "What is the weather in Douala?"),
        ("ai", ""),
        ("tool", "Sunny in Douala"),
    ]


def test_the_model_is_told_which_tools_it_can_use() -> None:
    model = RecordingChatModel(messages=iter([]))

    Agent(model, tools=[get_weather])

    assert model.bound_tools == ["get_weather"]


def test_a_tool_error_goes_back_to_the_model_so_it_can_recover() -> None:
    model = RecordingChatModel(
        messages=iter([ask_for_weather("Atlantis", "call_1"), "I don't know it."])
    )
    agent = Agent(model, tools=[get_weather])

    reply = agent.chat("What is the weather in Atlantis?")

    assert reply == "I don't know it."
    tool_result = model.received[1][-1]
    assert tool_result.type == "tool"
    assert "Unknown city: Atlantis" in tool_result.text


def test_a_turn_failing_after_a_tool_call_leaves_none_of_its_messages() -> None:
    model = RecordingChatModel(
        messages=iter(["Hello.", ask_for_weather("Douala", "call_1")]),
        failure=RuntimeError("API unavailable"),
        fail_from_call=3,
    )
    agent = Agent(model, tools=[get_weather])
    agent.chat("Hi.")

    # Call 2 asks for the tool, the tool runs, then call 3 fails.
    with pytest.raises(RuntimeError, match="API unavailable"):
        agent.chat("What is the weather in Douala?")

    assert [(message.type, message.text) for message in agent.history] == [
        ("human", "Hi."),
        ("ai", "Hello."),
    ]


def always_asks_for_weather() -> RecordingChatModel:
    """A model that never answers: every reply asks for the tool again."""
    replies = [ask_for_weather("Douala", f"call_{i}") for i in range(10)]
    return RecordingChatModel(messages=iter(replies))


def test_a_turn_stops_after_max_iterations_and_leaves_no_messages() -> None:
    model = always_asks_for_weather()
    agent = Agent(model, tools=[get_weather], max_iterations=2)

    with pytest.raises(GraphRecursionError):
        agent.chat("What is the weather in Douala?")

    assert len(model.received) == 2
    assert agent.history == []


def test_a_turn_may_use_all_of_its_max_iterations() -> None:
    model = RecordingChatModel(
        messages=iter([ask_for_weather("Douala", "call_1"), "Sunny in Douala."])
    )
    agent = Agent(model, tools=[get_weather], max_iterations=2)

    assert agent.chat("What is the weather in Douala?") == "Sunny in Douala."
