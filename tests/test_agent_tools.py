from langchain_core.messages import AIMessage, ToolCall
from langchain_core.tools import tool

from agent_ia_langgraph.agent import Agent
from tests.fakes import RecordingChatModel


@tool
def get_weather(city: str) -> str:
    """Get the current weather in a city."""
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
