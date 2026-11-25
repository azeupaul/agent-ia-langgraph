from langchain_core.messages import AIMessage

from agent_ia_langgraph.agent import Agent
from tests.fakes import RecordingChatModel


def test_chat_returns_the_text_of_the_reply() -> None:
    # Gemini 3 replies with a list of content blocks, not a plain string.
    reply = AIMessage(content=[{"type": "text", "text": "Hello! How can I help?"}])
    agent = Agent(RecordingChatModel(messages=iter([reply])))

    assert agent.chat("Hi") == "Hello! How can I help?"
