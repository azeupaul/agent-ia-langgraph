from langchain_core.messages import AIMessage

from agent_ia_langgraph.agent import Agent
from tests.fakes import RecordingChatModel


def test_chat_returns_the_text_of_the_reply() -> None:
    # Gemini 3 replies with a list of content blocks, not a plain string.
    reply = AIMessage(content=[{"type": "text", "text": "Hello! How can I help?"}])
    agent = Agent(RecordingChatModel(messages=iter([reply])))

    assert agent.chat("Hi") == "Hello! How can I help?"


def test_the_model_receives_the_whole_conversation_on_the_next_turn() -> None:
    model = RecordingChatModel(messages=iter(["Nice to meet you, Ada.", "Ada."]))
    agent = Agent(model)

    agent.chat("My name is Ada.")
    agent.chat("What is my name?")

    second_call = model.received[1]
    assert [(message.type, message.text) for message in second_call] == [
        ("human", "My name is Ada."),
        ("ai", "Nice to meet you, Ada."),
        ("human", "What is my name?"),
    ]


def test_the_system_prompt_comes_first_and_only_once_on_every_turn() -> None:
    model = RecordingChatModel(messages=iter(["Hello.", "Ada."]))
    agent = Agent(model, system_prompt="Answer briefly.")

    agent.chat("My name is Ada.")
    agent.chat("What is my name?")

    second_call = model.received[1]
    assert [(message.type, message.text) for message in second_call] == [
        ("system", "Answer briefly."),
        ("human", "My name is Ada."),
        ("ai", "Hello."),
        ("human", "What is my name?"),
    ]
