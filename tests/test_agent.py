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


def test_history_lists_the_conversation_without_the_system_prompt() -> None:
    model = RecordingChatModel(messages=iter(["Hello.", "Ada."]))
    agent = Agent(model, system_prompt="Answer briefly.")

    agent.chat("My name is Ada.")
    agent.chat("What is my name?")

    assert [(message.type, message.text) for message in agent.history] == [
        ("human", "My name is Ada."),
        ("ai", "Hello."),
        ("human", "What is my name?"),
        ("ai", "Ada."),
    ]


def test_reset_starts_a_new_conversation() -> None:
    model = RecordingChatModel(messages=iter(["Hello.", "I don't know."]))
    agent = Agent(model)
    agent.chat("My name is Ada.")

    agent.reset()

    assert agent.history == []
    agent.chat("What is my name?")
    assert [(message.type, message.text) for message in model.received[-1]] == [
        ("human", "What is my name?"),
    ]
