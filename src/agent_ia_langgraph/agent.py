"""Conversational agent built as a LangGraph graph."""

from collections.abc import Sequence
from uuid import uuid4

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant that helps users solve problems.\n"
    "Always reply in the language of the user's latest message."
)


class Agent:
    """Chat agent that remembers the conversation and can use tools.

    Each turn runs a graph: the model is called and, while it asks for
    tools, they run and their results go back to it. A checkpointer saves
    the messages under a thread id after each run, so the next turn sends
    the whole conversation to the model again.
    """

    def __init__(
        self,
        model: BaseChatModel,
        *,
        tools: Sequence[BaseTool] = (),
        system_prompt: str | None = None,
    ) -> None:
        """Create an agent with an empty conversation.

        Args:
            model: The chat model that answers, such as `ChatGoogleGenerativeAI`.
            tools: Tools the model may ask to run, such as `calculate`. Each
                requested tool runs and its result goes back to the model, until it
                answers with text. With no tools, the model only answers.
            system_prompt: Instructions sent before the conversation on every model
                call. They are not stored in the conversation. None sends none.
        """
        self._model: Runnable[LanguageModelInput, BaseMessage] = (
            model.bind_tools(tools) if tools else model
        )
        self._system_prompt = system_prompt
        self.reset()

        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("call_model", self._call_model)
        graph_builder.add_edge(START, "call_model")
        if tools:
            graph_builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
            graph_builder.add_conditional_edges("call_model", tools_condition)
            graph_builder.add_edge("tools", "call_model")
        else:
            graph_builder.add_edge("call_model", END)

        self._graph = graph_builder.compile(checkpointer=InMemorySaver())

    def _call_model(self, state: MessagesState) -> dict[str, list[BaseMessage]]:
        """Graph node: send the conversation to the model.

        Args:
            state: The current state of the graph, with the conversation so far.

        Returns:
            The update to merge into the state: the model's reply, which the
            `add_messages` reducer appends to the conversation.
        """
        messages = state["messages"]
        if self._system_prompt is not None:
            messages = [SystemMessage(content=self._system_prompt), *messages]

        response = self._model.invoke(messages)

        return {"messages": [response]}

    def chat(self, user_prompt: str) -> str:
        """Send a user message and return the model's reply.

        The message and the reply are added to the conversation,
        only if the turn succeeds.

        Args:
            user_prompt: What the user typed.

        Returns:
            The text of the reply, without its other content blocks.

        Raises:
            BaseException: Any error from the model, or an interruption such as
                Ctrl+C, is raised again once the turn is cancelled: the
                conversation is left as it was.
        """
        message_id = str(uuid4())
        try:
            result = self._graph.invoke(
                {"messages": [HumanMessage(user_prompt, id=message_id)]}, self._config
            )
        except BaseException:
            self._cancel(message_id)
            raise

        reply: BaseMessage = result["messages"][-1]

        return reply.text

    @property
    def history(self) -> list[BaseMessage]:
        """The conversation so far, oldest message first.

        The system prompt is not part of it: it is sent to the model on every
        call but never stored.
        """
        snapshot = self._graph.get_state(self._config)

        return list(snapshot.values.get("messages", []))

    def reset(self) -> None:
        """Start a new conversation.

        The next turns use a new thread id. The previous conversation is not
        deleted: it stays in the checkpointer, but the agent no longer reads it.
        """
        self._config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}

    def _cancel(self, message_id: str) -> None:
        """Remove the user message of a failed turn from the conversation.

        LangGraph saves the input in a checkpoint before running the node, so a
        failed turn would leave the user message behind. The removal goes
        through the `add_messages` reducer. Nothing is removed if the message
        was never saved, since removing an unknown id raises an error.
        """
        if any(message_id == message.id for message in self.history):
            self._graph.update_state(
                self._config, {"messages": [RemoveMessage(id=message_id)]}
            )
