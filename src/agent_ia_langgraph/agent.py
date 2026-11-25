"""Conversational agent built as a LangGraph graph."""

from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph


class Agent:
    """Chat agent that remembers the conversation.

    Each turn runs a graph with a single node that calls the model. A
    checkpointer saves the messages under a thread id after each run, so the
    next turn sends the whole conversation to the model again.
    """

    def __init__(
        self, model: BaseChatModel, *, system_prompt: str | None = None
    ) -> None:
        """Create an agent with an empty conversation.

        Args:
            model: The chat model that answers, such as `ChatGoogleGenerativeAI`.
            system_prompt: Instructions sent before the conversation on every model
                call. They are not stored in the conversation. None sends none.
        """
        self._model = model
        self._system_prompt = system_prompt
        self.reset()

        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("call_model", self._call_model)
        graph_builder.add_edge(START, "call_model")
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

        The message and the reply are added to the conversation.

        Args:
            user_prompt: What the user typed.

        Returns:
            The text of the reply, without its other content blocks.
        """
        result = self._graph.invoke(
            {"messages": [HumanMessage(user_prompt)]}, self._config
        )

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
