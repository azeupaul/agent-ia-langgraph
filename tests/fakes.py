"""Test doubles for the chat model, the only external boundary of the agent."""

from collections.abc import Callable, Sequence
from typing import Any, override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field


class RecordingChatModel(GenericFakeChatModel):
    """Fake chat model that replays scripted replies and records what it receives.

    `GenericFakeChatModel` only replays replies. Recording the messages of each
    call lets a test check what the agent actually sends to the model, such as
    the whole history on the second turn.

    A scripted reply can be an `AIMessage` with `tool_calls`: the fake model
    asks for a tool the same way Gemini does.
    """

    received: list[list[BaseMessage]] = Field(default_factory=list)
    """The messages of each call, in call order."""

    failure: BaseException | None = None
    """When set, every call raises it instead of replying: an API error, Ctrl+C."""

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        if self.failure is not None:
            raise self.failure
        return super()._generate(messages, stop, run_manager, **kwargs)

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        # The replies are scripted, so the tools change nothing: the fake model
        # only has to accept them, which GenericFakeChatModel does not.
        return self
