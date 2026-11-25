"""Test doubles for the chat model, the only external boundary of the agent."""

from typing import Any, override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import Field


class RecordingChatModel(GenericFakeChatModel):
    """Fake chat model that replays scripted replies and records what it receives.

    `GenericFakeChatModel` only replays replies. Recording the messages of each
    call lets a test check what the agent actually sends to the model, such as
    the whole history on the second turn.
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
