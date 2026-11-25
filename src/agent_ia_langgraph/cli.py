"""Terminal chat with the agent."""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from agent_ia_langgraph.agent import DEFAULT_SYSTEM_PROMPT, Agent

EXIT_COMMANDS = {"/quit", "/exit"}
DEFAULT_MODEL = "gemini-3.5-flash-lite"


class ChatCli:
    """Terminal chat loop around an agent.

    It reads the user's messages, sends them to the agent and prints the
    replies. The agent is created and configured by the caller.
    """

    def __init__(self, agent: Agent) -> None:
        """Create a chat around an agent.

        Args:
            agent: The agent that answers, already configured.
        """
        self._agent = agent

    def run(self) -> None:
        """Chat until the user leaves.

        The user leaves with /quit, /exit, Ctrl+D, or Ctrl+C at the prompt.
        Ctrl+C during a request cancels only that request. A failed request
        prints the error and the chat goes on; the conversation is left as it
        was before the request.
        """
        print("Welcome to your agent with LangGraph!")
        print("Available Commands:")
        print(" - /quit or /exit to end the session")
        print("Type your message and press Enter to send it.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break

            if not user_input:
                continue

            if user_input.lower() in EXIT_COMMANDS:
                break

            try:
                reply = self._agent.chat(user_input)
            except KeyboardInterrupt:
                print("[request cancelled]")
                continue
            except Exception as error:
                print(f"[error] {error}")
                continue

            print(f"AI: {reply}\n")

        print("Goodbye!")


def main() -> None:
    """Start the terminal chat with Gemini.

    The configuration comes from the environment, which the .env file can
    fill: GEMINI_API_KEY (required) and GEMINI_MODEL (optional). A missing
    key stops the program with a config error; an invalid key only fails
    at the first request.
    """
    load_dotenv()
    try:
        model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    except ValueError as error:
        raise SystemExit(f"[config error] {error}") from error

    agent = Agent(model, system_prompt=DEFAULT_SYSTEM_PROMPT)
    ChatCli(agent).run()
