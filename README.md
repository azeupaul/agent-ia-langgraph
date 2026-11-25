# agent-ia-langgraph

A learning project to rebuild an AI agent with [LangGraph](https://github.com/langchain-ai/langgraph), and to compare it with the same agent built **without a framework**: [simple-agent-ia](https://github.com/azeupaul/simple-agent-ia).

The goal is not only to use LangGraph, but to see, for each mechanism, what the framework replaces, what it hides and what it adds. The agent stays the same: a terminal chat with Gemini, the same tools (`get_current_time`, `calculate`, `write_note` with human confirmation) and, where possible, the same commands.

## Status

Step 1 is done: a terminal chat with Gemini that remembers the conversation, built as a graph with a single node. The agent has no tools yet: they come at step 2.

## Without a framework → LangGraph

One row per mechanism, completed at each step.

| Mechanism | Without a framework | LangGraph |
|---|---|---|
| Calling Gemini | The [Google Gen AI SDK](https://github.com/googleapis/python-genai) (`google-genai`), called directly | [`langchain-google-genai`](https://github.com/langchain-ai/langchain-google), which wraps the same `google-genai` SDK |
| Dependencies | 3 direct dependencies, 28 packages installed | 4 direct dependencies, 49 packages installed |
| Conversation memory | `Agent._history`, a list the agent appends to and sends on every call | The graph state (`MessagesState`), saved by a checkpointer (`InMemorySaver`) under a `thread_id`. `chat` sends only the new message: the graph reloads the rest |
| Adding a message | `self._history.append(...)` | The node returns `{"messages": [reply]}`, and the `add_messages` reducer appends it to the state |
| System prompt | `system_instruction` in the request config | A `SystemMessage` put first in the messages, in the node, and never stored. `langchain-google-genai` turns it into `system_instruction` |
| A failed turn | Nothing to undo: a turn is committed only once it completes | The input is checkpointed **before** the node runs, so a failed turn leaves the user message behind. `chat` removes it with a `RemoveMessage`, through the same reducer |
| Reading the history | A list of `Content` with parts. Displaying it joined the streamed text pieces and skipped the parts that only carried a thought signature | `graph.get_state(config).values["messages"]`. `message.text` joins the text blocks, so the display needs no special handling |
| Reset | `self._history = []`: the conversation is deleted | A new `thread_id`: the previous conversation stays in the checkpointer, unread |
| Retries | None: an API error fails the turn at once | `ChatGoogleGenerativeAI` makes up to 6 attempts by default (`max_retries=6`, where 1 means no retry) before raising |
| Test double | A fake Gemini client that replays scripted streams | `RecordingChatModel`, a `GenericFakeChatModel` that also records the messages of each call |

The framework does not replace the call to Gemini: it wraps it. `langgraph` builds on `langchain-core`, which provides the shared building blocks (messages, tools, chat models), while ready-made components such as `ToolNode` live in a separate package, `langgraph-prebuilt`.

## How it works

```
graph.invoke({"messages": [HumanMessage("Hi")]}, {"configurable": {"thread_id": ...}})
   │  the checkpointer reloads the thread, add_messages appends "Hi"
   ▼
START ──► call_model ──► END
             │ sends [SystemMessage, *messages] to the model
             │ returns {"messages": [AIMessage("Hello!")]}
             ▼
          add_messages appends the reply, the checkpointer saves the state
```

| Piece | Role |
|---|---|
| State | `MessagesState`: a dictionary with a `messages` list |
| Node | `call_model`, a function that receives the state and returns an **update**, not the whole state |
| Reducer | `add_messages`: merges the update into the state. It appends new messages and removes those named by a `RemoveMessage` |
| Edges | `START → call_model → END`: a single model call per turn, for now |
| Checkpointer | `InMemorySaver`: saves the state after each step, per `thread_id` |

| File | Role |
|---|---|
| `agent.py` | `Agent`: builds the graph, and exposes `chat`, `history` and `reset` |
| `cli.py` | `ChatCli`, the terminal chat loop, and `main`, which assembles the model, the agent and the chat |

### Gemini 3 replies

With Gemini 3, `AIMessage.content` is a list of content blocks such as `{"type": "text", "text": "..."}`, not a string. The agent reads the reply with `message.text`, which joins the text blocks and ignores the others.

### A failed turn

If the model call fails, or the user presses `Ctrl+C` during a request, `chat` raises the error and leaves the conversation as it was before the turn:

- The user message gets its id **before** the graph runs, so the agent knows exactly which message to remove. Removing the last message instead could remove a valid reply, if the error came before the input was saved.
- The message is removed only if it was saved: removing an unknown id raises an error, which would hide the original one.
- `except BaseException` covers `KeyboardInterrupt`, which is not an `Exception`.

LangGraph keeps the input on purpose: it can retry the unfinished node with `graph.invoke(None, config)`. A chat would rather cancel the turn and let the user type again; long tasks, such as plans, may prefer to resume.

### Commands

| Command | Action |
|---|---|
| `/history` | Show the conversation, one line per message |
| `/reset` | Start a new conversation |
| `/quit`, `/exit`, `Ctrl+D` | Leave |
| `Ctrl+C` | Cancel the current request, or leave at the prompt |

Each command is a method of `ChatCli`, found in a dictionary: adding a command adds a method and a line, without touching the chat loop.

## Getting started

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/) and a [Gemini API key](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/azeupaul/agent-ia-langgraph.git
cd agent-ia-langgraph
uv sync
cp .env.example .env   # then set your API key in .env
uv run agent-ia-langgraph
```

| Variable | Required | Default |
|---|---|---|
| `GEMINI_API_KEY` | yes | |
| `GEMINI_MODEL` | no | `gemini-3.5-flash-lite` |

`langchain-google-genai` reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) itself, so the `.env` file of the framework-free project works as is. Without a key, the chat refuses to start. An invalid key only fails at the first request.

## Development

```bash
uv run pytest                     # run the tests
uv run pre-commit install         # once: run the checks before each commit
uv run pre-commit run --all-files # run them manually
```

The checks are `ruff format`, `ruff check`, `mypy` in strict mode and `pytest`. GitHub Actions runs the same hooks on every push and pull request. `pre-commit run --all-files` only checks the files git tracks: add new files with `git add` first.

The code is a package in `src/agent_ia_langgraph/`, installed in editable mode by `uv sync`, so the tests import it like any other package.

The tests never call Gemini. They go through the real graph and checkpointer, and replace only the chat model, the one external boundary, with `RecordingChatModel` (`tests/fakes.py`). It replays scripted replies, records what the agent sends, and can fail like an API error or a `Ctrl+C`.

## Roadmap

Each step maps a mechanism written by hand in the framework-free project to its LangGraph equivalent.

- [x] 0. Setup: uv, `langgraph`, `langchain-google-genai`, quality checks
- [x] 1. Minimal graph: message state, a model node, an in-memory checkpointer and a thread
- [ ] 2. Tools: LangChain tools, a tool node and a conditional edge, recursion limit
- [ ] 3. Human in the loop: confirm `write_note` by interrupting the graph and resuming it
- [ ] 4. Streaming and tracing
- [ ] 5. Context management: trimming and summarizing messages
- [ ] 6. Structured output with Pydantic
- [ ] 7. Planning with replanning (plan-and-execute)
- [ ] 8. Tests: a fake chat model compared with the fake Gemini client
- [ ] Beyond the framework-free project
  - [ ] Evaluation: a question set, LLM as a judge, before/after comparisons
  - [ ] RAG: long-term memory over the notes written by `write_note`
  - [ ] Multi-agents: a supervisor delegating to specialized agents
  - [ ] MCP: plugging in standardized external tools

Some known limitations of the framework-free project were left on purpose, to be solved here the framework's way: the model requesting a declined call again (step 3), a summary on every turn when the threshold is too low (step 5), and a plan that is never revised (step 7).

## Known limitations

- The conversation lives in memory: it is lost when the chat ends. A persistent checkpointer, such as SQLite, would keep it.
- `/reset` does not free memory: every previous conversation stays in the checkpointer until the chat ends.
- With up to 6 attempts by default, a failing API can take a while before the error shows.
- `langchain-google-genai` only turns the **first** `SystemMessage` into `system_instruction`: a system message placed later, with none before it, is silently dropped. The agent always puts it first, and a test checks it.
- The removal of a failed turn's message checks that the message was saved first. No test covers the case where it was not, since it needs a checkpointer that fails.

## License

[MIT](LICENSE)
