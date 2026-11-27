# agent-ia-langgraph

A learning project to rebuild an AI agent with [LangGraph](https://github.com/langchain-ai/langgraph), and to compare it with the same agent built **without a framework**: [simple-agent-ia](https://github.com/azeupaul/simple-agent-ia).

The goal is not only to use LangGraph, but to see, for each mechanism, what the framework replaces, what it hides and what it adds. The agent stays the same: a terminal chat with Gemini, the same tools (`get_current_time`, `calculate`, `write_note` with human confirmation) and, where possible, the same commands.

## Status

Step 2 is done: a terminal chat with Gemini that remembers the conversation and uses two tools, `get_current_time` and `calculate`, built as a graph that loops between the model and the tools. `write_note`, which needs the user's confirmation, comes at step 3.

## Without a framework → LangGraph

One row per mechanism, completed at each step.

| Mechanism | Without a framework | LangGraph |
|---|---|---|
| Calling Gemini | The [Google Gen AI SDK](https://github.com/googleapis/python-genai) (`google-genai`), called directly | [`langchain-google-genai`](https://github.com/langchain-ai/langchain-google), which wraps the same `google-genai` SDK |
| Dependencies | 3 direct dependencies, 28 packages installed | 4 direct dependencies, 49 packages installed |
| Conversation memory | `Agent._history`, a list the agent appends to and sends on every call | The graph state (`MessagesState`), saved by a checkpointer (`InMemorySaver`) under a `thread_id`. `chat` sends only the new message: the graph reloads the rest |
| Adding a message | `self._history.append(...)` | The node returns `{"messages": [reply]}`, and the `add_messages` reducer appends it to the state |
| System prompt | `system_instruction` in the request config | A `SystemMessage` put first in the messages, in the node, and never stored. `langchain-google-genai` turns it into `system_instruction` |
| A failed turn | Nothing to undo: a turn is committed only once it completes | Each step is checkpointed as it ends, so a failed turn leaves its messages behind: the user message, and any tool request and result before the failure. `chat` removes them with `RemoveMessage`, through the same reducer |
| Reading the history | A list of `Content` with parts. Displaying it joined the streamed text pieces and skipped the parts that only carried a thought signature | `graph.get_state(config).values["messages"]`. `message.text` joins the text blocks, so the display needs no special handling |
| Reset | `self._history = []`: the conversation is deleted | A new `thread_id`: the previous conversation stays in the checkpointer, unread |
| Retries | None: an API error fails the turn at once | `ChatGoogleGenerativeAI` makes up to 6 attempts by default (`max_retries=6`, where 1 means no retry) before raising |
| Test double | A fake Gemini client that replays scripted streams | `RecordingChatModel`, a `GenericFakeChatModel` that also records the messages of each call, and accepts tools |
| Declaring a tool | A home-made `@tool` decorator, with its own docstring parser and JSON schema builder | `@tool(parse_docstring=True)` from `langchain-core`: the text before `Args:` describes the tool, each `Args:` line describes a parameter. Without `parse_docstring`, the parameters get no description at all, and a test checks that none is missing |
| Telling the model about the tools | `ToolRegistry.declarations`, a list of `FunctionDeclaration` in the request config | `model.bind_tools(tools)`, which returns a new runnable that sends them with every call |
| Running a requested tool | `Agent._run_tool`, which finds the tool in the `ToolRegistry` and calls it | `ToolNode`, a ready-made node that runs the calls of the last `AIMessage` and returns one `ToolMessage` per call |
| Several calls in one reply | A `for` loop: one call after the other | `ToolNode` runs them in parallel, in a single graph step |
| The tool loop | A `for` loop over `max_iterations`, which stops when the reply has no function call | Two edges: `tools_condition` goes to `tools` if the reply asks for tools, to `END` otherwise, and `tools` always goes back to `call_model` |
| A failing tool | `_run_tool` never raises: the error goes back to the model as `{"error": ...}` | `ToolNode(..., handle_tool_errors=True)`: the error goes back as a `ToolMessage` with `status="error"`. **By default, `ToolNode` only reports invalid arguments, and raises any other error**, which would end the turn |
| Too many model calls | `max_iterations` counts the model calls, then a `RuntimeError` | `recursion_limit` counts graph steps, not model calls: each model call takes two steps (model, then tools), so `recursion_limit = 2 * max_iterations`. Past it, LangGraph raises `GraphRecursionError`, and `chat` cancels the turn. The default limit is high enough to let a looping model run for a long while |
| Thought signatures | The model's parts are kept exactly as they are, so that Gemini 3 finds its signatures on the next call | Nothing to do: `langchain-google-genai` keeps them in the `AIMessage` and sends them back |

The framework does not replace the call to Gemini: it wraps it. `langgraph` builds on `langchain-core`, which provides the shared building blocks (messages, tools, chat models), while ready-made components such as `ToolNode` live in a separate package, `langgraph-prebuilt`.

## How it works

```
graph.invoke({"messages": [HumanMessage("What time is it in Tokyo?")]}, config)
   │  the checkpointer reloads the thread, add_messages appends the question
   ▼
START ──► call_model ──tools_condition──► END        (the reply asks for no tool)
             ▲              │
             │              ▼                        (the reply asks for tools)
             └────────── tools
```

1. `call_model` sends `[SystemMessage, *messages]` to the model, which replies with an `AIMessage` whose `tool_calls` ask for `get_current_time(timezone="Asia/Tokyo")`.
2. `tools_condition` sees the `tool_calls` and goes to `tools`. `ToolNode` runs the tool and returns a `ToolMessage` with the result, linked to the call by its id.
3. Back to `call_model`: the model now has the result, and replies with text. `tools_condition` goes to `END`.

Each step is saved by the checkpointer, and `add_messages` appends each new message to the state.

| Piece | Role |
|---|---|
| State | `MessagesState`: a dictionary with a `messages` list |
| Nodes | `call_model`, a function that receives the state and returns an **update**, not the whole state, and `tools`, a ready-made `ToolNode` |
| Reducer | `add_messages`: merges the update into the state. It appends new messages and removes those named by a `RemoveMessage` |
| Edges | `START → call_model`, a conditional edge `call_model → tools or END` (`tools_condition`), and `tools → call_model` |
| Checkpointer | `InMemorySaver`: saves the state after each step, per `thread_id` |

Without tools, the agent builds a simpler graph, `START → call_model → END`, and the model is not told about any tool.

| File | Role |
|---|---|
| `agent.py` | `Agent`: builds the graph, and exposes `chat`, `history` and `reset` |
| `tools.py` | The tools, and `TOOLS`, the tuple the chat gives the agent |
| `cli.py` | `ChatCli`, the terminal chat loop, and `main`, which assembles the model, the tools, the agent and the chat |

### Tools

| Tool | What it does |
|---|---|
| `get_current_time(timezone)` | The current date and time in an IANA timezone, in ISO 8601 |
| `calculate(expression)` | Evaluates an arithmetic expression: numbers, `+ - * / // % **` and parentheses |

Each tool is a function with `@tool(parse_docstring=True)`. Its docstring is written **for the model**: it says what the tool does, when to use it, and what each parameter expects.

`calculate` never runs `eval`. It parses the expression into a syntax tree with `ast`, and evaluates only numbers and arithmetic operators. It also refuses results of more than 1000 digits, checked **before** computing a power (`9 ** 9 ** 9` would take minutes), and results that are infinite or complex. Its errors go back to the model, which can fix the expression.

### Gemini 3 replies

With Gemini 3, `AIMessage.content` is a list of content blocks such as `{"type": "text", "text": "..."}`, not a string. The agent reads the reply with `message.text`, which joins the text blocks and ignores the others.

### A failed turn

If a model call fails, the model calls too many tools in a row (`GraphRecursionError`), or the user presses `Ctrl+C` during a request, `chat` raises the error and leaves the conversation as it was before the turn:

- Before the turn, `chat` notes the ids of the messages already saved. After a failure, it removes every message whose id is not in that list: the user message, and any tool request and result saved before the failure. Removing only the last message would leave the others behind.
- Only messages that were actually saved are removed: removing an unknown id raises an error, which would hide the original one.
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
- [x] 2. Tools: LangChain tools, a tool node and a conditional edge, recursion limit
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
- `/history` shows a tool request as an empty `[ai]` line: it does not say which tool was asked for. Step 4 (tracing) will show it.
## License

[MIT](LICENSE)
