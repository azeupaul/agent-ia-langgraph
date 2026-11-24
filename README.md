# agent-ia-langgraph

A learning project to rebuild an AI agent with [LangGraph](https://github.com/langchain-ai/langgraph), and to compare it with the same agent built **without a framework**: [simple-agent-ia](https://github.com/azeupaul/simple-agent-ia).

The goal is not only to use LangGraph, but to see, for each mechanism, what the framework replaces, what it hides and what it adds. The agent stays the same: a terminal chat with Gemini, the same tools (`get_current_time`, `calculate`, `write_note` with human confirmation) and, where possible, the same commands.

## Status

Step 0 is done: the project, its dependencies and the quality checks are set up. The agent itself starts at step 1.

## Without a framework → LangGraph

One row per mechanism, completed at each step.

| Mechanism | Without a framework | LangGraph |
|---|---|---|
| Calling Gemini | The [Google Gen AI SDK](https://github.com/googleapis/python-genai) (`google-genai`), called directly | [`langchain-google-genai`](https://github.com/langchain-ai/langchain-google), which wraps the same `google-genai` SDK |
| Dependencies | 3 direct dependencies, 28 packages installed | 4 direct dependencies, 49 packages installed |

The framework does not replace the call to Gemini: it wraps it. `langgraph` builds on `langchain-core`, which provides the shared building blocks (messages, tools, chat models), while ready-made components such as `ToolNode` live in a separate package, `langgraph-prebuilt`.

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

`langchain-google-genai` reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) itself, so the `.env` file of the framework-free project works as is.

## Development

```bash
uv run pytest                     # run the tests
uv run pre-commit install         # once: run the checks before each commit
uv run pre-commit run --all-files # run them manually
```

The checks are `ruff format`, `ruff check`, `mypy` in strict mode and `pytest`. GitHub Actions runs the same hooks on every push and pull request.

The code is a package in `src/agent_ia_langgraph/`, installed in editable mode by `uv sync`, so the tests import it like any other package.

## Roadmap

Each step maps a mechanism written by hand in the framework-free project to its LangGraph equivalent.

- [x] 0. Setup: uv, `langgraph`, `langchain-google-genai`, quality checks
- [ ] 1. Minimal graph: message state, a model node, an in-memory checkpointer and a thread
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

## License

[MIT](LICENSE)
