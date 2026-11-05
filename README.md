# Agentic AI Platform

A modular, observable task-execution platform built around LangGraph orchestration, validated tool routing, bounded execution, and persistent traces.

> **Status:** MVP foundation in active development. The deterministic planner works without credentials. OpenAI native function calling and multi-turn model-directed replanning are available as an optional planner.

## Why this project?

This is not a chat UI with tools bolted on. It is an agent runtime with explicit boundaries:
- typed task plans and validated tool inputs/outputs;
- a registry that exposes only approved tools;
- native OpenAI function schemas generated from Pydantic models;
- bounded multi-turn planning, tool steps, retries, and execution time;
- structured traces, persistent task history, and session-scoped conversation memory;
- deterministic tests that do not require network access or an LLM.

## Architecture

```text
Client -> FastAPI -> LangGraph Runtime
                       |
                 Planner node
               /             \
       Rules planner     OpenAI tool calling
               \             /
                Validated Plan
                       |
               Execute-tool node
                       |
             Reflection / retry
                       |
              SQLite trace store
                       |
                Task response
```

## Quick start

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
copy .env.example .env  # Windows
# cp .env.example .env  # macOS/Linux
uvicorn agentic_platform.api:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Example

```bash
curl -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"task":"Calculate 19 * 23"}'
```

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `AGENT_RUNTIME_BACKEND` | `langgraph` or `legacy` | `langgraph` |
| `AGENT_PLANNER` | `rules` or `openai` | `rules` |
| `AGENT_OPENAI_STRATEGY` | `tool_calling` or `json_plan` | `tool_calling` |
| `OPENAI_API_KEY` | Enables the optional OpenAI planner | unset |
| `OPENAI_MODEL` | Model for planning | `gpt-4.1-mini` |
| `AGENT_MAX_STEPS` | Maximum tool calls per task | `5` |
| `AGENT_MAX_RETRIES` | Maximum retries after tool failures | `1` |
| `AGENT_TIMEOUT_SECONDS` | Total execution budget | `20` |
| `AGENT_DATABASE_PATH` | SQLite database file | `./data/agentic_platform.db` |\n| `AGENT_MEMORY_MAX_MESSAGES` | Maximum stored messages per session | `20` |

Set `AGENT_PLANNER=openai` and provide `OPENAI_API_KEY` to enable model-directed tool selection. Native function calling is the default OpenAI strategy. Set `AGENT_OPENAI_STRATEGY=json_plan` to use the legacy JSON-plan adapter. If no API key is configured, the runtime safely falls back to the deterministic rules planner.

## Persistence and API

Task responses, ordered trace events, and session-scoped conversation memory are persisted in SQLite. Send the same `session_id` with successive task requests to restore prior user/assistant messages as context. If omitted, a new isolated session is created. Memory is bounded by `AGENT_MEMORY_MAX_MESSAGES` and can be inspected through the session memory endpoint. The database is created automatically at the configured path; the local `data/` directory is ignored by Git.

- `POST /tasks` executes a task and stores its report.
- `GET /tasks/{task_id}` retrieves a persisted task report.
- `GET /tasks/{task_id}/trace` retrieves its ordered execution trace.\n- `GET /sessions/{session_id}/memory` retrieves the bounded conversation history.

## Safety model

- Tools are explicitly registered; arbitrary Python or shell execution is not exposed.
- OpenAI receives only function schemas exported by the allow-listed registry.
- Tool arguments are validated with Pydantic before execution.
- Each task has a bounded number of steps, retries, and wall-clock time.
- Tool errors are recorded as observations rather than silently discarded.
- No secrets are returned in task results or traces.

## Development

```bash
pytest -q
ruff check .
```

## Roadmap

- [x] Typed task, plan, tool, and trace models
- [x] Safe local tool registry
- [x] Deterministic planner, bounded runtime, and retry policy
- [x] Structured reflection events for tool failures
- [x] FastAPI task endpoint and persisted task retrieval
- [x] LangGraph execution backend
- [x] SQLite task and trace storage\n- [x] Session-scoped persistent conversation memory
- [x] OpenAI native function-calling planner
- [x] Multi-turn tool observations and model-directed replanning
- [ ] Authentication, rate limiting, and Docker deployment

## License

MIT
