# Agentic AI Platform

A modular, observable task-execution platform built around explicit planning, tool routing, bounded execution, and evidence-backed results.

> **Status:** MVP foundation in active development. The deterministic planner and local tools run without credentials; an optional OpenAI planner can be enabled with an API key.

## Why this project?

This is not a chat UI with tools bolted on. It is an agent runtime with explicit boundaries:
- structured task plans and typed tool inputs/outputs;
- a registry that exposes only approved tools;
- bounded steps, retries, and execution time;
- an execution trace for every decision and tool call;
- deterministic tests that do not require network access or an LLM.

## Architecture

```text
Client -> FastAPI -> Agent Runtime
                       |
                 Planner interface
                       |
                 Validated plan
                       |
                Tool registry
              /       |       \
          Calculator Search  Date/time
              \       |       /
                Observations
                       |
              Reflection / retry
                       |
             Evidence-backed result
                       |
                 Trace report
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
  -d '{"task":"Calculate 19 * 23 and explain the result"}'
```

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `AGENT_PLANNER` | `rules` or `openai` | `rules` |
| `OPENAI_API_KEY` | Enables the optional OpenAI planner | unset |
| `OPENAI_MODEL` | Model for planning | `gpt-4.1-mini` |
| `AGENT_MAX_STEPS` | Maximum tool calls per task | `5` |
| `AGENT_MAX_RETRIES` | Maximum retries after tool failures | `1` |
| `AGENT_TIMEOUT_SECONDS` | Total execution budget | `20` |

The default rules planner is intentionally deterministic. The OpenAI planner is an optional adapter; tool execution remains constrained by the same registry and schemas.

## Safety model

- Tools are explicitly registered; arbitrary Python or shell execution is not exposed.
- Tool arguments are validated with Pydantic.
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
- [x] Deterministic planner and bounded runtime
- [x] FastAPI task endpoint
- [x] Unit and API tests
- [ ] LangGraph execution backend
- [ ] OpenAI tool-calling planner
- [ ] Persistent task/trace storage
- [ ] Authentication, rate limiting, and Docker deployment

## License

MIT
