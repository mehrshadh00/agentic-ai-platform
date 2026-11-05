# Agentic AI Platform

A modular, observable task-execution platform built around LangGraph orchestration, validated tool routing, bounded execution, and persistent memory.

> **Status:** MVP foundation in active development. The deterministic planner works without credentials. OpenAI native function calling and multi-turn model-directed replanning are available as an optional planner.

## Why this project?

This is not a chat UI with tools bolted on. It is an agent runtime with explicit boundaries:
- typed task plans and validated tool inputs/outputs;
- a registry that exposes only approved tools;
- native OpenAI function schemas generated from Pydantic models;
- bounded multi-turn planning, tool steps, retries, and execution time;
- structured traces and persistent task history;
- short-term SQLite history and optional ChromaDB semantic memory;
- memory deduplication, expiry, importance ranking, and retention caps;
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
          SQLite + ChromaDB Memory
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

## Docker deployment

Docker Compose runs the API as a non-root user and persists SQLite/Chroma data in a named volume. Copy `.env.example` to `.env`, set a long random `AGENT_API_KEY`, then run:

```bash
docker compose up --build -d
docker compose logs -f agentic-api
```

The API listens on `http://localhost:8000`; `/health` is intentionally public for container health checks. All task, trace, and session-memory endpoints require `X-API-Key` when `AGENT_API_KEY` is configured. Include the header in API requests:

```bash
curl http://localhost:8000/tasks -H "X-API-Key: $AGENT_API_KEY" -H "Content-Type: application/json" -d '{"task":"Calculate 19 * 23"}'
```

For local development, an unset API key leaves authentication disabled. **Do not expose the service publicly without setting `AGENT_API_KEY`.** Rate limiting is per client IP and process; it suits a single instance, not multi-worker or distributed deployments. Use a trusted reverse proxy/API gateway for horizontally scaled deployments.

## Enable semantic memory

Semantic memory is optional and disabled by default. It uses ChromaDB for persistent vector storage and SentenceTransformers for local embeddings.

```bash
pip install -e ".[dev,semantic-memory]"
```

Configure `.env`:

```dotenv
AGENT_SEMANTIC_MEMORY_ENABLED=true
AGENT_SEMANTIC_MEMORY_PATH=./data/semantic_memory
AGENT_SEMANTIC_MEMORY_MODEL=sentence-transformers/all-MiniLM-L6-v2
AGENT_SEMANTIC_MEMORY_TOP_K=3
AGENT_SEMANTIC_MEMORY_MAX_ITEMS=100
AGENT_SEMANTIC_MEMORY_TTL_DAYS=0
AGENT_SEMANTIC_MEMORY_DUPLICATE_DISTANCE=0.08
```

The embedding model is downloaded on first use. Memories are scoped by `session_id`; retrieval never searches across sessions. Each successful interaction is stored as a user/assistant memory pair. A TTL of `0` means no automatic expiry. The per-session cap evicts lower-importance, older memories first. Keep this feature disabled if local vector dependencies are not installed.

## Example

```bash
curl -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"task":"Calculate 19 * 23"}'
```

Reuse the returned `session_id` in subsequent requests to maintain continuity.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `AGENT_RUNTIME_BACKEND` | `langgraph` or `legacy` | `langgraph` |
| `AGENT_PLANNER` | `rules` or `openai` | `rules` |
| `AGENT_OPENAI_STRATEGY` | `tool_calling` or `json_plan` | `tool_calling` |
| `OPENAI_API_KEY` | Enables the optional OpenAI planner | unset |
| `AGENT_API_KEY` | Protects API endpoints through `X-API-Key`; unset disables auth for local development | unset |
| `AGENT_RATE_LIMIT_PER_MINUTE` | Per-IP request limit per process | `60` |
| `OPENAI_MODEL` | Model for planning | `gpt-4.1-mini` |
| `AGENT_MAX_STEPS` | Maximum tool calls per task | `5` |
| `AGENT_MAX_RETRIES` | Maximum retries after tool failures | `1` |
| `AGENT_TIMEOUT_SECONDS` | Total execution budget | `20` |
| `AGENT_DATABASE_PATH` | SQLite database file | `./data/agentic_platform.db` |
| `AGENT_MEMORY_MAX_MESSAGES` | Maximum short-term messages per session | `20` |
| `AGENT_SEMANTIC_MEMORY_ENABLED` | Enable ChromaDB vector memory | `false` |
| `AGENT_SEMANTIC_MEMORY_PATH` | Persistent vector database path | `./data/semantic_memory` |
| `AGENT_SEMANTIC_MEMORY_MODEL` | Local embedding model | `all-MiniLM-L6-v2` |
| `AGENT_SEMANTIC_MEMORY_TOP_K` | Retrieved memories per task | `3` |
| `AGENT_SEMANTIC_MEMORY_MAX_ITEMS` | Maximum vector memories per session | `100` |
| `AGENT_SEMANTIC_MEMORY_TTL_DAYS` | Memory expiry in days; `0` disables expiry | `0` |
| `AGENT_SEMANTIC_MEMORY_DUPLICATE_DISTANCE` | Cosine-distance threshold for deduplication | `0.08` |

Set `AGENT_PLANNER=openai` and provide `OPENAI_API_KEY` to enable model-directed tool selection. Native function calling is the default OpenAI strategy. Set `AGENT_OPENAI_STRATEGY=json_plan` to use the legacy JSON-plan adapter. If no API key is configured, the runtime safely falls back to the deterministic rules planner.

## Persistence and API

Task responses, ordered trace events, and short-term session history are persisted in SQLite. When enabled, ChromaDB stores vector embeddings for completed user/assistant turns. Both memory layers are scoped to the session ID.

- `POST /tasks` executes a task and stores its report.
- `GET /tasks/{task_id}` retrieves a persisted task report.
- `GET /tasks/{task_id}/trace` retrieves its ordered execution trace.
- `GET /sessions/{session_id}/memory` retrieves short-term conversation history.
- `DELETE /sessions/{session_id}/memory` clears both memory layers for that session.
- `GET /sessions/{session_id}/semantic-memory` lists active vector memories.
- `DELETE /sessions/{session_id}/semantic-memory/{memory_id}` deletes one memory owned by that session.
- `POST /sessions/{session_id}/semantic-memory/prune` removes expired and over-capacity memories.

## Safety model

- Tools are explicitly registered; arbitrary Python or shell execution is not exposed.
- OpenAI receives only function schemas exported by the allow-listed registry.
- Tool arguments are validated with Pydantic before execution.
- Each task has a bounded number of steps, retries, and wall-clock time.
- Retrieved memory is provided as context, not as executable instructions.
- Vector retrieval and deletion are isolated by session ID.
- Memory retention is bounded; TTL and explicit deletion are supported.
- No secrets are returned in task results or traces.
- Configured API keys use constant-time comparison and are never logged.
- Task, trace, and memory endpoints are authenticated when a key is configured; health checks remain public.
- Per-process rate limiting returns HTTP 429 and a `Retry-After` header.

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
- [x] SQLite task and trace storage
- [x] Session-scoped persistent conversation memory
- [x] OpenAI native function-calling planner
- [x] Multi-turn tool observations and model-directed replanning
- [x] Optional ChromaDB semantic memory with session-isolated retrieval
- [x] Memory deduplication, expiry, retention limits, and management endpoints
- [x] Optional API-key authentication and per-process rate limiting
- [x] Non-root Docker image, Compose deployment, persistent volume, and health check

## License

MIT
