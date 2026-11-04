from fastapi import FastAPI, HTTPException

from agentic_platform.config import settings
from agentic_platform.models import TaskRequest, TaskResponse
from agentic_platform.runtime import AgentRuntime


def build_runtime():
    if settings.agent_runtime_backend.lower() == "langgraph":
        from agentic_platform.graph_runtime import GraphAgentRuntime

        return GraphAgentRuntime()
    return AgentRuntime()


app = FastAPI(
    title="Agentic AI Platform",
    description="A bounded and observable agent runtime with validated tool execution.",
    version="0.2.0",
)
runtime = build_runtime()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "runtime": settings.agent_runtime_backend}


@app.post("/tasks", response_model=TaskResponse)
def create_task(request: TaskRequest) -> TaskResponse:
    try:
        return runtime.run(request.task)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Task execution failed") from exc
