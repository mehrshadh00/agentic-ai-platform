from fastapi import FastAPI, HTTPException

from agentic_platform.config import settings
from agentic_platform.models import TaskRequest, TaskResponse, TraceEvent
from agentic_platform.runtime import AgentRuntime
from agentic_platform.storage import SQLiteTraceStore


def build_runtime():
    if settings.agent_runtime_backend.lower() == "langgraph":
        from agentic_platform.graph_runtime import GraphAgentRuntime

        return GraphAgentRuntime()
    return AgentRuntime()


app = FastAPI(
    title="Agentic AI Platform",
    description="A bounded and observable agent runtime with validated tool execution.",
    version="0.3.0",
)
runtime = build_runtime()
store = SQLiteTraceStore(settings.agent_database_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "runtime": settings.agent_runtime_backend}


@app.post("/tasks", response_model=TaskResponse)
def create_task(request: TaskRequest) -> TaskResponse:
    try:
        response = runtime.run(request.task)
        store.save_response(response)
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Task execution failed") from exc


@app.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str) -> TaskResponse:
    response = store.get_response(task_id)
    if response is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return response


@app.get("/tasks/{task_id}/trace", response_model=list[TraceEvent])
def get_task_trace(task_id: str) -> list[TraceEvent]:
    trace = store.get_trace(task_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return trace
