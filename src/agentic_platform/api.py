from fastapi import FastAPI, HTTPException

from agentic_platform.models import TaskRequest, TaskResponse
from agentic_platform.runtime import AgentRuntime

app = FastAPI(
    title="Agentic AI Platform",
    description="A bounded and observable agent runtime with validated tool execution.",
    version="0.1.0",
)
runtime = AgentRuntime()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tasks", response_model=TaskResponse)
def create_task(request: TaskRequest) -> TaskResponse:
    try:
        return runtime.run(request.task)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Task execution failed") from exc
