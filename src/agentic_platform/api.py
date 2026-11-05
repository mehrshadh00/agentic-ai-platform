from uuid import uuid4

from fastapi import FastAPI, HTTPException

from agentic_platform.config import settings
from agentic_platform.memory import SQLiteConversationMemory
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
    version="0.4.0",
)
runtime = build_runtime()
store = SQLiteTraceStore(settings.agent_database_path)
memory = SQLiteConversationMemory(
    settings.agent_database_path,
    max_messages=settings.agent_memory_max_messages,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "runtime": settings.agent_runtime_backend}


@app.post("/tasks", response_model=TaskResponse)
def create_task(request: TaskRequest) -> TaskResponse:
    session_id = request.session_id or str(uuid4())
    try:
        history = memory.get_history(session_id)
        if history:
            context = "\\n".join(
                f"{item['role']}: {item['content']}" for item in history
            )
            task = (
                "Relevant conversation history for this session follows. "
                "Use it only when it helps answer the current request.\\n"
                f"{context}\\n\\nCurrent request: {request.task}"
            )
        else:
            task = request.task
        response = runtime.run(task).model_copy(update={"session_id": session_id})
        store.save_response(response)
        memory.add_turn(session_id, request.task, response.answer)
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


@app.get("/sessions/{session_id}/memory")
def get_session_memory(session_id: str) -> dict[str, object]:
    return {"session_id": session_id, "messages": memory.get_history(session_id)}
