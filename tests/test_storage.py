from agentic_platform.models import TaskResponse, TraceEvent
from agentic_platform.storage import SQLiteTraceStore


def sample_response() -> TaskResponse:
    return TaskResponse(
        task_id="task-123",
        status="completed",
        answer="The result is 437.",
        steps_executed=1,
        tool_results=[],
        trace=[
            TraceEvent(event="plan.created", details={"steps": 1}),
            TraceEvent(event="task.completed", details={"answer": "437"}),
        ],
    )


def test_task_and_trace_round_trip(tmp_path):
    store = SQLiteTraceStore(tmp_path / "agent.db")
    response = sample_response()

    store.save_response(response)

    restored = store.get_response("task-123")
    trace = store.get_trace("task-123")

    assert restored is not None
    assert restored.model_dump() == response.model_dump()
    assert trace is not None
    assert [event.event for event in trace] == ["plan.created", "task.completed"]


def test_missing_task_returns_none(tmp_path):
    store = SQLiteTraceStore(tmp_path / "agent.db")

    assert store.get_response("missing") is None
    assert store.get_trace("missing") is None
