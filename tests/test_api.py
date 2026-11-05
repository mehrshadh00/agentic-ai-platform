from fastapi.testclient import TestClient

from agentic_platform.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {\n        "status": "ok",\n        "runtime": "langgraph",\n        "semantic_memory": False,\n    }


def test_task_endpoint_returns_execution_report():
    response = client.post("/tasks", json={"task": "Calculate 19 * 23"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["steps_executed"] == 1
    assert "437" in payload["answer"]
    assert payload["trace"]

    task_id = payload["task_id"]
    persisted = client.get(f"/tasks/{task_id}")
    persisted_trace = client.get(f"/tasks/{task_id}/trace")

    assert persisted.status_code == 200
    assert persisted.json() == payload
    assert persisted_trace.status_code == 200
    assert persisted_trace.json() == payload["trace"]


def test_unknown_task_returns_404():
    assert client.get("/tasks/not-a-real-task").status_code == 404
    assert client.get("/tasks/not-a-real-task/trace").status_code == 404


def test_task_session_persists_conversation_memory():
    response = client.post(
        "/tasks",
        json={"task": "Calculate 19 * 23", "session_id": "memory-test-session"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "memory-test-session"

    history = client.get("/sessions/memory-test-session/memory")
    assert history.status_code == 200
    messages = history.json()["messages"]
    assert messages[0] == {"role": "user", "content": "Calculate 19 * 23"}
    assert messages[1]["role"] == "assistant"
    assert "437" in messages[1]["content"]
