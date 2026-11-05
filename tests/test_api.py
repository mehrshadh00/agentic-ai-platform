from fastapi.testclient import TestClient

from agentic_platform.api import app, rate_limiter
from agentic_platform.config import settings

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "runtime": "langgraph",
        "semantic_memory": False,
    }


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
    session_id = "memory-test-session"
    response = client.post(
        "/tasks",
        json={"task": "Calculate 19 * 23", "session_id": session_id},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session_id

    history = client.get(f"/sessions/{session_id}/memory")
    assert history.status_code == 200
    messages = history.json()["messages"]
    assert messages[-2] == {"role": "user", "content": "Calculate 19 * 23"}
    assert messages[-1]["role"] == "assistant"
    assert "437" in messages[-1]["content"]


def test_clear_session_memory_endpoint():
    session_id = "memory-cleanup-test"
    response = client.post(
        "/tasks",
        json={"task": "Calculate 19 * 23", "session_id": session_id},
    )
    assert response.status_code == 200

    cleared = client.delete(f"/sessions/{session_id}/memory")
    assert cleared.status_code == 200
    assert cleared.json()["deleted_short_term_messages"] == 2
    assert cleared.json()["semantic_memory_cleared"] is False
    assert client.get(f"/sessions/{session_id}/memory").json()["messages"] == []


def test_semantic_memory_management_requires_feature_flag():
    response = client.get("/sessions/test-session/semantic-memory")
    assert response.status_code == 503



def test_protected_endpoints_require_api_key_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "agent_api_key", "test-secret")
    rate_limiter.reset()
    denied = client.post("/tasks", json={"task": "Calculate 1 + 1"})
    assert denied.status_code == 401
    assert denied.headers["www-authenticate"] == "ApiKey"
    allowed = client.post(
        "/tasks",
        json={"task": "Calculate 1 + 1"},
        headers={"X-API-Key": "test-secret"},
    )
    assert allowed.status_code == 200
    assert client.get("/health").status_code == 200
    monkeypatch.setattr(settings, "agent_api_key", None)
    rate_limiter.reset()


def test_api_rate_limit_returns_retry_after(monkeypatch):
    monkeypatch.setattr(settings, "agent_api_key", None)
    rate_limiter.requests_per_minute = 1
    rate_limiter.reset()
    first = client.get("/tasks/not-a-real-task")
    second = client.get("/tasks/not-a-real-task")
    assert first.status_code == 404
    assert second.status_code == 429
    assert int(second.headers["retry-after"]) >= 1
    rate_limiter.requests_per_minute = 60
    rate_limiter.reset()
