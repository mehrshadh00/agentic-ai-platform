from fastapi.testclient import TestClient

from agentic_platform.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_task_endpoint_returns_execution_report():
    response = client.post("/tasks", json={"task": "Calculate 19 * 23"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["steps_executed"] == 1
    assert "437" in payload["answer"]
    assert payload["trace"]
