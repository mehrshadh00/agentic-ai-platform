from fastapi.testclient import TestClient

from agentic_platform.api import app, rate_limiter
from agentic_platform.config import settings

client = TestClient(app)


def test_prometheus_metrics_endpoint_exposes_application_metrics():
    rate_limiter.reset()
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "agentic_http_requests_total" in response.text
    assert "agentic_tasks_total" in response.text
    assert "agentic_llm_tokens_total" in response.text


def test_metrics_endpoint_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "agent_api_key", "metrics-secret")
    rate_limiter.reset()
    assert client.get("/metrics").status_code == 401
    allowed = client.get("/metrics", headers={"X-API-Key": "metrics-secret"})
    assert allowed.status_code == 200
    monkeypatch.setattr(settings, "agent_api_key", None)
    rate_limiter.reset()
