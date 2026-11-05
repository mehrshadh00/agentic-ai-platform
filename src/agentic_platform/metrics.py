"""Prometheus instrumentation for API and agent execution."""

from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter(
    "agentic_http_requests_total",
    "HTTP requests handled by the API.",
    ("method", "route", "status"),
)
HTTP_REQUEST_DURATION = Histogram(
    "agentic_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ("method", "route"),
)
TASKS_TOTAL = Counter(
    "agentic_tasks_total", "Agent tasks by final outcome.", ("status",)
)
TASK_DURATION = Histogram(
    "agentic_task_duration_seconds",
    "End-to-end agent task duration.",
    ("status",),
)
TOOL_CALLS_TOTAL = Counter(
    "agentic_tool_calls_total",
    "Tool calls by tool and outcome.",
    ("tool", "outcome"),
)
TOOL_DURATION = Histogram(
    "agentic_tool_duration_seconds",
    "Tool execution duration in seconds.",
    ("tool",),
)
LLM_TOKENS_TOTAL = Counter(
    "agentic_llm_tokens_total",
    "LLM tokens reported by provider usage metadata.",
    ("model", "kind"),
)


def record_llm_usage(model: str, usage: object | None) -> None:
    """Record provider-reported token usage when available."""
    if usage is None:
        return
    values = (
        ("prompt", getattr(usage, "prompt_tokens", None)),
        ("completion", getattr(usage, "completion_tokens", None)),
        ("total", getattr(usage, "total_tokens", None)),
    )
    for kind, value in values:
        if isinstance(value, int) and value >= 0:
            LLM_TOKENS_TOTAL.labels(model, kind).inc(value)
