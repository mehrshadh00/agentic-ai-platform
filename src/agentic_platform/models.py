from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1, max_length=80)
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="", max_length=500)
    tool_call_id: str | None = None


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[ToolCall] = Field(default_factory=list, max_length=20)
    summary: str = Field(default="", max_length=500)
    assistant_message: dict[str, Any] | None = None
    final_answer: str | None = None


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0


class TraceEvent(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    event: str
    details: dict[str, Any] = Field(default_factory=dict)


class TaskRequest(BaseModel):
    task: str = Field(min_length=3, max_length=4000)


class TaskResponse(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    status: Literal["completed", "partial", "failed"]
    answer: str
    steps_executed: int
    tool_results: list[ToolResult]
    trace: list[TraceEvent]
