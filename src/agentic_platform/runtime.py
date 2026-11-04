import time
from uuid import uuid4

from agentic_platform.config import settings
from agentic_platform.models import TaskResponse, ToolResult, TraceEvent
from agentic_platform.planner import OpenAIPlanner, Planner, RulesPlanner
from agentic_platform.tools import build_registry


def get_planner() -> Planner:
    if settings.agent_planner.lower() == "openai" and settings.openai_api_key:
        return OpenAIPlanner(settings.openai_api_key, settings.openai_model)
    return RulesPlanner()


def _is_retryable(result: ToolResult) -> bool:
    """Avoid retrying invalid requests and other deterministic failures."""
    if result.success or not result.error:
        return False
    permanent_markers = (
        "Unknown or disallowed tool",
        "ValidationError",
        "ValueError",
        "SyntaxError",
        "ZeroDivisionError",
    )
    return not any(marker in result.error for marker in permanent_markers)


class AgentRuntime:
    def __init__(self, planner: Planner | None = None) -> None:
        self.planner = planner or get_planner()
        self.registry = build_registry()

    def run(self, task: str) -> TaskResponse:
        task_id = str(uuid4())
        trace = [TraceEvent(event="task.started", details={"task_id": task_id})]
        results: list[ToolResult] = []
        final_results: list[ToolResult] = []
        started = time.monotonic()

        try:
            plan = self.planner.create_plan(task)
            trace.append(
                TraceEvent(
                    event="plan.created",
                    details={"summary": plan.summary, "step_count": len(plan.steps)},
                )
            )
        except Exception as exc:
            trace.append(TraceEvent(event="planner.failed", details={"error": str(exc)}))
            return TaskResponse(
                task_id=task_id,
                status="failed",
                answer="I could not create a valid plan.",
                steps_executed=0,
                tool_results=results,
                trace=trace,
            )

        for index, call in enumerate(plan.steps):
            if index >= settings.agent_max_steps:
                trace.append(TraceEvent(event="execution.limit_reached"))
                break
            if time.monotonic() - started > settings.agent_timeout_seconds:
                trace.append(TraceEvent(event="execution.timeout"))
                break

            attempt = 0
            step_result: ToolResult | None = None
            while True:
                if time.monotonic() - started > settings.agent_timeout_seconds:
                    trace.append(
                        TraceEvent(
                            event="execution.timeout",
                            details={"step": index + 1, "attempt": attempt + 1},
                        )
                    )
                    break

                trace.append(
                    TraceEvent(
                        event="tool.started",
                        details={
                            "step": index + 1,
                            "attempt": attempt + 1,
                            "tool": call.tool_name,
                            "rationale": call.rationale,
                        },
                    )
                )
                result = self.registry.execute(call.tool_name, call.arguments)
                results.append(result)
                step_result = result
                trace.append(
                    TraceEvent(
                        event="tool.completed" if result.success else "tool.failed",
                        details={
                            "step": index + 1,
                            "attempt": attempt + 1,
                            "tool": call.tool_name,
                            "duration_ms": round(result.duration_ms, 3),
                            "error": result.error,
                        },
                    )
                )

                if result.success:
                    break

                retryable = _is_retryable(result)
                trace.append(
                    TraceEvent(
                        event="tool.reflection",
                        details={
                            "step": index + 1,
                            "retryable": retryable,
                            "reason": result.error,
                        },
                    )
                )
                if not retryable or attempt >= settings.agent_max_retries:
                    break

                attempt += 1
                trace.append(
                    TraceEvent(
                        event="tool.retry",
                        details={"step": index + 1, "attempt": attempt + 1},
                    )
                )

            if step_result is not None:
                final_results.append(step_result)

        successful = [item for item in final_results if item.success]
        failed = [item for item in final_results if not item.success]
        if successful:
            answer = "\n".join(
                f"{item.tool_name}: {item.output}" for item in successful
            )
            status = "completed" if not failed else "partial"
        else:
            answer = "The task could not be completed with the available tools."
            status = "failed"

        trace.append(
            TraceEvent(
                event="task.finished",
                details={
                    "status": status,
                    "steps_executed": len(final_results),
                    "tool_attempts": len(results),
                },
            )
        )
        return TaskResponse(
            task_id=task_id,
            status=status,
            answer=answer,
            steps_executed=len(final_results),
            tool_results=results,
            trace=trace,
        )
