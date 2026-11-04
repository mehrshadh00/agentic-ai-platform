import time
from uuid import uuid4

from agentic_platform.config import settings
from agentic_platform.models import TaskResponse, TraceEvent
from agentic_platform.planner import OpenAIPlanner, Planner, RulesPlanner
from agentic_platform.tools import build_registry


def get_planner() -> Planner:
    if settings.agent_planner.lower() == "openai" and settings.openai_api_key:
        return OpenAIPlanner(settings.openai_api_key, settings.openai_model)
    return RulesPlanner()


class AgentRuntime:
    def __init__(self, planner: Planner | None = None) -> None:
        self.planner = planner or get_planner()
        self.registry = build_registry()

    def run(self, task: str) -> TaskResponse:
        task_id = str(uuid4())
        trace = [TraceEvent(event="task.started", details={"task_id": task_id})]
        results = []
        started = time.monotonic()

        try:
            plan = self.planner.create_plan(task)
            trace.append(TraceEvent(
                event="plan.created",
                details={"summary": plan.summary, "step_count": len(plan.steps)},
            ))
        except Exception as exc:
            trace.append(TraceEvent(event="planner.failed", details={"error": str(exc)}))
            return TaskResponse(
                task_id=task_id, status="failed", answer="I could not create a valid plan.",
                steps_executed=0, tool_results=results, trace=trace,
            )

        for index, call in enumerate(plan.steps):
            if index >= settings.agent_max_steps:
                trace.append(TraceEvent(event="execution.limit_reached"))
                break
            if time.monotonic() - started > settings.agent_timeout_seconds:
                trace.append(TraceEvent(event="execution.timeout"))
                break

            trace.append(TraceEvent(
                event="tool.started",
                details={"step": index + 1, "tool": call.tool_name, "rationale": call.rationale},
            ))
            result = self.registry.execute(call.tool_name, call.arguments)
            results.append(result)
            trace.append(TraceEvent(
                event="tool.completed" if result.success else "tool.failed",
                details={
                    "step": index + 1,
                    "tool": call.tool_name,
                    "duration_ms": round(result.duration_ms, 3),
                    "error": result.error,
                },
            ))

        successful = [item for item in results if item.success]
        if successful:
            answer = "\n".join(f"{item.tool_name}: {item.output}" for item in successful)
            status = "completed" if len(successful) == len(results) else "partial"
        else:
            answer = "The task could not be completed with the available tools."
            status = "failed"

        trace.append(TraceEvent(
            event="task.finished",
            details={"status": status, "steps_executed": len(results)},
        ))
        return TaskResponse(
            task_id=task_id,
            status=status,
            answer=answer,
            steps_executed=len(results),
            tool_results=results,
            trace=trace,
        )
