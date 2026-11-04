import time
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from agentic_platform.config import settings
from agentic_platform.models import Plan, TaskResponse, ToolCall, ToolResult, TraceEvent
from agentic_platform.planner import Planner
from agentic_platform.runtime import _is_retryable, get_planner
from agentic_platform.tools import ToolRegistry, build_registry


class AgentState(TypedDict, total=False):
    task: str
    task_id: str
    plan: Plan | None
    step_index: int
    attempt: int
    tool_results: list[ToolResult]
    final_results: list[ToolResult]
    trace: list[TraceEvent]
    started_at: float
    status: str
    answer: str
    steps_executed: int


class GraphAgentRuntime:
    """LangGraph-backed agent execution with bounded tool retries."""

    def __init__(
        self,
        planner: Planner | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.planner = planner or get_planner()
        self.registry = registry or build_registry()

        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("execute_tool", self._execute_tool)
        graph.add_node("finalize", self._finalize)
        graph.set_entry_point("plan")
        graph.add_conditional_edges(
            "plan",
            self._after_plan,
            {"execute_tool": "execute_tool", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "execute_tool",
            self._after_tool,
            {"execute_tool": "execute_tool", "finalize": "finalize"},
        )
        graph.add_edge("finalize", END)
        self.graph = graph.compile()

    def _plan(self, state: AgentState) -> dict:
        trace = list(state["trace"])
        try:
            plan = self.planner.create_plan(state["task"])
            trace.append(
                TraceEvent(
                    event="plan.created",
                    details={"summary": plan.summary, "step_count": len(plan.steps)},
                )
            )
            return {"plan": plan, "trace": trace}
        except Exception as exc:
            trace.append(TraceEvent(event="planner.failed", details={"error": str(exc)}))
            return {"plan": None, "status": "failed", "trace": trace}

    @staticmethod
    def _after_plan(state: AgentState) -> str:
        return "execute_tool" if state.get("plan") is not None else "finalize"

    def _execute_tool(self, state: AgentState) -> dict:
        plan = state["plan"]
        index = state["step_index"]
        attempt = state["attempt"]
        trace = list(state["trace"])
        results = list(state["tool_results"])
        final_results = list(state["final_results"])
        elapsed = time.monotonic() - state["started_at"]

        if index >= settings.agent_max_steps:
            trace.append(TraceEvent(event="execution.limit_reached"))
            return {"step_index": len(plan.steps), "trace": trace}
        if elapsed > settings.agent_timeout_seconds:
            trace.append(TraceEvent(event="execution.timeout"))
            return {"step_index": len(plan.steps), "trace": trace}

        call: ToolCall = plan.steps[index]
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
            final_results.append(result)
            return {
                "tool_results": results,
                "final_results": final_results,
                "step_index": index + 1,
                "attempt": 0,
                "trace": trace,
            }

        retryable = _is_retryable(result)
        trace.append(
            TraceEvent(
                event="tool.reflection",
                details={"step": index + 1, "retryable": retryable, "reason": result.error},
            )
        )
        if retryable and attempt < settings.agent_max_retries:
            trace.append(
                TraceEvent(
                    event="tool.retry",
                    details={"step": index + 1, "attempt": attempt + 2},
                )
            )
            return {
                "tool_results": results,
                "attempt": attempt + 1,
                "trace": trace,
            }

        final_results.append(result)
        return {
            "tool_results": results,
            "final_results": final_results,
            "step_index": index + 1,
            "attempt": 0,
            "trace": trace,
        }

    @staticmethod
    def _after_tool(state: AgentState) -> str:
        plan = state["plan"]
        if state["step_index"] >= min(len(plan.steps), settings.agent_max_steps):
            return "finalize"
        if time.monotonic() - state["started_at"] > settings.agent_timeout_seconds:
            return "finalize"
        return "execute_tool"

    @staticmethod
    def _finalize(state: AgentState) -> dict:
        trace = list(state["trace"])
        final_results = state.get("final_results", [])
        successful = [item for item in final_results if item.success]
        failed = [item for item in final_results if not item.success]

        if state.get("status") == "failed" and state.get("plan") is None:
            status = "failed"
            answer = "I could not create a valid plan."
        elif successful:
            status = "completed" if not failed else "partial"
            answer = "\\n".join(
                f"{item.tool_name}: {item.output}" for item in successful
            )
        else:
            status = "failed"
            answer = "The task could not be completed with the available tools."

        trace.append(
            TraceEvent(
                event="task.finished",
                details={
                    "status": status,
                    "steps_executed": len(final_results),
                    "tool_attempts": len(state.get("tool_results", [])),
                },
            )
        )
        return {
            "status": status,
            "answer": answer,
            "steps_executed": len(final_results),
            "trace": trace,
        }

    def run(self, task: str) -> TaskResponse:
        task_id = str(uuid4())
        initial: AgentState = {
            "task": task,
            "task_id": task_id,
            "plan": None,
            "step_index": 0,
            "attempt": 0,
            "tool_results": [],
            "final_results": [],
            "trace": [TraceEvent(event="task.started", details={"task_id": task_id})],
            "started_at": time.monotonic(),
        }
        state = self.graph.invoke(initial)
        return TaskResponse(
            task_id=task_id,
            status=state["status"],
            answer=state["answer"],
            steps_executed=state["steps_executed"],
            tool_results=state["tool_results"],
            trace=state["trace"],
        )
