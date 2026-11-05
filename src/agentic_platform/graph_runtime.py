import json
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
    total_steps: int
    attempt: int
    history: list[dict[str, object]]
    tool_results: list[ToolResult]
    final_results: list[ToolResult]
    trace: list[TraceEvent]
    started_at: float
    status: str
    answer: str
    final_answer: str | None
    steps_executed: int


class GraphAgentRuntime:
    """LangGraph agent with bounded, model-directed multi-turn tool execution."""

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
            {"plan": "plan", "execute_tool": "execute_tool", "finalize": "finalize"},
        )
        graph.add_edge("finalize", END)
        self.graph = graph.compile()

    def _plan(self, state: AgentState) -> dict:
        trace = list(state["trace"])
        history = list(state.get("history", []))
        try:
            if hasattr(self.planner, "create_plan_with_history"):
                plan = self.planner.create_plan_with_history(state["task"], history)
            else:
                plan = self.planner.create_plan(state["task"])
            if plan.assistant_message is not None:
                history.append(plan.assistant_message)
            trace.append(
                TraceEvent(
                    event="plan.created",
                    details={
                        "summary": plan.summary,
                        "step_count": len(plan.steps),
                        "turn": sum(1 for item in trace if item.event == "plan.created") + 1,
                    },
                )
            )
            if not plan.steps:
                trace.append(
                    TraceEvent(
                        event="agent.completed" if plan.final_answer else "agent.no_action",
                        details={"has_final_answer": bool(plan.final_answer)},
                    )
                )
            return {
                "plan": plan,
                "step_index": 0,
                "attempt": 0,
                "history": history,
                "final_answer": plan.final_answer,
                "trace": trace,
            }
        except Exception as exc:
            trace.append(TraceEvent(event="planner.failed", details={"error": str(exc)}))
            return {"plan": None, "status": "failed", "trace": trace}

    @staticmethod
    def _after_plan(state: AgentState) -> str:
        plan = state.get("plan")
        return "execute_tool" if plan is not None and plan.steps else "finalize"

    def _execute_tool(self, state: AgentState) -> dict:
        plan = state["plan"]
        index = state["step_index"]
        attempt = state["attempt"]
        trace = list(state["trace"])
        history = list(state.get("history", []))
        results = list(state["tool_results"])
        final_results = list(state["final_results"])
        elapsed = time.monotonic() - state["started_at"]

        if state.get("total_steps", 0) >= settings.agent_max_steps:
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
                    "step": state.get("total_steps", 0) + 1,
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
                    "step": state.get("total_steps", 0) + 1,
                    "attempt": attempt + 1,
                    "tool": call.tool_name,
                    "duration_ms": round(result.duration_ms, 3),
                    "error": result.error,
                },
            )
        )

        if result.success:
            final_results.append(result)
        if call.tool_call_id:
            observation = (
                json.dumps(result.output, ensure_ascii=False, default=str)
                if result.success
                else json.dumps({"error": result.error}, ensure_ascii=False)
            )
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call.tool_call_id,
                    "content": observation,
                }
            )

        if result.success:
            return {
                "tool_results": results,
                "final_results": final_results,
                "history": history,
                "step_index": index + 1,
                "total_steps": state.get("total_steps", 0) + 1,
                "attempt": 0,
                "trace": trace,
            }

        retryable = _is_retryable(result)
        trace.append(
            TraceEvent(
                event="tool.reflection",
                details={"step": state.get("total_steps", 0) + 1, "retryable": retryable, "reason": result.error},
            )
        )
        if retryable and attempt < settings.agent_max_retries:
            trace.append(
                TraceEvent(
                    event="tool.retry",
                    details={"step": state.get("total_steps", 0) + 1, "attempt": attempt + 2},
                )
            )
            return {
                "tool_results": results,
                "history": history,
                "attempt": attempt + 1,
                "trace": trace,
            }

        final_results.append(result)
        return {
            "tool_results": results,
            "final_results": final_results,
            "history": history,
            "step_index": index + 1,
            "total_steps": state.get("total_steps", 0) + 1,
            "attempt": 0,
            "trace": trace,
        }

    def _after_tool(self, state: AgentState) -> str:
        plan = state["plan"]
        if state["step_index"] < len(plan.steps):
            if state.get("total_steps", 0) >= settings.agent_max_steps:
                return "finalize"
            if time.monotonic() - state["started_at"] > settings.agent_timeout_seconds:
                return "finalize"
            return "execute_tool"
        if hasattr(self.planner, "create_plan_with_history"):
            if state.get("total_steps", 0) < settings.agent_max_steps:
                if time.monotonic() - state["started_at"] <= settings.agent_timeout_seconds:
                    return "plan"
        return "finalize"

    @staticmethod
    def _finalize(state: AgentState) -> dict:
        trace = list(state["trace"])
        final_results = state.get("final_results", [])
        successful = [item for item in final_results if item.success]
        failed = [item for item in final_results if not item.success]

        if state.get("status") == "failed" and state.get("plan") is None:
            status = "failed"
            answer = "I could not create a valid plan."
        elif state.get("final_answer"):
            status = "completed" if not failed else "partial"
            answer = state["final_answer"]
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
                    "steps_executed": state.get("total_steps", len(final_results)),
                    "tool_attempts": len(state.get("tool_results", [])),
                },
            )
        )
        return {
            "status": status,
            "answer": answer,
            "steps_executed": state.get("total_steps", len(final_results)),
            "trace": trace,
        }

    def run(self, task: str) -> TaskResponse:
        task_id = str(uuid4())
        initial: AgentState = {
            "task": task,
            "task_id": task_id,
            "plan": None,
            "step_index": 0,
            "total_steps": 0,
            "attempt": 0,
            "history": [],
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
