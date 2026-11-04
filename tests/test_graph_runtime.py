from agentic_platform.graph_runtime import GraphAgentRuntime
from agentic_platform.models import Plan, ToolCall, ToolResult


class CalculatorPlanner:
    def create_plan(self, task: str) -> Plan:
        return Plan(
            summary="Calculate a deterministic expression.",
            steps=[
                ToolCall(
                    tool_name="calculator",
                    arguments={"expression": "19 * 23"},
                )
            ],
        )


class FailingPlanner:
    def create_plan(self, task: str) -> Plan:
        raise RuntimeError("planner unavailable")


def test_langgraph_runtime_executes_a_plan():
    result = GraphAgentRuntime(planner=CalculatorPlanner()).run("calculate 19 * 23")

    assert result.status == "completed"
    assert result.steps_executed == 1
    assert "437" in result.answer
    assert any(event.event == "plan.created" for event in result.trace)
    assert any(event.event == "tool.completed" for event in result.trace)
    assert result.trace[-1].event == "task.finished"


def test_langgraph_runtime_records_planner_failure():
    result = GraphAgentRuntime(planner=FailingPlanner()).run("do something")

    assert result.status == "failed"
    assert result.steps_executed == 0
    assert any(event.event == "planner.failed" for event in result.trace)


def test_langgraph_runtime_retries_transient_tool_failure():
    runtime = GraphAgentRuntime(planner=CalculatorPlanner())
    original_execute = runtime.registry.execute
    attempts = 0

    def flaky_execute(name, arguments):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ToolResult(
                tool_name=name,
                success=False,
                error="RuntimeError: temporary backend failure",
            )
        return original_execute(name, arguments)

    runtime.registry.execute = flaky_execute
    result = runtime.run("calculate 19 * 23")

    assert attempts == 2
    assert result.status == "completed"
    assert result.steps_executed == 1
    assert len(result.tool_results) == 2
    assert any(event.event == "tool.reflection" for event in result.trace)
    assert any(event.event == "tool.retry" for event in result.trace)
