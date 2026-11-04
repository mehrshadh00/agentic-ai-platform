from agentic_platform.models import Plan, ToolCall, ToolResult
from agentic_platform.runtime import AgentRuntime


class FixedPlanner:
    def create_plan(self, task: str) -> Plan:
        return Plan(
            steps=[
                ToolCall(
                    tool_name="calculator",
                    arguments={"expression": "19 * 23"},
                )
            ]
        )


class BrokenPlanner:
    def create_plan(self, task: str) -> Plan:
        raise RuntimeError("planner unavailable")


def test_runtime_executes_plan_and_returns_trace():
    result = AgentRuntime(planner=FixedPlanner()).run("calculate 19 * 23")
    assert result.status == "completed"
    assert result.steps_executed == 1
    assert "437" in result.answer
    assert any(event.event == "tool.completed" for event in result.trace)


def test_runtime_handles_planner_failure():
    result = AgentRuntime(planner=BrokenPlanner()).run("do something")
    assert result.status == "failed"
    assert result.steps_executed == 0
    assert any(event.event == "planner.failed" for event in result.trace)


def test_runtime_records_tool_failure():
    class UnknownToolPlanner:
        def create_plan(self, task: str) -> Plan:
            return Plan(steps=[ToolCall(tool_name="not_allowed", arguments={})])

    result = AgentRuntime(planner=UnknownToolPlanner()).run("run an unknown tool")
    assert result.status == "failed"
    assert any(event.event == "tool.failed" for event in result.trace)
    assert any(event.event == "tool.reflection" for event in result.trace)
    assert not any(event.event == "tool.retry" for event in result.trace)


def test_runtime_retries_transient_tool_failure():
    runtime = AgentRuntime(planner=FixedPlanner())
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
