from agentic_platform.models import Plan, ToolCall
from agentic_platform.runtime import AgentRuntime


class FixedPlanner:
    def create_plan(self, task: str) -> Plan:
        return Plan(steps=[
            ToolCall(tool_name="calculator", arguments={"expression": "19 * 23"}),
        ])


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
