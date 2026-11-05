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


class MultiTurnPlanner:
    def __init__(self):
        self.turns = 0

    def create_plan(self, task: str) -> Plan:
        raise AssertionError("Multi-turn planner path should be used")

    def create_plan_with_history(self, task: str, history):
        self.turns += 1
        if self.turns == 1:
            return Plan(
                summary="First turn: calculate the expression.",
                steps=[
                    ToolCall(
                        tool_name="calculator",
                        arguments={"expression": "19 * 23"},
                        tool_call_id="call-calc-1",
                    )
                ],
                assistant_message={
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-calc-1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression":"19 * 23"}',
                            },
                        }
                    ],
                },
            )

        assert history[-1]["role"] == "tool"
        assert history[-1]["tool_call_id"] == "call-calc-1"
        assert "437" in history[-1]["content"]
        return Plan(
            summary="Second turn: the tool result is sufficient.",
            final_answer="The calculated result is 437.",
        )


def test_langgraph_runtime_replans_after_observing_tool_output():
    planner = MultiTurnPlanner()
    result = GraphAgentRuntime(planner=planner).run(
        "Calculate 19 * 23 and explain the result."
    )

    assert planner.turns == 2
    assert result.status == "completed"
    assert result.steps_executed == 1
    assert result.answer == "The calculated result is 437."
    assert sum(event.event == "plan.created" for event in result.trace) == 2
    assert any(event.event == "agent.completed" for event in result.trace)
