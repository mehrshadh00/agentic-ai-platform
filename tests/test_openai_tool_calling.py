from types import SimpleNamespace

import pytest

from agentic_platform.planner import OpenAIToolCallingPlanner
from agentic_platform.tools import build_registry


class FakeCompletions:
    def __init__(self, message):
        self.message = message
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=self.message)]
        )


class FakeClient:
    def __init__(self, message):
        self.completions = FakeCompletions(message)
        self.chat = SimpleNamespace(completions=self.completions)


def make_tool_call(name="calculator", arguments='{"expression":"19 * 23"}', call_id="call-123"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments)
    )


def test_native_tool_calling_uses_registry_schemas_and_parses_calls():
    client = FakeClient(
        SimpleNamespace(tool_calls=[make_tool_call()])
    )
    planner = OpenAIToolCallingPlanner(
        api_key="test-key",
        model="test-model",
        registry=build_registry(),
        client=client,
    )

    plan = planner.create_plan("Calculate 19 * 23")

    request = client.completions.request
    assert request["tool_choice"] == "auto"
    assert request["parallel_tool_calls"] is False
    assert {item["function"]["name"] for item in request["tools"]} == {
        "calculator",
        "knowledge_search",
        "utc_time",
    }
    assert plan.steps[0].tool_name == "calculator"
    assert plan.steps[0].arguments == {"expression": "19 * 23"}


def test_native_tool_calling_can_return_multiple_calls():
    client = FakeClient(
        SimpleNamespace(
            tool_calls=[
                make_tool_call("calculator", '{"expression":"2 + 2"}'),
                make_tool_call("utc_time", "{}"),
            ]
        )
    )
    planner = OpenAIToolCallingPlanner(
        api_key="test-key", model="test-model", client=client
    )

    plan = planner.create_plan("Calculate and get the current time")

    assert [step.tool_name for step in plan.steps] == ["calculator", "utc_time"]


def test_native_tool_calling_rejects_invalid_json_arguments():
    client = FakeClient(
        SimpleNamespace(tool_calls=[make_tool_call(arguments="{invalid")])
    )
    planner = OpenAIToolCallingPlanner(
        api_key="test-key", model="test-model", client=client
    )

    with pytest.raises(RuntimeError, match="invalid tool arguments"):
        planner.create_plan("calculate something")


def test_native_tool_calling_accepts_final_answer_without_tool_calls():
    client = FakeClient(SimpleNamespace(tool_calls=None, content="The task is complete."))
    planner = OpenAIToolCallingPlanner(
        api_key="test-key", model="test-model", client=client
    )

    plan = planner.create_plan("calculate something")

    assert plan.steps == []
    assert plan.final_answer == "The task is complete."
