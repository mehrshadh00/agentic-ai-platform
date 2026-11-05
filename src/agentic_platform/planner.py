import re
from typing import Protocol

from agentic_platform.models import Plan, ToolCall


class Planner(Protocol):
    def create_plan(self, task: str) -> Plan: ...


class RulesPlanner:
    """Credential-free baseline planner for reproducible local execution."""

    def create_plan(self, task: str) -> Plan:
        lowered = task.lower()
        expression_pattern = (
            r"[-+]?\d+(?:\.\d+)?\s*[+*/%\-]\s*[-+]?\d+(?:\.\d+)?"
        )
        match = re.search(expression_pattern, task)
        calculation_intent = ("calculate", "compute", "what is", "solve")
        if match and any(word in lowered for word in calculation_intent):
            return Plan(
                summary="Evaluate the arithmetic expression with the restricted calculator.",
                steps=[
                    ToolCall(
                        tool_name="calculator",
                        arguments={"expression": match.group(0)},
                        rationale="The request contains a simple arithmetic expression.",
                    )
                ],
            )
        if any(word in lowered for word in ("time", "date", "utc")):
            return Plan(
                summary="Retrieve the current UTC timestamp.",
                steps=[
                    ToolCall(
                        tool_name="utc_time",
                        rationale="The request asks for current time.",
                    )
                ],
            )
        return Plan(
            summary="Search the local technical knowledge base for relevant context.",
            steps=[
                ToolCall(
                    tool_name="knowledge_search",
                    arguments={"query": task},
                    rationale="Use the approved knowledge search tool to gather context.",
                )
            ],
        )


class OpenAIPlanner:
    """Optional LLM planner. It can propose calls but cannot execute tools itself."""

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def create_plan(self, task: str) -> Plan:
        import json

        from pydantic import ValidationError

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Create a concise JSON plan with keys summary and steps. "
                        "Each step has tool_name, arguments, rationale. Allowed tools: "
                        "calculator(expression), knowledge_search(query), utc_time({}). "
                        "Never invent tools. Maximum 5 steps. Return JSON only."
                    ),
                },
                {"role": "user", "content": task},
            ],
        )
        try:
            payload = json.loads(response.choices[0].message.content or "{}")
            return Plan.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            raise RuntimeError("Planner returned an invalid plan") from exc


class OpenAIToolCallingPlanner:
    """Use native OpenAI function calling to select registered tools."""

    def __init__(self, api_key: str, model: str, registry=None, client=None) -> None:
        from openai import OpenAI

        from agentic_platform.tools import build_registry

        self.client = client or OpenAI(api_key=api_key)
        self.model = model
        self.registry = registry or build_registry()

    def create_plan(self, task: str) -> Plan:
        import json

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            tool_choice="required",
            parallel_tool_calls=False,
            tools=self.registry.openai_tools(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Select the approved function tool or tools needed to complete "
                        "the user's task. Only call tools exposed in the tool list. "
                        "Use valid JSON arguments matching each tool schema. "
                        "Do not attempt shell, filesystem, network, or arbitrary code execution."
                    ),
                },
                {"role": "user", "content": task},
            ],
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            raise RuntimeError("OpenAI did not return a required tool call")

        steps = []
        for item in tool_calls:
            try:
                arguments = json.loads(item.function.arguments or "{}")
            except (TypeError, ValueError) as exc:
                raise RuntimeError("OpenAI returned invalid tool arguments") from exc
            steps.append(
                ToolCall(
                    tool_name=item.function.name,
                    arguments=arguments,
                    rationale="Selected by OpenAI native function calling.",
                )
            )
        return Plan(
            summary="OpenAI selected tools using native function calling.",
            steps=steps,
        )
