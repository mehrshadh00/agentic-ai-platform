import re
from typing import Protocol

from agentic_platform.metrics import record_llm_usage
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
        record_llm_usage(self.model, getattr(response, "usage", None))
        try:
            payload = json.loads(response.choices[0].message.content or "{}")
            return Plan.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            raise RuntimeError("Planner returned an invalid plan") from exc


class OpenAIToolCallingPlanner:
    """Use native OpenAI function calling with multi-turn tool observations."""

    def __init__(self, api_key: str, model: str, registry=None, client=None) -> None:
        from openai import OpenAI

        from agentic_platform.tools import build_registry

        self.client = client or OpenAI(api_key=api_key)
        self.model = model
        self.registry = registry or build_registry()

    def create_plan(self, task: str) -> Plan:
        return self.create_plan_with_history(task, [])

    def create_plan_with_history(
        self, task: str, history: list[dict[str, object]]
    ) -> Plan:
        import json

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            tool_choice="auto",
            parallel_tool_calls=False,
            tools=self.registry.openai_tools(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an agent that completes the user's task using approved tools. "
                        "Use only the provided tools. After receiving tool results, decide whether "
                        "another tool is needed. If the task is complete, respond with a concise "
                        "final answer and do not call a tool. Never request shell, filesystem, "
                        "network, or arbitrary code execution."
                    ),
                },
                {"role": "user", "content": task},
                *history,
            ],
        )
        record_llm_usage(self.model, getattr(response, "usage", None))
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        assistant_message = {
            "role": "assistant",
            "content": getattr(message, "content", None),
            "tool_calls": [
                {
                    "id": item.id,
                    "type": "function",
                    "function": {
                        "name": item.function.name,
                        "arguments": item.function.arguments or "{}",
                    },
                }
                for item in tool_calls
            ],
        } if tool_calls else None

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
                    tool_call_id=item.id,
                )
            )
        return Plan(
            summary=(
                "OpenAI selected tools using native function calling."
                if steps
                else "OpenAI completed the task without another tool call."
            ),
            steps=steps,
            assistant_message=assistant_message,
            final_answer=getattr(message, "content", None) if not steps else None,
        )
