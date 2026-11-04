import ast
import operator
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from agentic_platform.models import ToolResult


class CalculatorInput(BaseModel):
    expression: str = Field(min_length=1, max_length=200)


class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=200)


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        handler: Callable[[BaseModel], Any],
    ) -> None:
        self.name = name
        self.description = description
        self.input_model = input_model
        self.handler = handler

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            validated = self.input_model.model_validate(arguments)
            output = self.handler(validated)
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=output,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.perf_counter() - started) * 1000,
            )


class ToolRegistry:
    """Allow-list of tools available to the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                tool_name=name,
                success=False,
                error="Unknown or disallowed tool",
            )
        return tool.run(arguments)

    def descriptions(self) -> list[dict[str, str]]:
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]


_ALLOWED_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARYOPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_calculate(expression: str) -> float | int:
    """Evaluate arithmetic AST nodes only; never execute arbitrary Python."""
    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            if isinstance(node.op, ast.Pow) and abs(visit(node.right)) > 8:
                raise ValueError("Exponent magnitude is limited to 8")
            left, right = visit(node.left), visit(node.right)
            result = _ALLOWED_BINOPS[type(node.op)](left, right)
            if abs(result) > 10**12:
                raise ValueError("Result exceeds the calculator limit")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[type(node.op)](visit(node.operand))
        raise ValueError("Only basic arithmetic expressions are allowed")

    return visit(tree)


def calculator_handler(data: CalculatorInput) -> float | int:
    return _safe_calculate(data.expression)


def search_handler(data: SearchInput) -> list[str]:
    """Search a tiny built-in knowledge base; no external network access."""
    knowledge = [
        "An agent runtime plans tasks, selects approved tools, observes results, and adapts.",
        "Retrieval-augmented generation combines retrieved evidence with language generation.",
        "Pydantic validates structured data at application boundaries.",
        "FastAPI provides typed HTTP endpoints using Python type hints.",
        "Observability uses traces, metrics, and logs to explain system behavior.",
    ]
    terms = set(re.findall(r"[a-z0-9]+", data.query.lower()))
    ranked = sorted(
        knowledge,
        key=lambda item: len(terms & set(re.findall(r"[a-z0-9]+", item.lower()))),
        reverse=True,
    )
    return [
        item
        for item in ranked
        if terms & set(re.findall(r"[a-z0-9]+", item.lower()))
    ][:3]


def time_handler(_: BaseModel) -> str:
    return datetime.now(UTC).isoformat()


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            "calculator",
            "Safely evaluate basic arithmetic.",
            CalculatorInput,
            calculator_handler,
        )
    )
    registry.register(
        Tool(
            "knowledge_search",
            "Search the platform's built-in technical notes.",
            SearchInput,
            search_handler,
        )
    )
    registry.register(
        Tool(
            "utc_time",
            "Return the current UTC timestamp.",
            BaseModel,
            time_handler,
        )
    )
    return registry
