import pytest

from agentic_platform.tools import _safe_calculate, build_registry


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("19 * 23", 437), ("(10 + 2) / 3", 4), ("-5 + 8", 3), ("2 ** 3", 8)],
)
def test_safe_calculator(expression, expected):
    assert _safe_calculate(expression) == expected


@pytest.mark.parametrize("expression", ["__import__('os').system('whoami')", "2 ** 99", "1 < 2"])
def test_calculator_rejects_unsafe_or_unsupported_expressions(expression):
    with pytest.raises((ValueError, TypeError)):
        _safe_calculate(expression)


def test_registry_executes_registered_tool():
    result = build_registry().execute("calculator", {"expression": "6 * 7"})
    assert result.success is True
    assert result.output == 42


def test_registry_rejects_unknown_tool():
    result = build_registry().execute("shell", {"command": "whoami"})
    assert result.success is False
    assert result.error == "Unknown or disallowed tool"


def test_registry_validates_arguments():
    result = build_registry().execute("calculator", {"unexpected": "1 + 1"})
    assert result.success is False
