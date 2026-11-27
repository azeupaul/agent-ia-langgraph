"""Tools the model can ask the agent to run."""

import ast
import math
import operator
from collections.abc import Callable
from datetime import datetime
from typing import ClassVar
from zoneinfo import ZoneInfo

from langchain_core.tools import BaseTool, tool


class _Calculator:
    """Safe evaluator of arithmetic expressions.

    The expression is parsed into a syntax tree and never executed: only
    numbers and arithmetic operators are evaluated, anything else is rejected.
    """

    _MAX_DIGITS: ClassVar[int] = 1000
    """The most digits a result may have, so that a request cannot make the
    agent compute for minutes."""

    _BINARY_OPERATORS: ClassVar[
        dict[type[ast.operator], Callable[[float, float], float]]
    ] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    _UNARY_OPERATORS: ClassVar[dict[type[ast.unaryop], Callable[[float], float]]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def evaluate(self, expression: str) -> float:
        """Evaluate an arithmetic expression.

        Args:
            expression: Arithmetic expression in Python syntax, such as "2 + 3 * 4".

        Returns:
            The result of the expression.

        Raises:
            SyntaxError: If the expression cannot be parsed.
            ValueError: If the expression contains anything but numbers and
                arithmetic operators, or if its result is too large or not a
                finite number.
        """
        tree = ast.parse(expression, mode="eval")

        result = self._evaluate(tree.body)

        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("Result is not a finite number")

        if isinstance(result, int) and abs(result) >= 10**self._MAX_DIGITS:
            raise ValueError("Result too large")

        return result

    def _evaluate(self, node: ast.expr) -> float:
        """Evaluate a node of the syntax tree, and its children recursively."""
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return node.value

        if isinstance(node, ast.UnaryOp) and type(node.op) in self._UNARY_OPERATORS:
            unary_op = self._UNARY_OPERATORS[type(node.op)]
            return unary_op(self._evaluate(node.operand))

        if isinstance(node, ast.BinOp) and type(node.op) in self._BINARY_OPERATORS:
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            if isinstance(node.op, ast.Pow):
                self._check_power(left, right)

            binary_op = self._BINARY_OPERATORS[type(node.op)]
            return binary_op(left, right)

        raise ValueError(f"Unsupported expression: {ast.unparse(node)!r}")

    def _check_power(self, base: float, exponent: float) -> None:
        """Check that a power can be computed safely, before computing it.

        Args:
            base: The base of the power.
            exponent: The exponent of the power.

        Raises:
            ValueError: If the base is negative and the exponent fractional,
                since the result would be a complex number, or if the result
                would have more than ``_MAX_DIGITS`` digits.
        """
        if base < 0 and not (isinstance(exponent, int) or exponent.is_integer()):
            raise ValueError("Negative base with a fractional exponent")

        # The result has about |exponent| * |log10(|base|)| digits. Comparing
        # |exponent| to _MAX_DIGITS / |log10(|base|)| avoids converting a huge
        # integer exponent to float, which would overflow.
        digits_per_unit = abs(math.log10(abs(base))) if base != 0 else 0.0
        if digits_per_unit and abs(exponent) > self._MAX_DIGITS / digits_per_unit:
            raise ValueError("Result too large")


_CALCULATOR = _Calculator()


@tool(parse_docstring=True)
def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression and return the exact result.

    Always use it instead of calculating in your head, especially with large
    or decimal numbers. Supported: numbers, + - * / // % ** and parentheses.
    Not supported: functions (sqrt, round...), variables, dates or times. Use
    x ** 0.5 for a square root, and convert times to minutes or seconds before
    calculating.

    Args:
        expression: Arithmetic expression in Python syntax, e.g.
            "(24 * 60 - 935) / 60". Use ** for powers, not ^.

    Returns:
        The result of the expression.
    """
    return _CALCULATOR.evaluate(expression)


@tool(parse_docstring=True)
def get_current_time(timezone: str) -> str:
    """Get the current date and time in a given timezone.

    Use it whenever the user asks for the time or date somewhere, since you
    cannot know the current time otherwise.

    Args:
        timezone: IANA timezone name, e.g. "Africa/Douala". Use the timezone
            of the main city if the user gives a country.

    Returns:
        The current date and time in ISO 8601 format, with the UTC offset.
    """
    return datetime.now(ZoneInfo(timezone)).isoformat()


TOOLS: tuple[BaseTool, ...] = (
    get_current_time,
    calculate,
)
"""The tools the agent gives the model, in the order the model sees them."""
