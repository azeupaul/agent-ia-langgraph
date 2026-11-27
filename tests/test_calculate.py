import pytest

from agent_ia_langgraph.tools import calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        # Operator precedence and parentheses
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        # Division operators
        ("7 / 2", 3.5),
        ("7 // 2", 3),
        ("7 % 3", 1),
        # Unary operators: ** binds tighter than the unary minus
        ("-2 ** 2", -4),
        ("(-2) ** 2", 4),
        ("--5", 5),
        # Powers
        ("16 ** 0.5", 4.0),
        ("2 ** -1", 0.5),
        ("(-2) ** 3", -8),
        # Floats
        ("0.1 + 0.2", pytest.approx(0.3)),
        ("(24 * 60 - 935) / 60", pytest.approx(8.4167, abs=1e-4)),
    ],
)
def test_calculate_evaluates_valid_expressions(
    expression: str, expected: float
) -> None:
    assert calculate.invoke({"expression": expression}) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "int(3.7)",  # a function call
        '__import__("os")',  # what eval would run
        "x + 1",  # a variable
        "True + 1",  # a bool, which Python treats as the int 1
        '"a" * 3',  # a string
        "2 < 3",  # a comparison
        "[1, 2]",  # a list
        "~5",  # a unary operator that is not arithmetic
        "1 << 3",  # a binary operator that is not arithmetic
    ],
)
def test_calculate_rejects_anything_but_arithmetic_on_numbers(
    expression: str,
) -> None:
    with pytest.raises(ValueError, match="Unsupported expression"):
        calculate.invoke({"expression": expression})


def test_calculate_accepts_the_largest_allowed_integer() -> None:
    # 10 ** 999 has 1000 digits, the most allowed.
    result = calculate.invoke({"expression": "10 ** 999"})

    assert result == 10**999


@pytest.mark.parametrize(
    "expression",
    [
        # The result itself is too large
        "10 ** 1000",
        # A power too large to compute at all: 9 ** 387420489
        "9 ** 9 ** 9",
        # Each power is allowed, but not their product
        "10 ** 600 * 10 ** 600",
    ],
)
def test_calculate_rejects_results_too_large(expression: str) -> None:
    # Without a limit, the model could make the agent compute for minutes.
    with pytest.raises(ValueError, match="Result too large"):
        calculate.invoke({"expression": expression})
