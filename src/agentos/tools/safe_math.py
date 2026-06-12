"""Safe arithmetic expression evaluator using the ast module.

Replaces ``eval()`` for calculator tools — only numeric literals and basic
math operators (+, -, *, /, **, unary +/-) are permitted.
"""

from __future__ import annotations

import ast
import operator
import re

_BINARY_OPS: dict[type, operator] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_UNARY_OPS: dict[type, operator] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval_math(expression: str) -> float | int:
    """Evaluate a math expression safely.

    Only supports numeric literals and the operators
    ``+  -  *  /  **  %  //`` with parentheses for grouping.

    Raises ``ValueError`` for anything else.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {exc}") from None
    return _eval_node(tree.body)


def _eval_node(node: ast.expr) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _BINARY_OPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))

    raise ValueError(
        f"Unsupported expression node: {ast.dump(node)}"
    )


_PERCENT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_EXPR_RE = re.compile(r"([\d.]+(?:\s*[\+\-\*/]\s*[\d.]+)+)")


def extract_calculator_expression(text: str) -> str:
    """Pull a safe math expression from natural-language calculator queries."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty message")

    percent_match = _PERCENT_RE.search(text)
    if percent_match:
        pct = float(percent_match.group(1))
        base = percent_match.group(2)
        return f"{base} * {pct / 100}"

    cleaned = re.sub(
        r"^(?:what\s+is|what's|calculate|compute|solve|find|evaluate)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip().rstrip("?.")

    for candidate in (cleaned, text):
        candidate = candidate.strip()
        if not candidate:
            continue
        for expr in (candidate, re.sub(r"\s+", "", candidate)):
            try:
                safe_eval_math(expr)
                return expr
            except ValueError:
                continue

    expr_match = _EXPR_RE.search(text)
    if expr_match:
        for expr in (expr_match.group(1), re.sub(r"\s+", "", expr_match.group(1))):
            try:
                safe_eval_math(expr)
                return expr
            except ValueError:
                continue

    raise ValueError(f"no calculable expression in: {text!r}")
