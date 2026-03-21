"""Safe arithmetic expression evaluator using the ast module.

Replaces ``eval()`` for calculator tools — only numeric literals and basic
math operators (+, -, *, /, **, unary +/-) are permitted.
"""

from __future__ import annotations

import ast
import operator

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
