from __future__ import annotations

import os
from unittest.mock import patch

from agentos.providers.mock import _extract_expression, call_mock
from agentos.tools.http_tool import calculator_tool
from agentos.tools.safe_math import safe_eval_math


def test_mock_calculator_uses_user_expression():
    assert _extract_expression("2+2") == "2+2"
    assert _extract_expression("What is 15% of 200?") == "200 * 15 / 100"

    calc = calculator_tool()
    messages = [{"role": "user", "content": "2+2"}]

    with patch.dict(os.environ, {"AGENTOS_DEMO_MODE": "true"}, clear=False):
        tool_msg, _ = call_mock(messages, [calc], agent_name="calc-agent")
        assert tool_msg.tool_calls is not None
        expression = tool_msg.tool_calls[0].arguments["expression"]
        result = calc.fn(**tool_msg.tool_calls[0].arguments)
        assert safe_eval_math(expression) == 4
        assert result == "4"

        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_msg.tool_calls[0].id,
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": f'{{"expression": "{expression}"}}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": result, "tool_call_id": tool_msg.tool_calls[0].id},
            ]
        )
        final_msg, _ = call_mock(messages, [calc], agent_name="calc-agent")
        assert final_msg.content is not None
        assert "4" in final_msg.content
        assert "12.825" not in final_msg.content
