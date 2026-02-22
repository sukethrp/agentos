from __future__ import annotations
import json
from anthropic import AsyncAnthropic
from agentos.sandbox.evaluation_scenario import EvaluationScenario


JUDGE_PROMPT = """Score the agent's response against the expected output.

SCENARIO INPUT: {input}
EXPECTED OUTPUT: {expected_output}
RUBRIC: {rubric}

ACTUAL OUTPUT: {actual_output}

Respond with ONLY valid JSON: {{"score": <float 0-1>, "reasoning": "<string>"}}
"""


class LLMJudgeScorer:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self._client = AsyncAnthropic()
        self._model = model

    async def score(self, scenario: EvaluationScenario, actual_output: str) -> float:
        prompt = JUDGE_PROMPT.format(
            input=scenario.input,
            expected_output=scenario.expected_output,
            rubric=scenario.rubric,
            actual_output=actual_output or "",
        )
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(text)
            s = float(parsed.get("score", 0.0))
            return max(0.0, min(1.0, s))
        except (json.JSONDecodeError, ValueError):
            return 0.0
