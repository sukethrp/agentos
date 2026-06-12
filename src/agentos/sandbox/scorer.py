from __future__ import annotations
import json
import os
from agentos.demo import is_demo_mode
from agentos.sandbox.evaluation_scenario import EvaluationScenario
from agentos.sandbox.metrics import bleu_score, lexical_overlap, rouge_l_score


JUDGE_PROMPT = """Score the agent's response against the expected output.

SCENARIO INPUT: {input}
EXPECTED OUTPUT: {expected_output}
RUBRIC: {rubric}

ACTUAL OUTPUT: {actual_output}

Respond with ONLY valid JSON: {{"score": <float 0-1>, "reasoning": "<string>"}}
"""


def _heuristic_score(scenario: EvaluationScenario, actual_output: str) -> float:
    actual = (actual_output or "").strip()
    if not actual:
        return 0.0

    expected = (scenario.expected_output or "").strip()
    if expected:
        bleu = bleu_score(expected, actual)
        rouge = rouge_l_score(expected, actual)
        overlap = lexical_overlap(expected, actual)
    else:
        bleu = 0.0
        rouge = rouge_l_score(scenario.input, actual)
        overlap = lexical_overlap(scenario.input, actual)

    return max(0.0, min(1.0, 0.25 * bleu + 0.25 * rouge + 0.50 * overlap))


class LLMJudgeScorer:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self._model = model
        self._client = None
        if not is_demo_mode() and os.getenv("ANTHROPIC_API_KEY"):
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic()

    async def score(self, scenario: EvaluationScenario, actual_output: str) -> float:
        if self._client is None:
            return _heuristic_score(scenario, actual_output)

        prompt = JUDGE_PROMPT.format(
            input=scenario.input,
            expected_output=scenario.expected_output,
            rubric=scenario.rubric,
            actual_output=actual_output or "",
        )
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            s = float(parsed.get("score", 0.0))
            return max(0.0, min(1.0, s))
        except Exception:
            return _heuristic_score(scenario, actual_output)
