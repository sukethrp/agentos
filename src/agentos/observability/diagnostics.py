"""Root Cause Diagnostics — pinpoint the EXACT step where things went wrong.

Given a :class:`Trace`, the diagnostics engine runs five checks:

1. **Context check** — Did the LLM receive the right system prompt and history?
2. **Tool selection check** — Did it choose an appropriate tool (or skip one it should have used)?
3. **Tool execution check** — Did the tool return valid data (or errors / empty results)?
4. **Interpretation check** — Did the LLM interpret the tool result correctly?
5. **Faithfulness check** — Is the final response faithful to the data it received?

Each check produces a :class:`CheckResult` (pass / warn / fail) with a
human-readable explanation.  The combined output is a :class:`Diagnosis`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agentos.observability.tracer import Trace, TraceStep, StepType


class Severity(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class CheckResult:
    """Outcome of a single diagnostic check."""

    check_name: str
    severity: Severity = Severity.PASS
    title: str = ""
    explanation: str = ""
    step_index: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check": self.check_name,
            "severity": self.severity.value,
            "title": self.title,
            "explanation": self.explanation,
            "step_index": self.step_index,
            "evidence": self.evidence,
        }


@dataclass
class Diagnosis:
    """Full root-cause analysis for a single trace."""

    trace_id: str = ""
    agent_name: str = ""
    user_query: str = ""
    root_cause: str = ""              # one-line verdict
    root_cause_step: int | None = None  # step index of the failure point
    overall_severity: Severity = Severity.PASS
    checks: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "user_query": self.user_query[:120],
            "root_cause": self.root_cause,
            "root_cause_step": self.root_cause_step,
            "overall_severity": self.overall_severity.value,
            "checks": [c.to_dict() for c in self.checks],
        }

    def summary_text(self) -> str:
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
        lines = [
            "=" * 60,
            f"  DIAGNOSIS — Trace {self.trace_id}",
            "=" * 60,
            f"  Agent: {self.agent_name}",
            f"  Query: {self.user_query[:80]}",
            f"  Verdict: {icon.get(self.overall_severity.value, '?')} {self.root_cause}",
        ]
        if self.root_cause_step is not None:
            lines.append(f"  Failure at step: {self.root_cause_step}")
        lines.append("")
        for c in self.checks:
            sev = icon.get(c.severity.value, "?")
            lines.append(f"  {sev} {c.check_name}: {c.title}")
            if c.explanation:
                lines.append(f"     {c.explanation}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Individual checks ────────────────────────────────────────────────────────

def _check_context(trace: Trace) -> CheckResult:
    """1. Did the LLM receive the right context?"""
    cr = CheckResult(check_name="context_quality")

    if not trace.system_prompt:
        cr.severity = Severity.WARN
        cr.title = "Empty system prompt"
        cr.explanation = "The agent ran without a system prompt — this often degrades quality."
        cr.evidence = {"system_prompt_length": 0}
        return cr

    # Check the first LLM call's message snapshot
    llm_steps = [s for s in trace.steps if s.step_type == StepType.LLM_CALL]
    if not llm_steps:
        cr.severity = Severity.FAIL
        cr.title = "No LLM calls found"
        cr.explanation = "The trace has no LLM call steps — the agent may not have run."
        return cr

    first = llm_steps[0]
    msg_count = len(first.messages_snapshot)
    has_system = any(m.get("role") == "system" for m in first.messages_snapshot)
    has_user = any(m.get("role") == "user" for m in first.messages_snapshot)

    issues = []
    if not has_system:
        issues.append("system prompt was missing from the message list")
    if not has_user:
        issues.append("user query was missing from the message list")
    if msg_count < 2:
        issues.append(f"only {msg_count} message(s) — may lack context")

    prompt_len = len(trace.system_prompt)
    if prompt_len < 20:
        issues.append(f"system prompt is very short ({prompt_len} chars)")

    if issues:
        cr.severity = Severity.WARN
        cr.title = "Context issues detected"
        cr.explanation = "; ".join(issues)
        cr.step_index = first.step_index
    else:
        cr.title = "Context looks good"
        cr.explanation = f"System prompt ({prompt_len} chars), {msg_count} messages."

    cr.evidence = {
        "system_prompt_length": prompt_len,
        "message_count": msg_count,
        "has_system": has_system,
        "has_user": has_user,
    }
    return cr


def _check_tool_selection(trace: Trace) -> CheckResult:
    """2. Did the LLM choose the right tool?"""
    cr = CheckResult(check_name="tool_selection")

    tool_steps = [s for s in trace.steps if s.step_type == StepType.TOOL_CALL]
    if not tool_steps:
        # No tools called — might be fine for a direct answer
        llm_steps = [s for s in trace.steps if s.step_type == StepType.LLM_CALL]
        if llm_steps and llm_steps[0].available_tools:
            cr.severity = Severity.WARN
            cr.title = "No tools called despite tools being available"
            cr.explanation = (
                f"Tools available: {', '.join(llm_steps[0].available_tools)}. "
                "The LLM answered directly — this may be correct or it may have "
                "missed an opportunity to use a tool for better accuracy."
            )
        else:
            cr.title = "No tools available or needed"
        return cr

    # Check for not-found tools
    missing = [s for s in tool_steps if s.tool_not_found]
    if missing:
        cr.severity = Severity.FAIL
        cr.title = f"Tool not found: {missing[0].tool_name}"
        cr.explanation = (
            f"The LLM tried to call '{missing[0].tool_name}' but it doesn't exist. "
            "This usually means the tool list is out of date or the LLM hallucinated a tool name."
        )
        cr.step_index = missing[0].step_index
        cr.evidence = {"missing_tools": [s.tool_name for s in missing]}
        return cr

    cr.title = f"Selected {len(tool_steps)} tool(s)"
    cr.explanation = "Tools called: " + ", ".join(s.tool_name for s in tool_steps)
    cr.evidence = {"tools_called": [s.tool_name for s in tool_steps]}
    return cr


def _check_tool_execution(trace: Trace) -> CheckResult:
    """3. Did the tool return correct data?"""
    cr = CheckResult(check_name="tool_execution")

    tool_steps = [s for s in trace.steps if s.step_type == StepType.TOOL_CALL]
    if not tool_steps:
        cr.title = "No tool calls to check"
        return cr

    error_steps = []
    empty_steps = []
    for s in tool_steps:
        result = s.tool_result.strip()
        if s.is_error or result.upper().startswith("ERROR"):
            error_steps.append(s)
        elif not result or result in ("null", "None", "{}"):
            empty_steps.append(s)

    if error_steps:
        worst = error_steps[0]
        cr.severity = Severity.FAIL
        cr.title = f"Tool '{worst.tool_name}' returned an error"
        cr.explanation = f"Error: {worst.tool_result[:200]}"
        cr.step_index = worst.step_index
        cr.evidence = {
            "errored_tools": [
                {"tool": s.tool_name, "error": s.tool_result[:120], "step": s.step_index}
                for s in error_steps
            ]
        }
        return cr

    if empty_steps:
        worst = empty_steps[0]
        cr.severity = Severity.WARN
        cr.title = f"Tool '{worst.tool_name}' returned empty result"
        cr.explanation = "An empty tool result may cause the LLM to hallucinate data."
        cr.step_index = worst.step_index
        return cr

    cr.title = "All tools returned data"
    cr.evidence = {
        "tool_results": [
            {"tool": s.tool_name, "result_length": len(s.tool_result)}
            for s in tool_steps
        ]
    }
    return cr


def _check_interpretation(trace: Trace) -> CheckResult:
    """4. Did the LLM interpret the tool result correctly?"""
    cr = CheckResult(check_name="interpretation")

    tool_steps = [s for s in trace.steps if s.step_type == StepType.TOOL_CALL and s.tool_result]
    final_steps = [s for s in trace.steps if s.step_type == StepType.FINAL_ANSWER]
    if not tool_steps or not final_steps:
        cr.title = "No tool+answer pair to check"
        return cr

    final_text = final_steps[-1].response_text.lower()

    # Heuristic: if tool returned specific data (numbers, names) check if
    # any of it appears in the final answer
    data_fragments: list[str] = []
    for s in tool_steps:
        # Extract numbers and capitalized words from tool results as "facts"
        numbers = re.findall(r"\d+\.?\d*", s.tool_result)
        data_fragments.extend(numbers[:5])
        caps = re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", s.tool_result)
        data_fragments.extend(w.lower() for w in caps[:5])

    if not data_fragments:
        cr.title = "No structured data to verify interpretation"
        return cr

    found = sum(1 for frag in data_fragments if frag.lower() in final_text)
    total = len(data_fragments)
    ratio = found / max(total, 1)

    if ratio < 0.1 and total >= 3:
        cr.severity = Severity.WARN
        cr.title = "Final answer may ignore tool data"
        cr.explanation = (
            f"Only {found}/{total} data fragments from tool results appear in "
            "the final answer. The LLM may have ignored or misinterpreted the data."
        )
        cr.evidence = {"found": found, "total": total, "ratio": round(ratio, 2)}
    else:
        cr.title = "Tool data reflected in answer"
        cr.explanation = f"{found}/{total} data fragments carried through."
        cr.evidence = {"found": found, "total": total, "ratio": round(ratio, 2)}

    return cr


def _check_faithfulness(trace: Trace) -> CheckResult:
    """5. Is the final response faithful to the data?"""
    cr = CheckResult(check_name="faithfulness")

    final_steps = [s for s in trace.steps if s.step_type == StepType.FINAL_ANSWER]
    if not final_steps:
        if trace.error:
            cr.severity = Severity.FAIL
            cr.title = "No final answer produced"
            cr.explanation = f"Agent errored: {trace.error[:200]}"
        else:
            cr.severity = Severity.WARN
            cr.title = "No final answer step found"
        return cr

    response = final_steps[-1].response_text

    # Check for obvious hallucination markers
    hallucination_markers = [
        "I don't have access",
        "I cannot",
        "As an AI",
        "I'm not sure",
        "I apologize, but",
    ]
    hedging_count = sum(1 for m in hallucination_markers if m.lower() in response.lower())

    # Check response quality
    if not response.strip():
        cr.severity = Severity.FAIL
        cr.title = "Empty response"
        cr.explanation = "The agent produced an empty final answer."
        return cr

    if len(response.strip()) < 10:
        cr.severity = Severity.WARN
        cr.title = "Very short response"
        cr.explanation = f"Response is only {len(response.strip())} characters."
        return cr

    if response.strip() == "Could not complete the task.":
        cr.severity = Severity.FAIL
        cr.title = "Agent hit max iterations"
        cr.explanation = "The agent exhausted its iteration budget without producing a useful answer."
        return cr

    if hedging_count >= 2:
        cr.severity = Severity.WARN
        cr.title = "Response contains hedging language"
        cr.explanation = (
            f"Found {hedging_count} hedging phrases. The agent may be uncertain "
            "or refusing to use available data."
        )
        cr.evidence = {"hedging_count": hedging_count}
    else:
        cr.title = "Response appears coherent"
        cr.explanation = f"Length: {len(response)} chars, no major red flags."

    cr.evidence["response_length"] = len(response)
    return cr


# ── Main engine ──────────────────────────────────────────────────────────────

_CHECKS = [
    _check_context,
    _check_tool_selection,
    _check_tool_execution,
    _check_interpretation,
    _check_faithfulness,
]


def diagnose(trace: Trace) -> Diagnosis:
    """Run all diagnostic checks on a trace and produce a root-cause report."""
    diag = Diagnosis(
        trace_id=trace.trace_id,
        agent_name=trace.agent_name,
        user_query=trace.user_query,
    )

    for check_fn in _CHECKS:
        result = check_fn(trace)
        diag.checks.append(result)

    # Determine overall severity and root cause
    fails = [c for c in diag.checks if c.severity == Severity.FAIL]
    warns = [c for c in diag.checks if c.severity == Severity.WARN]

    if fails:
        diag.overall_severity = Severity.FAIL
        first_fail = fails[0]
        diag.root_cause = first_fail.title
        diag.root_cause_step = first_fail.step_index
    elif warns:
        diag.overall_severity = Severity.WARN
        diag.root_cause = f"{len(warns)} warning(s): {warns[0].title}"
        diag.root_cause_step = warns[0].step_index
    else:
        diag.root_cause = "All checks passed — interaction looks healthy."

    return diag


def diagnose_batch(traces: list[Trace]) -> list[Diagnosis]:
    """Diagnose multiple traces and return sorted by severity (worst first)."""
    results = [diagnose(t) for t in traces]
    order = {Severity.FAIL: 0, Severity.WARN: 1, Severity.PASS: 2}
    return sorted(results, key=lambda d: order.get(d.overall_severity, 3))
