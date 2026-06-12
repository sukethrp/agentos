#!/usr/bin/env python3
"""
Self-Optimization Demo — evaluate config variants with no API keys.

Runs the SelfOptimizer over a small bundled task set using a stub runner
that simulates prompt quality via lexical overlap scoring.

Run:
    python examples/optimizer_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentos.optimize.optimizer import EvalExample, SelfOptimizer

DIVIDER = "=" * 60

BUNDLED_TASKS = [
    EvalExample(
        input="How do I get a refund?",
        expected="Acknowledge the request, explain eligibility, and give a 5-7 day timeline.",
    ),
    EvalExample(
        input="My invoice looks wrong.",
        expected="Ask for the invoice ID, verify line items, and offer a correction path.",
    ),
    EvalExample(
        input="App keeps crashing on login.",
        expected="Collect OS version, steps to reproduce, and suggest clearing cache first.",
    ),
    EvalExample(
        input="Can I upgrade my plan?",
        expected="Compare plans, mention proration, and link to billing settings.",
    ),
    EvalExample(
        input="Where is my order?",
        expected="Request order ID, check shipment status, and provide an ETA window.",
    ),
]


RESPONSES: dict[str, dict[str, str]] = {
    "incumbent": {
        "How do I get a refund?": "Please contact support for help.",
        "My invoice looks wrong.": "Billing can help with invoices.",
        "App keeps crashing on login.": "Try restarting the app.",
        "Can I upgrade my plan?": "Visit the website for plans.",
        "Where is my order?": "Check your email for tracking.",
    },
    "structured": {
        "How do I get a refund?": (
            "Acknowledge the request, explain eligibility, and give a 5-7 day timeline."
        ),
        "My invoice looks wrong.": (
            "Ask for the invoice ID, verify line items, and offer a correction path."
        ),
        "App keeps crashing on login.": (
            "Collect OS version, steps to reproduce, and suggest clearing cache first."
        ),
        "Can I upgrade my plan?": (
            "Compare plans, mention proration, and link to billing settings."
        ),
        "Where is my order?": (
            "Request order ID, check shipment status, and provide an ETA window."
        ),
    },
}


def stub_run(config: dict, example: EvalExample) -> str:
    return RESPONSES[config["name"]].get(example.input, "I am not sure.")


def pp(label: str, obj: dict | str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {label}")
    print(DIVIDER)
    if isinstance(obj, dict):
        print(textwrap.indent(json.dumps(obj, indent=2, default=str), "  "))
    else:
        print(textwrap.indent(str(obj), "  "))


def main() -> None:
    print("AgentOS Self-Optimization Demo")
    print(DIVIDER)

    optimizer = SelfOptimizer(alpha=0.05, effect_floor=0.15)
    result = optimizer.optimize(
        eval_set=BUNDLED_TASKS * 4,
        incumbent_config={"name": "incumbent", "prompt": "Be brief."},
        incumbent_name="incumbent",
        candidates=[("structured", {"name": "structured", "prompt": "Use a 3-step format."})],
        run_fn=stub_run,
    )

    pp(
        "Variant means",
        {
            name: round(stats.mean_score, 4)
            for name, stats in result.variant_stats.items()
        },
    )
    pp("Decision", result.decision_rationale)
    if result.decisions:
        d = result.decisions[0]
        pp(
            "Stats",
            {
                "welch_p": round(d.ab_result.welch_p_value, 6),
                "mann_whitney_p": round(d.ab_result.mann_whitney_p, 6),
                "cohens_d": round(d.ab_result.cohens_d, 4),
                "ci_incumbent": d.ab_result.ci_a,
                "ci_challenger": d.ab_result.ci_b,
                "adopted": d.adopted,
            },
        )
    print(f"\nChosen config: {result.chosen_name}")
    print("\nDemo complete (no API keys required).")


if __name__ == "__main__":
    main()
