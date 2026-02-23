"""Prompt Optimizer — automatically improve system prompts from feedback.

Pure prompt-engineering optimization.  NO fine-tuning.

The optimizer:
1. Reads the analysis report (weak topics, common complaints, corrections)
2. Generates *prompt patches* — extra instructions to append to the system prompt
3. Optionally calls an LLM to synthesise a better prompt (or uses templates)
4. Tracks which patches are active so you can roll back

Example output patch:
    "When handling REFUND questions, always:
     1. Acknowledge the customer's frustration first
     2. Explain the refund policy clearly
     3. Provide a timeline for the refund"
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass

from agentos.learning.analyzer import AnalysisReport, FeedbackAnalyzer
from agentos.learning.feedback import FeedbackStore


# ── Prompt patch ─────────────────────────────────────────────────────────────


@dataclass
class PromptPatch:
    """A single optimisation patch to append to the system prompt."""

    id: str = ""
    topic: str = ""
    instruction: str = ""
    source: str = "auto"  # "auto" | "manual" | "llm"
    confidence: float = 0.0  # 0-1 — how sure we are this helps
    created_at: float = 0.0
    active: bool = True

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex[:10]
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "instruction": self.instruction,
            "source": self.source,
            "confidence": round(self.confidence, 2),
            "active": self.active,
        }


# ── Template-based patch generation ─────────────────────────────────────────

_TOPIC_TEMPLATES: dict[str, str] = {
    "refund": (
        "When handling REFUND or CANCELLATION requests:\n"
        "1. Acknowledge the customer's frustration empathetically\n"
        "2. Clearly explain the refund policy and eligibility\n"
        "3. Provide a concrete timeline (e.g. '5-7 business days')\n"
        "4. Offer alternatives if refund is not possible"
    ),
    "billing": (
        "When answering BILLING or PRICING questions:\n"
        "1. Be specific — mention exact prices, plan names, and limits\n"
        "2. Compare plans clearly if the user is deciding\n"
        "3. Mention any free trial or discount options\n"
        "4. Direct them to the billing page for changes"
    ),
    "technical": (
        "When troubleshooting TECHNICAL issues:\n"
        "1. Ask clarifying questions (error message, steps to reproduce)\n"
        "2. Provide step-by-step debugging instructions\n"
        "3. Link to relevant documentation\n"
        "4. Escalate to engineering if the issue persists"
    ),
    "account": (
        "When handling ACCOUNT or LOGIN issues:\n"
        "1. Never ask the user for their password\n"
        "2. Guide them through the password reset flow\n"
        "3. Check if there are known outages first\n"
        "4. Offer to verify account status securely"
    ),
    "onboarding": (
        "When helping with ONBOARDING or SETUP:\n"
        "1. Assume zero prior knowledge\n"
        "2. Use numbered step-by-step instructions\n"
        "3. Link to the quickstart guide and video tutorials\n"
        "4. Offer to walk them through it"
    ),
    "integration": (
        "When answering INTEGRATION or API questions:\n"
        "1. Provide code examples when possible\n"
        "2. Mention rate limits, authentication, and SDKs\n"
        "3. Link to the API reference documentation\n"
        "4. Suggest testing with the sandbox environment first"
    ),
    "performance": (
        "When addressing PERFORMANCE or SPEED concerns:\n"
        "1. Ask which specific action is slow\n"
        "2. Check the status page for known issues\n"
        "3. Suggest common fixes (caching, pagination, filters)\n"
        "4. Offer to open an engineering ticket if systemic"
    ),
    "security": (
        "When answering SECURITY or COMPLIANCE questions:\n"
        "1. Reference specific certifications (SOC2, GDPR, etc.)\n"
        "2. Explain data handling and encryption practices\n"
        "3. Link to the security whitepaper or trust center\n"
        "4. Offer to connect with the security team for details"
    ),
}

_TONE_PATCHES: dict[str, str] = {
    "angry": (
        "When the customer is FRUSTRATED or ANGRY:\n"
        "- Lead with empathy: 'I completely understand your frustration'\n"
        "- Never be defensive or dismissive\n"
        "- Offer a concrete resolution, not just an apology\n"
        "- Follow up proactively"
    ),
    "confused": (
        "When the customer seems CONFUSED or LOST:\n"
        "- Use simple, jargon-free language\n"
        "- Break instructions into small numbered steps\n"
        "- Ask 'Would you like me to explain that differently?'\n"
        "- Be patient and encouraging"
    ),
}


def _generate_template_patch(
    topic: str, analysis: AnalysisReport
) -> PromptPatch | None:
    """Generate a patch from the built-in templates."""
    template = _TOPIC_TEMPLATES.get(topic)
    if not template:
        return None

    topic_data = next((t for t in analysis.topics if t.topic == topic), None)
    confidence = 0.5
    if topic_data:
        confidence = min(0.9, topic_data.failure_rate / 100 + 0.3)

    return PromptPatch(
        topic=topic,
        instruction=template,
        source="auto",
        confidence=confidence,
    )


def _generate_correction_patch(corrections: list[dict]) -> PromptPatch | None:
    """Generate a patch from user corrections."""
    if not corrections:
        return None

    lines = ["Based on user corrections, remember these rules:"]
    for c in corrections[:5]:
        lines.append(
            f"- When asked about '{c['query'][:60]}', the correct approach is: {c['correction'][:120]}"
        )

    return PromptPatch(
        topic="corrections",
        instruction="\n".join(lines),
        source="corrections",
        confidence=0.8,
    )


def _generate_llm_patch(
    topic: str, complaints: list[str], corrections: list[dict]
) -> PromptPatch | None:
    """Use GPT-4o-mini to synthesise a prompt patch. Falls back to template."""
    try:
        from openai import OpenAI

        client = OpenAI()

        complaint_text = "\n".join(f"- {c}" for c in complaints[:8])
        correction_text = "\n".join(
            f"- Q: {c['query'][:60]}  Correct: {c['correction'][:80]}"
            for c in corrections[:5]
        )

        prompt = (
            f"You are a prompt-engineering expert.\n\n"
            f"An AI customer-support agent is failing on '{topic}' questions.\n\n"
            f"Customer complaints:\n{complaint_text}\n\n"
            f"Corrections (what the agent should have said):\n{correction_text}\n\n"
            f"Write 3-5 clear, actionable instructions to add to the agent's system "
            f"prompt so it handles '{topic}' questions better. Be specific. "
            f"Format as a numbered list."
        )

        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        text = (result.choices[0].message.content or "").strip()
        if text:
            return PromptPatch(
                topic=topic,
                instruction=text,
                source="llm",
                confidence=0.7,
            )
    except Exception:
        pass
    return None


# ── Optimizer ────────────────────────────────────────────────────────────────


class PromptOptimizer:
    """Generate and manage prompt patches based on feedback analysis."""

    def __init__(
        self,
        store: FeedbackStore | None = None,
        use_llm: bool = False,
    ) -> None:
        from agentos.learning.feedback import get_feedback_store

        self.store = store or get_feedback_store()
        self.analyzer = FeedbackAnalyzer(self.store)
        self.use_llm = use_llm and bool(os.getenv("OPENAI_API_KEY"))
        self.patches: list[PromptPatch] = []
        self._history: list[dict] = []

    def optimize(self) -> list[PromptPatch]:
        """Analyze feedback and generate prompt patches for weak areas."""
        report = self.analyzer.analyze()
        new_patches: list[PromptPatch] = []

        # Patch each weak topic
        for topic in report.worst_topics:
            topic_data = next((t for t in report.topics if t.topic == topic), None)
            complaints = topic_data.common_complaints if topic_data else []
            topic_corrections = [
                c for c in report.top_corrections if c.get("topic") == topic
            ]

            patch = None
            if self.use_llm and (complaints or topic_corrections):
                patch = _generate_llm_patch(topic, complaints, topic_corrections)
            if patch is None:
                patch = _generate_template_patch(topic, report)
            if patch:
                new_patches.append(patch)

        # Correction-based patch
        if report.top_corrections:
            corr_patch = _generate_correction_patch(report.top_corrections)
            if corr_patch:
                new_patches.append(corr_patch)

        # Tone patches for topics with lots of angry/confused feedback
        negative = [e for e in self.store.all() if not e.is_positive]
        for mood, template in _TONE_PATCHES.items():
            mood_hits = sum(
                1
                for e in negative
                if mood in e.query.lower() or mood in e.comment.lower()
            )
            if mood_hits >= 2:
                new_patches.append(
                    PromptPatch(
                        topic=f"tone_{mood}",
                        instruction=template,
                        source="auto",
                        confidence=min(0.8, mood_hits / len(negative) + 0.3)
                        if negative
                        else 0.5,
                    )
                )

        self.patches = new_patches
        self._history.append(
            {
                "timestamp": time.time(),
                "patches_generated": len(new_patches),
                "topics_covered": [p.topic for p in new_patches],
            }
        )
        return new_patches

    def apply_patches(self, base_prompt: str) -> str:
        """Return *base_prompt* augmented with all active patches."""
        active = [p for p in self.patches if p.active]
        if not active:
            return base_prompt

        sections = [base_prompt.rstrip(), "", "# Learned Optimisations"]
        for p in active:
            sections.append(f"\n## {p.topic.replace('_', ' ').title()}")
            sections.append(p.instruction)

        return "\n".join(sections)

    def deactivate_patch(self, patch_id: str) -> bool:
        for p in self.patches:
            if p.id == patch_id:
                p.active = False
                return True
        return False

    def get_patches(self) -> list[dict]:
        return [p.to_dict() for p in self.patches]

    def get_history(self) -> list[dict]:
        return list(self._history)
