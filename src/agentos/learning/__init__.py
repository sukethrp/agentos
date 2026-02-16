"""AgentOS Learning — improve agents from user feedback.

Collect thumbs-up/down, star ratings, and text corrections, then
automatically analyse patterns, optimise system prompts, and build
few-shot examples from the best interactions.  No fine-tuning needed.
"""

from agentos.learning.feedback import (
    FeedbackEntry,
    FeedbackStore,
    FeedbackType,
    get_feedback_store,
)
from agentos.learning.analyzer import (
    AnalysisReport,
    FeedbackAnalyzer,
    TopicAnalysis,
    ToolAnalysis,
    detect_topic,
)
from agentos.learning.prompt_optimizer import (
    PromptOptimizer,
    PromptPatch,
)
from agentos.learning.few_shot import (
    FewShotBuilder,
    FewShotExample,
)
from agentos.learning.report import (
    LearningReport,
    build_learning_report,
)

__all__ = [
    # Feedback
    "FeedbackEntry",
    "FeedbackStore",
    "FeedbackType",
    "get_feedback_store",
    # Analysis
    "AnalysisReport",
    "FeedbackAnalyzer",
    "TopicAnalysis",
    "ToolAnalysis",
    "detect_topic",
    # Prompt optimisation
    "PromptOptimizer",
    "PromptPatch",
    # Few-shot
    "FewShotBuilder",
    "FewShotExample",
    # Report
    "LearningReport",
    "build_learning_report",
]
