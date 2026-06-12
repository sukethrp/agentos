"""AgentOS self-optimization — orchestrate eval, A/B testing, and drift checks."""

from agentos.optimize.optimizer import (
    ChallengerDecision,
    EvalExample,
    OptimizationResult,
    RunOutput,
    SelfOptimizer,
    VariantStats,
)

__all__ = [
    "ChallengerDecision",
    "EvalExample",
    "OptimizationResult",
    "RunOutput",
    "SelfOptimizer",
    "VariantStats",
]
