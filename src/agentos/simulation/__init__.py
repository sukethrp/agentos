"""AgentOS Simulation — stress-test agents with realistic workloads.

Generate simulated customers with varied personas (happy, angry, confused,
edge-case) and traffic patterns (steady, burst, ramp-up), run them
concurrently, evaluate every interaction, and produce a comprehensive report.
"""

from agentos.simulation.personas import (
    ALL_PERSONAS,
    PERSONA_MAP,
    Mood,
    Persona,
    get_persona,
    get_random_persona,
    get_weighted_personas,
)
from agentos.simulation.traffic import (
    TrafficConfig,
    TrafficPattern,
    describe_pattern,
    generate_traffic,
)
from agentos.simulation.evaluator import Evaluator, InteractionResult
from agentos.simulation.report import (
    PersonaStats,
    SimulationReport,
    build_report,
)
from agentos.simulation.world import SimulatedWorld, WorldConfig

__all__ = [
    # Personas
    "ALL_PERSONAS",
    "PERSONA_MAP",
    "Mood",
    "Persona",
    "get_persona",
    "get_random_persona",
    "get_weighted_personas",
    # Traffic
    "TrafficConfig",
    "TrafficPattern",
    "describe_pattern",
    "generate_traffic",
    # Evaluation
    "Evaluator",
    "InteractionResult",
    # Report
    "PersonaStats",
    "SimulationReport",
    "build_report",
    # World
    "SimulatedWorld",
    "WorldConfig",
]
