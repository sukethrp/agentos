"""Traffic patterns — generate realistic request timing for simulation.

Three built-in patterns:

* **steady** — even spacing, like a constant 5 RPS
* **burst** — quiet periods punctuated by sharp spikes
* **ramp_up** — gradually increasing load (stress test)

Each generator yields ``(delay_seconds, persona_index)`` tuples so the
simulation world knows *when* to fire the next request and *which*
persona to use.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Generator


class TrafficPattern(str, Enum):
    STEADY = "steady"
    BURST = "burst"
    RAMP_UP = "ramp_up"
    RANDOM = "random"
    WAVE = "wave"


@dataclass
class TrafficConfig:
    """Tuneable knobs for the traffic generator."""

    pattern: TrafficPattern = TrafficPattern.STEADY
    total_requests: int = 50
    requests_per_second: float = 2.0  # baseline RPS
    burst_size: int = 10  # how many in a burst
    burst_pause: float = 3.0  # seconds of quiet between bursts
    ramp_duration_seconds: float = 30.0  # how long the ramp-up lasts
    jitter: float = 0.2  # ±20 % random jitter on delays


def _jitter(base: float, factor: float) -> float:
    """Add uniform jitter to a delay."""
    if factor <= 0 or base <= 0:
        return max(base, 0)
    low = base * (1 - factor)
    high = base * (1 + factor)
    return max(random.uniform(low, high), 0.001)


# ── Generators ───────────────────────────────────────────────────────────────


def steady(cfg: TrafficConfig) -> Generator[tuple[float, int], None, None]:
    """Constant rate with optional jitter."""
    interval = 1.0 / max(cfg.requests_per_second, 0.01)
    for i in range(cfg.total_requests):
        yield _jitter(interval, cfg.jitter), i
    return


def burst(cfg: TrafficConfig) -> Generator[tuple[float, int], None, None]:
    """Alternating bursts and pauses."""
    idx = 0
    while idx < cfg.total_requests:
        # fire a burst
        burst_count = min(cfg.burst_size, cfg.total_requests - idx)
        for j in range(burst_count):
            delay = _jitter(0.05, cfg.jitter)  # near-simultaneous
            yield delay, idx
            idx += 1
        # pause
        if idx < cfg.total_requests:
            yield _jitter(cfg.burst_pause, cfg.jitter), idx
    return


def ramp_up(cfg: TrafficConfig) -> Generator[tuple[float, int], None, None]:
    """Linearly increasing RPS over the ramp duration."""
    start_rps = max(cfg.requests_per_second * 0.1, 0.1)
    end_rps = cfg.requests_per_second
    for i in range(cfg.total_requests):
        progress = i / max(cfg.total_requests - 1, 1)
        current_rps = start_rps + (end_rps - start_rps) * progress
        interval = 1.0 / current_rps
        yield _jitter(interval, cfg.jitter), i
    return


def random_traffic(cfg: TrafficConfig) -> Generator[tuple[float, int], None, None]:
    """Purely random delays between a minimum and 2x the base interval."""
    base = 1.0 / max(cfg.requests_per_second, 0.01)
    for i in range(cfg.total_requests):
        yield random.uniform(0.01, base * 2), i
    return


def wave(cfg: TrafficConfig) -> Generator[tuple[float, int], None, None]:
    """Sinusoidal wave pattern — peaks and troughs."""
    base = 1.0 / max(cfg.requests_per_second, 0.01)
    for i in range(cfg.total_requests):
        progress = i / max(cfg.total_requests - 1, 1)
        # full sine cycle over the run
        factor = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(2 * math.pi * progress))
        yield _jitter(base * factor, cfg.jitter), i
    return


# ── Dispatcher ───────────────────────────────────────────────────────────────

_GENERATORS = {
    TrafficPattern.STEADY: steady,
    TrafficPattern.BURST: burst,
    TrafficPattern.RAMP_UP: ramp_up,
    TrafficPattern.RANDOM: random_traffic,
    TrafficPattern.WAVE: wave,
}


def generate_traffic(
    cfg: TrafficConfig | None = None,
) -> Generator[tuple[float, int], None, None]:
    """Return the traffic generator for the given config."""
    cfg = cfg or TrafficConfig()
    gen_fn = _GENERATORS.get(cfg.pattern, steady)
    yield from gen_fn(cfg)


def describe_pattern(cfg: TrafficConfig) -> str:
    """Human-readable one-liner."""
    return (
        f"{cfg.pattern.value} pattern — {cfg.total_requests} requests, "
        f"~{cfg.requests_per_second} RPS baseline, jitter ±{int(cfg.jitter * 100)}%"
    )
