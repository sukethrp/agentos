"""Budget enforcement for governed agents.

Defense in depth — multiple budget layers (per-action, hourly, daily, total)
so no single misconfigured limit can cause a runaway.  Even if the per-action
limit is set too high, the hourly and daily caps act as safety nets, and the
lifetime total provides an absolute ceiling.

Budget checks happen BEFORE the LLM call (fail fast).  This is intentional:
we don't want to waste tokens on a call that will be rejected once the
response comes back.  The cost estimate used for pre-checks is based on the
prompt token count; actual spend is recorded after the call completes via
``record_spend``.
"""

from __future__ import annotations
import time


class BudgetGuard:
    """Controls how much an agent can spend across multiple time windows.

    Each action is checked against four independent limits before execution:

    1. **Per-action** — rejects any single call whose estimated cost is too
       high, catching prompt-injection attacks or unexpectedly large inputs.
    2. **Per-hour** — prevents short bursts from draining the budget.
    3. **Per-day** — caps sustained usage over a work day.
    4. **Lifetime total** — absolute ceiling that survives restarts of the
       hourly/daily windows.

    The guard is intentionally stateful and *not* thread-safe; each agent
    instance should own its own ``BudgetGuard``.

    Usage::

        budget = BudgetGuard(max_per_action=0.10, max_per_day=5.00)
        ok, msg = budget.check_action(cost=0.05)
        if not ok:
            print(f"BLOCKED: {msg}")
    """

    def __init__(
        self,
        max_per_action: float = 1.00,
        max_per_hour: float = 10.00,
        max_per_day: float = 50.00,
        max_total: float = 500.00,
    ) -> None:
        self.max_per_action = max_per_action
        self.max_per_hour = max_per_hour
        self.max_per_day = max_per_day
        self.max_total = max_total

        self.total_spent = 0.0
        self.hourly_spent = 0.0
        self.daily_spent = 0.0
        self.last_hour_reset = time.time()
        self.last_day_reset = time.time()
        self.action_count = 0
        self.blocked_count = 0

    def _reset_windows(self) -> None:
        """Reset hourly/daily accumulators when their window has elapsed.

        Hourly budget resets on the clock hour, not a rolling 60-minute
        window, to keep the logic simple and predictable.  The same
        applies to the daily window (fixed 24 h from last reset).
        A rolling window would be more precise but requires storing every
        individual spend event, which isn't worth the complexity today.
        """
        now = time.time()
        if now - self.last_hour_reset > 3600:
            self.hourly_spent = 0.0
            self.last_hour_reset = now
        if now - self.last_day_reset > 86400:
            self.daily_spent = 0.0
            self.last_day_reset = now

    def check_action(self, cost: float) -> tuple[bool, str]:
        """Pre-flight budget check — call this BEFORE making the LLM request.

        Validates the estimated ``cost`` against every budget layer in order
        from most specific (per-action) to broadest (lifetime total).  This
        ordering means the most common rejection reason (a single expensive
        call) is caught first, keeping the fast path short.

        Returns:
            A ``(allowed, message)`` tuple.  ``allowed`` is ``True`` when all
            limits pass; ``message`` is ``"OK"`` on success or a human-readable
            rejection reason on failure.
        """
        # Reset time-based windows before checking so stale accumulators
        # don't cause false rejections.
        self._reset_windows()
        self.action_count += 1

        # Layer 1: per-action — catch obviously oversized requests immediately.
        if cost > self.max_per_action:
            self.blocked_count += 1
            return (
                False,
                f"Action cost ${cost:.4f} exceeds per-action limit ${self.max_per_action:.4f}",
            )

        # Layer 2: hourly — prevent short bursts from draining the budget.
        if self.hourly_spent + cost > self.max_per_hour:
            self.blocked_count += 1
            return (
                False,
                f"Hourly spend ${self.hourly_spent + cost:.4f} would exceed limit ${self.max_per_hour:.4f}",
            )

        # Layer 3: daily — cap sustained usage.
        if self.daily_spent + cost > self.max_per_day:
            self.blocked_count += 1
            return (
                False,
                f"Daily spend ${self.daily_spent + cost:.4f} would exceed limit ${self.max_per_day:.4f}",
            )

        # Layer 4: lifetime total — absolute ceiling.
        if self.total_spent + cost > self.max_total:
            self.blocked_count += 1
            return (
                False,
                f"Total spend ${self.total_spent + cost:.4f} would exceed limit ${self.max_total:.4f}",
            )

        return True, "OK"

    def record_spend(self, cost: float) -> None:
        """Record actual spending AFTER an action completes successfully.

        This is separate from ``check_action`` because the estimated cost
        (checked before the call) may differ from the actual cost (known
        only after the LLM response).  Keeping pre-check and recording as
        two steps avoids double-counting and lets callers skip recording
        if the call failed mid-stream.
        """
        self.total_spent += cost
        self.hourly_spent += cost
        self.daily_spent += cost

    def get_status(self) -> dict:
        """Return a snapshot of current budget utilization.

        Resets stale windows first so the returned numbers are always
        up-to-date, even if no actions have been checked recently.
        """
        self._reset_windows()
        return {
            "total_spent": round(self.total_spent, 6),
            "hourly_spent": round(self.hourly_spent, 6),
            "daily_spent": round(self.daily_spent, 6),
            "total_limit": self.max_total,
            "hourly_limit": self.max_per_hour,
            "daily_limit": self.max_per_day,
            "action_count": self.action_count,
            "blocked_count": self.blocked_count,
            "budget_remaining": round(self.max_total - self.total_spent, 6),
        }
