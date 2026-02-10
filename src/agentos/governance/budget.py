from __future__ import annotations
import time


class BudgetGuard:
    """Controls how much an agent can spend.

    Usage:
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
    ):
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

    def _reset_windows(self):
        now = time.time()
        if now - self.last_hour_reset > 3600:
            self.hourly_spent = 0.0
            self.last_hour_reset = now
        if now - self.last_day_reset > 86400:
            self.daily_spent = 0.0
            self.last_day_reset = now

    def check_action(self, cost: float) -> tuple[bool, str]:
        """Check if an action is within budget. Returns (allowed, message)."""
        self._reset_windows()
        self.action_count += 1

        if cost > self.max_per_action:
            self.blocked_count += 1
            return False, f"Action cost ${cost:.4f} exceeds per-action limit ${self.max_per_action:.4f}"

        if self.hourly_spent + cost > self.max_per_hour:
            self.blocked_count += 1
            return False, f"Hourly spend ${self.hourly_spent + cost:.4f} would exceed limit ${self.max_per_hour:.4f}"

        if self.daily_spent + cost > self.max_per_day:
            self.blocked_count += 1
            return False, f"Daily spend ${self.daily_spent + cost:.4f} would exceed limit ${self.max_per_day:.4f}"

        if self.total_spent + cost > self.max_total:
            self.blocked_count += 1
            return False, f"Total spend ${self.total_spent + cost:.4f} would exceed limit ${self.max_total:.4f}"

        return True, "OK"

    def record_spend(self, cost: float):
        """Record actual spending after an action completes."""
        self.total_spent += cost
        self.hourly_spent += cost
        self.daily_spent += cost

    def get_status(self) -> dict:
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