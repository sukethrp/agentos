from __future__ import annotations

PHI_APPROVED_PROVIDERS = frozenset({"openai", "anthropic"})


class PolicyViolationError(Exception):
    pass


class PolicyEngine:
    def __init__(self, phi_approved_providers: frozenset[str] | None = None):
        self._phi_approved = phi_approved_providers or PHI_APPROVED_PROVIDERS

    def enforce(
        self,
        action: str,
        data_class: str,
        provider: str,
        tool_logged: bool = True,
    ) -> None:
        if data_class == "PHI" and provider.lower() not in self._phi_approved:
            raise PolicyViolationError(
                f"PHI data cannot be sent to provider '{provider}'. Approved: {sorted(self._phi_approved)}"
            )
        if action in ("tool_call", "tool_execute") and not tool_logged:
            raise PolicyViolationError(f"Tool call '{action}' must be logged before execution")
