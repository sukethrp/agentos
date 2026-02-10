"""GoverndAgent — The complete AgentOS experience in one class.

Combines: Agent + Governance + Monitoring + Sandbox testing.

Usage:
    from agentos.governed_agent import GovernedAgent
    from agentos.governance import BudgetGuard, PermissionGuard
    from agentos.sandbox import Scenario, Sandbox
    from agentos.core.tool import tool

    @tool(description="Calculator")
    def calc(expression: str) -> str:
        return str(eval(expression))

    agent = GovernedAgent(
        name="my-agent",
        model="gpt-4o-mini",
        tools=[calc],
        budget=BudgetGuard(max_per_day=5.00),
        permissions=PermissionGuard(blocked_tools=["dangerous_tool"]),
    )

    # Run with full governance
    agent.run("What's 2 + 2?")

    # Test before deploying
    report = agent.test([Scenario(...), Scenario(...)])

    # Check governance status
    agent.status()

    # Emergency stop
    agent.kill("Reason here")
"""

from __future__ import annotations
from agentos.core.agent import Agent
from agentos.core.tool import Tool
from agentos.core.types import AgentEvent, Message, Role
from agentos.governance.budget import BudgetGuard
from agentos.governance.permissions import PermissionGuard
from agentos.governance.guardrails import GovernanceEngine
from agentos.governance.audit import AuditLog
from agentos.monitor.store import store
from agentos.sandbox.scenario import Scenario
from agentos.sandbox.runner import Sandbox


class GovernedAgent:
    """A fully governed, monitored, and testable AI agent.

    This is the main class most users should use. It wraps:
    - Agent (thinking + tool calling)
    - GovernanceEngine (budget + permissions + kill switch)
    - MonitorStore (event tracking)
    - Sandbox (testing)
    """

    def __init__(
        self,
        name: str = "agent",
        model: str = "gpt-4o-mini",
        tools: list[Tool] | None = None,
        system_prompt: str = "You are a helpful assistant. Use tools when needed.",
        budget: BudgetGuard | None = None,
        permissions: PermissionGuard | None = None,
        max_iterations: int = 10,
        temperature: float = 0.7,
        enable_monitoring: bool = True,
    ):
        # Core agent
        self.agent = Agent(
            name=name,
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            temperature=temperature,
        )

        # Governance
        self.governance = GovernanceEngine(
            agent_name=name,
            budget=budget,
            permissions=permissions,
        )

        # Monitoring
        self.enable_monitoring = enable_monitoring
        self.run_count = 0
        self.total_cost = 0.0

        # Wrap tool execution with governance
        self._wrap_tools()

    def _wrap_tools(self):
        """Add governance checks to every tool."""
        for t in self.agent.tools:
            original = t.execute

            def make_governed(tool_obj, orig):
                def governed_execute(call):
                    check = self.governance.check_tool_call(tool_obj.name, estimated_cost=0.001)
                    if not check.allowed:
                        print(f"   🚫 BLOCKED: {check.message}")
                        from agentos.core.types import ToolResult
                        return ToolResult(
                            tool_call_id=call.id,
                            name=tool_obj.name,
                            result=f"BLOCKED: {check.message}",
                        )
                    result = orig(call)
                    self.governance.record_action(tool_obj.name, cost=0.001)
                    return result
                return governed_execute

            t.execute = make_governed(t, original)

    def run(self, user_input: str) -> Message:
        """Run the agent with full governance and monitoring."""
        self.run_count += 1
        self.governance.permissions.reset()

        result = self.agent.run(user_input)

        # Log events to monitor store
        if self.enable_monitoring:
            for event in self.agent.events:
                store.log_event(event)

        # Track costs
        run_cost = sum(e.cost_usd for e in self.agent.events)
        self.total_cost += run_cost

        return result

    def test(self, scenarios: list[Scenario], pass_threshold: float = 6.0):
        """Test the agent against scenarios using the Simulation Sandbox."""
        sandbox = Sandbox(self.agent, pass_threshold=pass_threshold)
        return sandbox.run(scenarios)

    def kill(self, reason: str = "Manual kill switch"):
        """Emergency stop the agent."""
        self.governance.kill(reason)

    def revive(self):
        """Re-enable the agent after kill switch."""
        self.governance.revive()

    def status(self):
        """Print full governance status."""
        self.governance.print_status()

    def audit(self):
        """Print the audit trail."""
        self.governance.audit.print_report()

    def get_stats(self) -> dict:
        """Get agent statistics."""
        return {
            "name": self.agent.config.name,
            "model": self.agent.config.model,
            "tools": [t.name for t in self.agent.tools],
            "total_runs": self.run_count,
            "total_cost": round(self.total_cost, 6),
            "governance": self.governance.get_status(),
        }