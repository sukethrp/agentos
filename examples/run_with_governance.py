"""AgentOS Governance Demo — Budget, Permissions, Kill Switch, Audit."""

import sys
sys.path.insert(0, "src")

from agentos.core.tool import tool
from agentos.core.agent import Agent
from agentos.governance.budget import BudgetGuard
from agentos.governance.permissions import PermissionGuard
from agentos.governance.guardrails import GovernanceEngine


# --- Tools ---

@tool(description="Calculate a math expression")
def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Only basic math allowed"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool(description="Get current weather for a city")
def get_weather(city: str) -> str:
    weather = {
        "boston": "45°F, Cloudy with light snow",
        "new york": "48°F, Partly cloudy",
        "san francisco": "62°F, Sunny",
    }
    return weather.get(city.lower(), f"No weather data for {city}")


@tool(description="Send an email to someone")
def send_email(to: str, subject: str, body: str) -> str:
    return f"Email sent to {to} with subject '{subject}'"


@tool(description="Delete a file from the system")
def delete_file(filepath: str) -> str:
    return f"Deleted {filepath}"


# --- Create Governed Agent ---

agent = Agent(
    name="governed-bot",
    model="gpt-4o-mini",
    tools=[calculator, get_weather, send_email, delete_file],
    system_prompt="You are a helpful assistant. Use tools when needed.",
)

# --- Set Up Governance ---

gov = GovernanceEngine(
    agent_name="governed-bot",
    budget=BudgetGuard(
        max_per_action=0.05,
        max_per_hour=1.00,
        max_per_day=5.00,
        max_total=10.00,
    ),
    permissions=PermissionGuard(
        allowed_tools=["calculator", "get_weather", "send_email"],
        blocked_tools=["delete_file"],
        require_approval=["send_email"],
        max_actions_per_run=20,
    ),
)


# --- Run Governed Agent (wraps original run with governance checks) ---

def governed_run(agent: Agent, gov: GovernanceEngine, query: str):
    """Run an agent query with full governance enforcement."""
    print(f"\n{'='*60}")
    print(f"📝 Query: {query}")
    print(f"{'='*60}")

    # Store original tool execute methods
    original_executes = {}
    for t in agent.tools:
        original_executes[t.name] = t.execute

        def make_governed_execute(tool_obj, orig_exec):
            def governed_execute(call):
                # Check governance before executing
                result = gov.check_tool_call(tool_obj.name, estimated_cost=0.001)
                if not result.allowed:
                    print(f"   🚫 GOVERNANCE BLOCKED: {result.message}")
                    from agentos.core.types import ToolResult
                    return ToolResult(
                        tool_call_id=call.id,
                        name=tool_obj.name,
                        result=f"BLOCKED BY GOVERNANCE: {result.message}",
                    )
                # Execute and record
                res = orig_exec(call)
                gov.record_action(tool_obj.name, cost=0.001)
                return res
            return governed_execute

        t.execute = make_governed_execute(t, original_executes[t.name])

    # Run the agent
    msg = agent.run(query)

    # Restore original executes
    for t in agent.tools:
        t.execute = original_executes[t.name]

    return msg


# --- Demo ---

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AgentOS v0.3.0 — Governance Engine Demo")
    print("=" * 60)

    # TEST 1: Normal query — should work fine
    print("\n" + "🟢 " * 20)
    print("TEST 1: Normal calculator query (should PASS)")
    governed_run(agent, gov, "What's 25% of 400?")

    # TEST 2: Normal weather — should work
    print("\n" + "🟢 " * 20)
    print("TEST 2: Weather query (should PASS)")
    governed_run(agent, gov, "What's the weather in Boston?")

    # TEST 3: Blocked tool — delete_file is blocked
    print("\n" + "🔴 " * 20)
    print("TEST 3: Try to delete a file (should be BLOCKED)")
    governed_run(agent, gov, "Delete the file at /etc/passwd")

    # TEST 4: Requires approval — send_email needs human approval
    print("\n" + "🟡 " * 20)
    print("TEST 4: Try to send email (should require APPROVAL)")
    governed_run(agent, gov, "Send an email to boss@company.com saying the report is ready")

    # TEST 5: Kill switch
    print("\n" + "🛑 " * 20)
    print("TEST 5: Activate KILL SWITCH then try a query")
    gov.kill(reason="Suspicious activity detected — automated shutdown")
    governed_run(agent, gov, "What's the weather in New York?")

    # Revive and try again
    print("\n" + "🟢 " * 20)
    print("TEST 6: REVIVE agent and try again")
    gov.revive()
    governed_run(agent, gov, "What's 100 + 200?")

    # Print governance status and audit report
    gov.print_status()
    gov.audit.print_report()