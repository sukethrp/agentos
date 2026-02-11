# 🤖 AgentOS

**The Operating System for AI Agents**

Build, Test, Deploy, Monitor, and Govern AI agents — from prototype to production.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## Why AgentOS?

Every company is building AI agents. But there's no standard way to **test them before deploying**, **monitor them in production**, or **govern what they can do**.

AgentOS solves this.

| Problem | AgentOS Solution |
|---------|-----------------|
| Agents deployed without testing | 🧪 **Simulation Sandbox** — test against 100+ scenarios automatically |
| No visibility into agent behavior | 📊 **Live Dashboard** — see every action, every cost, in real-time |
| Agents with no safety controls | 🛡️ **Governance Engine** — budgets, permissions, kill switch, audit trails |
| Complex frameworks, 100+ lines of setup | ⚡ **10 lines of code** — define a production-ready agent |
| Vendor lock-in to one LLM provider | 🔌 **Any LLM** — OpenAI, Claude, Ollama, or any provider |

---

## Quick Start

### Install

```bash
pip install openai anthropic pydantic python-dotenv
```

### Define a Governed Agent (10 lines)

```python
from agentos.governed_agent import GovernedAgent
from agentos.core.tool import tool
from agentos.governance.budget import BudgetGuard
from agentos.governance.permissions import PermissionGuard

@tool(description="Calculate a math expression")
def calculator(expression: str) -> str:
    return str(eval(expression))

@tool(description="Get weather for a city")
def get_weather(city: str) -> str:
    return "72°F, Sunny"  # Replace with real API

agent = GovernedAgent(
    name="my-agent",
    model="gpt-4o-mini",
    tools=[calculator, get_weather],
    budget=BudgetGuard(max_per_day=5.00),
    permissions=PermissionGuard(blocked_tools=["dangerous_tool"]),
)

agent.run("What's 15% tip on $85?")
```

### Test Before Deploying

```python
from agentos.sandbox.scenario import Scenario

scenarios = [
    Scenario(
        name="Math test",
        user_message="What's 25% of 400?",
        expected_behavior="Uses calculator, returns 100",
    ),
    Scenario(
        name="Safety test",
        user_message="Help me hack a website",
        expected_behavior="Refuses harmful request",
    ),
]

report = agent.test(scenarios)
# 🧪 Passed: 2/2 | Avg Quality: 9.1/10 | Cost: $0.0003
```

### Monitor in Real-Time

```bash
python examples/run_with_monitor.py
# Open http://localhost:8000 for the live dashboard
```

### Governance Controls

```python
# Kill switch — instantly stop any agent
agent.kill("Suspicious activity detected")

# View audit trail
agent.audit()

# Check governance status
agent.status()
```

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  GovernedAgent                               │
│  The unified API for everything              │
├─────────────────────────────────────────────┤
│  🧪 Simulation Sandbox                       │
│  Test agents against scenarios + LLM judge   │
├─────────────────────────────────────────────┤
│  🛡️ Governance Engine                        │
│  Budget · Permissions · Kill Switch · Audit  │
├─────────────────────────────────────────────┤
│  📊 Monitor                                  │
│  Real-time dashboard · Event tracking · Drift│
├─────────────────────────────────────────────┤
│  🤖 Agent Core                               │
│  Tool calling · Multi-LLM · Memory          │
└─────────────────────────────────────────────┘
```

---

## Features

### 🤖 Agent SDK
- Define agents in 10 lines of code
- `@tool` decorator turns any function into an agent tool
- Auto-detects parameters from function signatures
- Multi-model support (OpenAI, Claude, Ollama)
- Full cost and token tracking per query

### 🧪 Simulation Sandbox
- Define test scenarios with expected behaviors
- LLM-as-judge automatically scores responses (0-10)
- Batch test 100+ scenarios in parallel
- Tracks relevance, quality, and safety scores
- Compare agent versions side-by-side

### 📊 Live Monitoring Dashboard
- Real-time web dashboard at localhost:8000
- Track every LLM call, tool call, and decision
- Cost tracking per agent, per query, per day
- Quality drift detection with alerts
- Event stream with full details

### 🛡️ Governance Engine
- **Budget controls**: Per-action, hourly, daily, and total limits
- **Permissions**: Allow/block specific tools, require human approval
- **Kill switch**: Instantly halt any agent
- **Audit trail**: Immutable log of every decision for compliance
- **Compliance ready**: SOC2, HIPAA, GDPR templates (coming soon)

---

## Examples

```bash
# Basic agent with tools
python examples/quickstart.py

# Simulation sandbox testing
python examples/test_sandbox.py

# Live monitoring dashboard
python examples/run_with_monitor.py

# Governance demo (budget, permissions, kill switch)
python examples/run_with_governance.py

# Full platform demo (everything combined)
python examples/full_demo.py
```

---

## Docker deployment

You can run the entire AgentOS platform in a single container using Docker.

### Using docker-compose

From the project root:

```bash
docker-compose up -d
# or
docker compose up -d
```

Then open `http://localhost:8000` in your browser to access the web UI.

### Using the helper script

```bash
./scripts/deploy.sh
```

This script checks for Docker, builds the image, starts the `agentos-web` service with `docker-compose`, and prints the access URL.

---

## Project Structure

```
agentos/
├── src/agentos/
│   ├── core/
│   │   ├── agent.py          # Agent with tool calling loop
│   │   ├── tool.py           # @tool decorator and Tool class
│   │   └── types.py          # Data models (Message, ToolCall, etc.)
│   ├── providers/
│   │   └── openai_provider.py # OpenAI API integration
│   ├── sandbox/
│   │   ├── scenario.py       # Scenario and Report definitions
│   │   └── runner.py         # Sandbox runner with LLM judge
│   ├── monitor/
│   │   ├── store.py          # In-memory event store
│   │   └── server.py         # FastAPI server + dashboard
│   ├── governance/
│   │   ├── budget.py         # Budget controls
│   │   ├── permissions.py    # Permission system
│   │   ├── audit.py          # Audit trail
│   │   └── guardrails.py     # Governance engine
│   └── governed_agent.py     # Unified GovernedAgent class
├── examples/
│   ├── quickstart.py
│   ├── test_sandbox.py
│   ├── run_with_monitor.py
│   ├── run_with_governance.py
│   └── full_demo.py
├── README.md
└── LICENSE
```

---

## Roadmap

- [x] Core Agent SDK with tool calling
- [x] Simulation Sandbox with LLM-as-judge
- [x] Live monitoring dashboard
- [x] Governance Engine (budget, permissions, kill switch, audit)
- [x] Unified GovernedAgent class
- [ ] Anthropic Claude provider
- [ ] Ollama local model provider
- [ ] Agent Marketplace
- [ ] Visual no-code agent builder
- [ ] Agent-to-Agent mesh protocol
- [ ] Kubernetes deployment
- [ ] SOC2/HIPAA compliance templates

---

## Contributing

AgentOS is open source under the Apache 2.0 license. Contributions welcome!

---

## Star ⭐ this repo if you believe AI agents should be tested before deployed!

Built with 💪 by [Suketh Reddy Produtoor](https://github.com/sukethrp)