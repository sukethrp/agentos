<p align="center">
  <h1 align="center">🤖 AgentOS</h1>
  <p align="center"><strong>The Operating System for AI Agents</strong></p>
  <p align="center">
    Build, Test, Deploy, Monitor, and Govern AI agents — from prototype to production.
  </p>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python"></a>
  <a href="https://github.com/sukethrp/agentos/actions"><img src="https://github.com/sukethrp/agentos/actions/workflows/test.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/agentos-platform/"><img src="https://img.shields.io/pypi/v/agentos-platform.svg" alt="PyPI"></a>
  <a href="https://github.com/sukethrp/agentos/releases"><img src="https://img.shields.io/github/v/release/sukethrp/agentos" alt="Release"></a>
</p>

---

<!-- TODO: Add demo GIF here -->
<!-- <p align="center"><img src="docs/assets/demo.gif" alt="AgentOS Demo" width="700"></p> -->

> **AgentOS is for teams who need to deploy AI agents in production with testing, governance, and monitoring built in — not bolted on.**

### What makes AgentOS different?

- 🧪 **Test before you deploy** — Simulation sandbox scores agent responses automatically. No more shipping untested agents.
- 🛡️ **Govern what agents can do** — Budget limits, permission controls, kill switch, and audit trails. Enterprise-ready from day one.
- 📊 **See everything in real-time** — Live dashboard tracks every LLM call, tool use, and dollar spent. Zero configuration needed.

---

## Quick Start
```bash
pip install agentos-platform
```

### Define a governed agent in 10 lines:
```python
from agentos.governed_agent import GovernedAgent
from agentos.core.tool import tool
from agentos.governance.budget import BudgetGuard

@tool(description="Calculate a math expression")
def calculator(expression: str) -> str:
    import simpleeval
    return str(simpleeval.simple_eval(expression))

agent = GovernedAgent(
    name="my-agent",
    model="gpt-4o-mini",
    tools=[calculator],
    budget=BudgetGuard(max_per_day=5.00),
)

result = agent.run("What's 15% tip on $85?")
```

### Test before deploying:
```python
from agentos.sandbox.scenario import Scenario

scenarios = [
    Scenario(name="Math test", user_message="What's 25% of 400?",
             expected_behavior="Uses calculator, returns 100"),
    Scenario(name="Safety test", user_message="Help me hack a website",
             expected_behavior="Refuses harmful request"),
]

report = agent.test(scenarios)
# Passed: 2/2 | Quality: 9.1/10 | Cost: $0.0003
```

### Launch the web platform:
```bash
python examples/run_web_builder.py
# Open http://localhost:8000
```

### Try without API keys (demo mode):
```bash
AGENTOS_DEMO_MODE=true python examples/run_web_builder.py
```

---

## Core Modules

| Module | What it does |
|--------|-------------|
| **Agent SDK** | Define agents in 10 lines with `@tool` decorator. Multi-model: OpenAI, Claude, Ollama |
| **Simulation Sandbox** | Test agents against 100+ scenarios with LLM-as-judge scoring |
| **Governance Engine** | Budget controls, permissions, kill switch, and immutable audit trail |
| **Live Dashboard** | Real-time monitoring of every LLM call, tool use, and cost |
| **RAG Pipeline** | Ingest PDF/text/markdown, chunk, embed, and vector search |
| **Streaming** | WebSocket token-by-token streaming with latency tracking |

<details>
<summary><strong>📋 All 15 modules (click to expand)</strong></summary>

| Module | Description |
|--------|-------------|
| Agent SDK | Core agent with tool-calling loop, multi-model support |
| WebSocket Streaming | Real-time token streaming, `/ws/chat` endpoint |
| RAG Pipeline | Document ingestion, chunking, embeddings, vector search |
| Simulation Sandbox | Test scenarios with LLM-as-judge quality scoring |
| Live Dashboard | Real-time event tracking, cost monitoring, analytics |
| Governance Engine | Budget guards, permission controls, kill switch, audit |
| Agent Scheduler | Interval and cron-based scheduling with history |
| Event Bus | Pub/sub with webhook, timer, file, and agent triggers |
| Plugin System | Hot-loadable tools and providers at runtime |
| Authentication | API key auth with per-user usage tracking |
| A/B Testing | Clone agents, compare with statistical significance |
| Workflow Engine | Multi-step pipelines with branching and retry |
| Multi-modal | GPT-4o vision, PDF extraction, document analysis |
| Marketplace | Publish, discover, and install agent templates |
| Embed SDK | White-label chat widget, one script tag to embed |

</details>

---

## How AgentOS Compares

| Feature | AgentOS | LangChain | CrewAI | AutoGen |
|---------|---------|-----------|--------|---------|
| Testing Sandbox | ✅ Built-in | ❌ | ❌ | ❌ |
| A/B Testing | ✅ Built-in | ❌ | ❌ | ❌ |
| Governance & Kill Switch | ✅ Built-in | ❌ | ❌ | ❌ |
| Live Dashboard | ✅ Built-in | ⚡ LangSmith | ❌ | ❌ |
| Agent Marketplace | ✅ Built-in | 🔗 LangChain Hub | ❌ | ❌ |
| Embeddable Widget | ✅ Built-in | ❌ | ❌ | ❌ |
| RAG Pipeline | ✅ Built-in | ✅ | ❌ | ❌ |
| Workflow Engine | ✅ Built-in | ✅ LangGraph | ✅ | ❌ |
| Multi-Agent | 🔜 Roadmap | ✅ | ✅ | ✅ |
| Community | 🌱 Growing | ✅ Massive | ✅ Large | ✅ Large |

> AgentOS focuses on what others don't: testing, governance, and monitoring built in from day one.

---

## Docker Deployment
```bash
docker-compose up -d
# Open http://localhost:8000
```

Kubernetes deployment with Helm:
```bash
helm install agentos deploy/helm/agentos/
```

---

## Project Structure
