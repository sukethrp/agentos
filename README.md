<p align="center">
  <h1 align="center">🤖 AgentOS</h1>
  <p align="center"><strong>The Operating System for AI Agents</strong></p>
  <p align="center">Build, Test, Deploy, Monitor, and Govern AI agents — from prototype to production.</p>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg"></a>
  <a href="https://github.com/sukethrp/agentos/actions"><img src="https://github.com/sukethrp/agentos/actions/workflows/test.yml/badge.svg"></a>
  <a href="https://github.com/sukethrp/agentos/releases"><img src="https://img.shields.io/github/v/release/sukethrp/agentos"></a>
</p>

<p align="center">
  <a href="https://agentos-mocha.vercel.app">🌐 Live Demo</a> ·
  <a href="#quick-start">🚀 Quick Start</a> ·
  <a href="https://github.com/sukethrp/agentos/issues">📋 Issues</a>
</p>

<!-- Architecture diagram -->
<p align="center">
  <img src="docs/assets/architecture.png" alt="AgentOS Architecture" width="700">
</p>

> **For teams who need to deploy AI agents with testing, governance, and monitoring built in — not bolted on.**

## 3 Differentiators

- 🧪 **Test**: Run scenario-based simulation before deploy, with quality and cost scoring.
- 🛡️ **Govern**: Enforce budgets, permissions, and kill-switch policies with auditability.
- 📊 **Monitor**: Observe live agent runs, tool usage, latency, and spend in one dashboard.

## Quick Start

```bash
pip install agentos-platform
```

10-line example:

```python
from agentos.governed_agent import GovernedAgent
from agentos.core.tool import tool

@tool(description="Add two numbers")
def add(a: float, b: float) -> float:
    return a + b

agent = GovernedAgent(name="demo", model="gpt-4o-mini", tools=[add])
print(agent.run("What is 12.5 + 7.5?"))
```

Demo mode:

```bash
AGENTOS_DEMO_MODE=true python examples/run_web_builder.py
```

## MCP Setup (Claude Desktop + Cursor)

Install the MCP extra:

```bash
pip install 'agentos-platform[mcp]'
```

### 1) Start the MCP server

Expose built-in AgentOS tools (stdio transport is the safest choice for MCP clients like Claude Desktop and Cursor):

```bash
agentos mcp serve --transport stdio
```

Expose tools from a specific agent module (example `./my_agent/agent.py`):

```bash
agentos mcp serve --transport stdio --agent ./my_agent
```

Optional: run the HTTP SSE transport for clients that support it:

```bash
agentos mcp serve --transport sse --host 127.0.0.1 --port 8080
```

### 2) Configure Claude Desktop

Add the following snippet to your `claude_desktop_config.json` (restart Claude Desktop after editing):

```json
{
  "mcpServers": {
    "agentos": {
      "command": "agentos",
      "args": ["mcp", "serve", "--transport", "stdio"]
    }
  }
}
```

If you want a specific agent module:

```json
{
  "mcpServers": {
    "agentos": {
      "command": "agentos",
      "args": ["mcp", "serve", "--transport", "stdio", "--agent", "/absolute/path/to/agent.py"]
    }
  }
}
```

### 3) Configure Cursor

Add to Cursor `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "agentos": {
      "command": "agentos",
      "args": ["mcp", "serve", "--transport", "stdio"]
    }
  }
}
```

## Core Modules

| Module | What it does |
|--------|---------------|
| Agent SDK | Define agents and tools with provider-agnostic model routing |
| Simulation Sandbox | Test scenarios with LLM-as-judge quality and pass/fail scoring |
| Governance Engine | Budget controls, permissions, kill switch, and audit logging |
| Live Dashboard | Real-time traces for prompts, tool calls, latency, and spend |
| RAG Pipeline | Ingest, chunk, embed, and retrieve knowledge sources |
| Workflow Engine | Compose repeatable multi-step agent workflows |

<details>
<summary><strong>📋 Full 15-module list (click to expand)</strong></summary>

| Module | Description |
|--------|-------------|
| Agent SDK | Core governed agent runtime and tool-calling loop |
| WebSocket Streaming | Token streaming and low-latency interactive sessions |
| RAG Pipeline | Ingestion, chunking, embeddings, retrieval, and reranking |
| Simulation Sandbox | Scenario simulation, scoring, and comparison reports |
| Live Dashboard | Event stream, usage analytics, and operational visibility |
| Governance Engine | Guardrails, budget caps, permission checks, and audits |
| Agent Scheduler | Interval and cron scheduling with execution history |
| Event Bus | Trigger-driven orchestration via internal and external events |
| Plugin System | Runtime-extensible tools, providers, and adapters |
| Authentication | API key auth, org and user usage tracking, and middleware |
| A/B Testing | Side-by-side evaluation for variants and prompt changes |
| Workflow Engine | DAG-based execution with retries and branching |
| Multimodal | Vision and document flows for image and file-aware agents |
| Marketplace | Template registry for reusable agents and workflows |
| Embed SDK | Embeddable widget and integration surface for web apps |

</details>

## Honest Comparison

| Capability | AgentOS | LangChain | CrewAI | AutoGen |
|------------|---------|-----------|--------|---------|
| Built-in testing sandbox | ✅ Native | ❌ External setup | ❌ External setup | ❌ External setup |
| Governance (budget/kill switch) | ✅ Native | ⚠️ Custom code | ⚠️ Custom code | ⚠️ Custom code |
| Real-time ops dashboard | ✅ Native | ⚠️ LangSmith add-on | ❌ | ❌ |
| Batteries-included platform | ✅ Yes | ⚠️ Framework-first | ⚠️ Orchestration-first | ⚠️ Research-first |
| Ecosystem maturity | 🌱 Growing | ✅ Very mature | ✅ Mature | ✅ Mature |

## Benchmarks
See [full benchmark results](docs/benchmarks.md). Key findings:
- Our weighted evaluation ensemble correlates 0.91 with human judgment
- Local embeddings achieve 95% of OpenAI quality at zero cost
- Governance adds <5ms overhead to any query

## Architecture

See the architecture diagram above and `docs/` for component-level details and ADRs.

## Project Structure

```text
agentos/
├── src/agentos/      # Core platform modules
├── frontend/         # React frontend
├── dashboard/        # Web dashboard UI
├── deploy/helm/      # Helm charts
├── examples/         # Runnable examples
├── tests/            # Unit and integration tests
└── docs/             # Docs and ADRs
```

## Contributing

Contributions are welcome: [CONTRIBUTING.md](CONTRIBUTING.md)

## Roadmap

Roadmap and upcoming work are tracked in [GitHub Issues](https://github.com/sukethrp/agentos/issues).
