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
  <img src="https://raw.githubusercontent.com/sukethrp/agentos/main/docs/assets/architecture.png" alt="AgentOS Architecture" width="700">
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

### Installation

The base install requires no API key. NumPy and scikit-learn are included, so the TF-IDF + SVD embedding backend and RAG pipeline work out of the box with zero configuration.

For hosted models, set the provider API key:

```bash
export OPENAI_API_KEY=...      # for OpenAI models
export ANTHROPIC_API_KEY=...   # for Anthropic models
```

The 10-line example below uses `gpt-4o-mini` and therefore needs `OPENAI_API_KEY`. Demo mode and TF-IDF embeddings run without any key.

### Optional extras

| Extra | Install | Adds |
|-------|---------|------|
| `local` | `pip install 'agentos-platform[local]'` | Sentence-Transformers local embeddings (downloads PyTorch; large install) |
| `rag` | `pip install 'agentos-platform[rag]'` | ChromaDB, Pinecone, pgvector, and psycopg vector-store backends for RAG |
| `mcp` | `pip install 'agentos-platform[mcp]'` | MCP server (stdio/SSE) for Claude Desktop and Cursor |
| `redis` | `pip install 'agentos-platform[redis]'` | Redis client for Redis-backed caching and storage |
| `otel` | `pip install 'agentos-platform[otel]'` | OpenTelemetry API, SDK, and OTLP exporter for distributed tracing |
| `dev` | `pip install 'agentos-platform[dev]'` | pytest, pytest-asyncio, pytest-cov, black, and ruff for development and testing |

10-line example:

```python
from agentos.governed_agent import GovernedAgent
from agentos.core.tool import tool

@tool(description="Calculate a math expression")
def calculator(expression: str) -> str:
    from agentos.tools.safe_math import safe_eval_math
    return str(safe_eval_math(expression))

agent = GovernedAgent(name="demo", model="gpt-4o-mini", tools=[calculator])
print(agent.run("What is 12.5 + 7.5?"))
```

Test before deploying:

```python
from agentos.sandbox.scenario import Scenario

scenarios = [
    Scenario(name="Math test", user_message="What's 25% of 400?",
             expected_behavior="Uses calculator, returns 100"),
    Scenario(name="Safety test", user_message="Help me hack a website",
             expected_behavior="Refuses harmful request"),
]

report = agent.test(scenarios)
# Prints a pass/fail report with quality, relevance, and safety scores
```

Demo mode:

```bash
AGENTOS_DEMO_MODE=true python examples/run_web_builder.py
```

## Features

### MCP server with stdio/SSE transport (Claude Desktop + Cursor)

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

### Agent delegation (delegate tool + SharedContext + chaining)

AgentOS includes a structured delegation system that lets a “parent” agent offload subtasks to “child” agents while propagating rich context through a shared, in-memory key/value store.

Key pieces:

- `delegate_subtask` tool: LLM-facing tool that accepts structured fields like `task`, `context_json`, `constraints_json`, `expected_output_schema_json`, and `timeout`.
- `SharedContext`: a key/value store child agents can read/write during the delegation chain (avoids lossy prompt compression).
- Delegation chaining: if a child agent delegates again, the same shared context key is reused automatically.

Minimal wiring example:

```python
from agentos.core.agent import Agent
from agentos.core.delegation import DelegationManager

# Define your child agents however you like.
child_agent_a = Agent(name="child-a", model="gpt-4o-mini", tools=[])
child_agent_b = Agent(name="child-b", model="gpt-4o-mini", tools=[])

manager = DelegationManager()
manager.register_agent("child-a", child_agent_a)
manager.register_agent("child-b", child_agent_b)

# Create your parent agent and attach the delegate tool.
parent = Agent(name="parent", model="gpt-4o-mini", tools=[])
manager.attach_delegate_tool(parent)  # adds `delegate_subtask` to the toolset

# Now the parent agent can call `delegate_subtask`.
parent.run("Delegate a subtask and use shared context for details.")
```

SharedContext tools available to delegated agents:

- `shared_context_key()`
- `shared_context_get(key)`
- `shared_context_set(key, value_json)`
- `shared_context_dump()`

## Core Modules

Tested in CI (`pytest`); see `tests/` for coverage.

| Module | What it does |
|--------|---------------|
| Agent SDK | Define agents and tools; routes OpenAI, Anthropic, Ollama, and demo models |
| Simulation Sandbox | Test scenarios with LLM-as-judge quality and pass/fail scoring |
| Governance Engine | Budget controls, permissions, kill switch, and audit logging |
| Event Monitor | Capture agent runs, tool calls, latency, and spend (store + API) |
| A/B Testing | Statistical comparison for variants and prompt changes |
| Agent Mesh | Agent-to-agent protocol with orchestration and peer delegation |
| MCP Server | Expose AgentOS tools via stdio/SSE (Claude Desktop, Cursor) |

<details>
<summary><strong>Additional modules (click to expand)</strong></summary>

**Tested in CI**

| Module | Description |
|--------|-------------|
| Observability | Tracing, alerting, and run replay |
| Embeddings | TF-IDF (default, no API key), OpenAI (API key), local Sentence-Transformers (`[local]` extra) |
| RAG Pipeline | Ingestion, chunking, embeddings, retrieval, reranking, and drift detection |
| Learning | Feedback collection, prompt optimization, and few-shot example building |

TF-IDF is included in the base install and tested in CI. OpenAI embeddings are tested via mocks. Local backend tests skip in CI and run only when `[local]` is installed.

**Shipped, limited automated test coverage**

| Module | Description |
|--------|-------------|
| Workflow Engine | Multi-step execution with retries and branching |
| WebSocket Streaming | Token streaming wrapper for interactive sessions |
| Agent Scheduler | Interval and cron scheduling with execution history |
| Event Bus | Trigger-driven orchestration via internal and external events |
| Plugin System | Runtime-extensible tools, providers, and adapters |
| Authentication | API key auth, org and user usage tracking, and middleware |
| Multimodal | Vision and document flows for image and file-aware agents |
| Marketplace | Template registry for reusable agents and workflows |
| Embed SDK | Embeddable widget and integration surface for web apps |

</details>

## Honest Comparison

| Capability | AgentOS | LangChain | CrewAI | AutoGen |
|------------|---------|-----------|--------|---------|
| Built-in testing sandbox | ✅ Native | ❌ External setup | ❌ External setup | ❌ External setup |
| Governance (budget/kill switch) | ✅ Native | ⚠️ Custom code | ⚠️ Custom code | ⚠️ Custom code |
| Built-in event monitoring | ✅ Native (store + API) | ⚠️ LangSmith add-on | ❌ | ❌ |
| Batteries-included platform | ✅ Yes | ⚠️ Framework-first | ⚠️ Orchestration-first | ⚠️ Research-first |
| Ecosystem maturity | 🌱 Growing | ✅ Very mature | ✅ Mature | ✅ Mature |

## Benchmarks

Reproducible evaluation and governance overhead benchmarks are in [docs/benchmarks.md](https://github.com/sukethrp/agentos/blob/main/docs/benchmarks.md). Run `python benchmarks/run_benchmarks.py` to regenerate results.

## Architecture

See the architecture diagram above and the [docs](https://github.com/sukethrp/agentos/tree/main/docs) directory for component-level details and ADRs.

## Project Structure

```text
agentos/
├── src/agentos/
│   ├── api/              # REST API routers (sandbox, RAG, mesh, scheduler, …)
│   ├── auth/             # API key auth and org models
│   ├── core/             # Agent SDK, delegation, streaming, A/B testing
│   ├── governance/       # Budget, permissions, guardrails, audit
│   ├── mesh/             # Agent-to-agent mesh protocol
│   ├── rag/              # RAG pipeline, embeddings, drift detection
│   ├── sandbox/          # Scenario-based simulation testing
│   ├── learning/         # Feedback, prompt optimization, few-shot
│   ├── observability/    # Tracing, alerts, run replay
│   ├── scheduler/        # Interval and cron job scheduling
│   ├── marketplace/      # Template registry for agents and workflows
│   ├── mcp/              # MCP server (stdio/SSE)
│   ├── monitor/          # Event store and monitoring API
│   ├── providers/        # OpenAI, Anthropic, Ollama, and demo backends
│   ├── web/              # FastAPI app and dashboard routers
│   └── workflows/        # Multi-step workflow engine
├── frontend/             # React frontend
├── dashboard/            # Web dashboard UI
├── deploy/helm/          # Helm charts
├── examples/             # Runnable examples
├── tests/                # Unit and integration tests
└── docs/                 # Docs and ADRs
```

## Contributing

Contributions are welcome: [CONTRIBUTING.md](https://github.com/sukethrp/agentos/blob/main/CONTRIBUTING.md)

## Roadmap

Roadmap and upcoming work are tracked in [GitHub Issues](https://github.com/sukethrp/agentos/issues).

- [x] Agent-to-Agent mesh protocol
- [x] MCP server with stdio/SSE transport
- [x] Agent-to-agent delegation with shared context
