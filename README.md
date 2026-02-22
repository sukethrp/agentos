# AgentOS

**The Operating System for AI Agents**

Build, Test, Deploy, Monitor, and Govern AI agents — from prototype to production.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/agentos-platform.svg)](https://pypi.org/project/agentos-platform/)
[![Version](https://img.shields.io/badge/version-0.3.1-green.svg)](#)

---

## Why AgentOS?

Every company is building AI agents. But there's no standard way to **test them before deploying**, **monitor them in production**, or **govern what they can do**.

AgentOS solves this.

| Problem | AgentOS Solution |
|---------|-----------------|
| Agents deployed without testing | **Simulation Sandbox** — test against 100+ scenarios automatically |
| No visibility into agent behavior | **Live Dashboard** — see every action, every cost, in real-time |
| Agents with no safety controls | **Governance Engine** — budgets, permissions, kill switch, audit trails |
| Complex frameworks, 100+ lines of setup | **10 lines of code** — define a production-ready agent |
| Vendor lock-in to one LLM provider | **Any LLM** — OpenAI, Claude, Ollama, or any provider |
| No way to share or reuse agents | **Marketplace** — publish packages, install via CLI, scope API keys |
| Multi-agent orchestration | **Teams** — WorkflowDAG, PlannerAgent, parallel execution, org budget caps |
| Can't embed agents in your product | **Embed SDK** — white-label chat widget in one script tag |

---

## Quick Start

### Install

```bash
pip install agentos-platform
```

Or install from source:

```bash
git clone https://github.com/sukethrp/agentos.git
cd agentos
pip install -e ".[dev]"
```

Optional extras:
- `pip install -e ".[rag]"` — ChromaDB, Pinecone, pgvector for RAG vector stores

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
    return "72F, Sunny"  # Replace with real API

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
# Passed: 2/2 | Avg Quality: 9.1/10 | Cost: $0.0003
```

### Launch the Web Platform

```bash
python examples/run_web_builder.py
# Open http://localhost:8000
```

---

## Features

### Agent SDK
- Define agents in 10 lines of code
- `@tool` decorator turns any function into an agent tool (supports `timeout_seconds`, `max_retries`)
- **Parallel tool execution**: multiple tool calls per turn run via `asyncio.gather` with `asyncio.wait_for`
- **TTLCache**: tool result caching by `hash(tool_name+args)` (maxsize=256, ttl=300s)
- **ToolExecutionContext**: `agent_id`, `session_id`, `budget_remaining` passed to tools
- **Retry**: failed tool calls retried up to `max_retries` with 2^n exponential backoff
- Auto-detects parameters from function signatures
- Multi-model support (OpenAI, Claude, Ollama)
- Full cost and token tracking per query

### WebSocket Streaming & Monitor
- Real-time token-by-token streaming like ChatGPT
- `Agent.run(query, stream=True)` returns a generator
- WebSocket endpoints: `/ws/chat` (streaming), `/ws/monitor` (tool/team events)
- Tool events: `{agent_id, tool_name, result, latency_ms}` broadcast to monitor clients
- Team events: `{team_id, node_id, status, output}` for workflow execution
- Streaming stats (first-token latency, cost, token count)

### RAG Pipeline
- Ingest PDF, text, and markdown documents
- **Vector stores**: ChromaDB, Pinecone, pgvector (select via `rag_config.vector_store`)
- **Chunk strategies**: fixed, sentence, semantic (CLI: `agentos rag ingest --source ./docs --collection name --chunk-strategy sentence`)
- **HybridRetriever**: dense embedding + BM25 with Reciprocal Rank Fusion
- **CrossEncoderReranker**: sentence-transformers `ms-marco-MiniLM-L-6-v2`
- OpenAI `text-embedding-3-small` with batching and caching
- Exposed as `rag_search(query, collection, top_k)` tool for any agent

### Simulation Sandbox
- **EvaluationScenario**: load from YAML (`scenario_id`, `input`, `expected_output`, `rubric`, `tags`)
- **LLMJudgeScorer**: async scoring with Claude Sonnet (0–1 scale, structured JSON)
- **SimulationRunner.run_batch**: parallel execution with `asyncio.Semaphore`, SQLite storage
- **ComparisonReport**: per-scenario delta scores between runs
- `GET /sandbox/runs/{run_id}/report` for run results
- Define test scenarios with expected behaviors
- LLM-as-judge automatically scores responses
- Compare agent versions side-by-side

### Live Monitoring & Analytics Dashboard
- Real-time web dashboard at `localhost:8000`
- Track every LLM call, tool call, and decision
- Cost over time (line chart), popular tools (bar chart)
- Model comparison table, agent leaderboard
- Quality drift detection with alerts

### Governance Engine
- **Budget controls**: Per-action, hourly, daily, and total limits
- **Org monthly caps**: Token and cost caps per organization; raises `BudgetExceededError` when exceeded
- **Permissions**: Allow/block specific tools, require human approval
- **Kill switch**: Instantly halt any agent
- **Audit trail**: Immutable log of every decision for compliance

### Agent Scheduler
- Schedule agents with intervals (`5m`, `1h`, `1d`) or cron expressions (`0 9 * * *`)
- Execution history tracking (last run, next run, results)
- Max concurrent job limits
- Start/stop/pause via API

### Event Bus
- Publish/subscribe event system for agent orchestration
- Event types: `webhook.received`, `file.changed`, `agent.completed`, `schedule.triggered`, `custom.*`
- Triggers: WebhookTrigger, TimerTrigger, AgentCompleteTrigger, FileTrigger
- Query templates with variable substitution

### Plugin System
- Discover and load plugins from any directory
- Class-based (`BasePlugin`) or function-based (`register()`) plugins
- Register tools, providers, or middleware at runtime
- Built-in example plugins (translate, GitHub integration)

### User Authentication & Usage Tracking
- API key-based authentication
- **Organizations**: `Organization`, `OrgMembership`, `Role` (OWNER, ADMIN, DEVELOPER, VIEWER) with permission matrix
- **API key scopes**: Restrict keys to specific `agent_id` or `tool_name`; enforced via middleware
- **UsageTrackerAsync**: SQLite-backed `record(api_key, tokens, cost)` and `get_usage(api_key, month)`
- Org routers: `POST /auth/orgs`, `GET /auth/orgs/{org_id}/members`, `POST /auth/orgs/{org_id}/members`, `DELETE /auth/orgs/{org_id}/members/{user_id}`
- Per-user usage tracking (queries, tokens, cost)

### A/B Testing
- Clone agents and compare performance
- LLM-as-judge scoring with statistical significance
- Per-query breakdown with confidence intervals
- Integrated with agent versioning system

### Workflow System & Multi-Agent Teams
- **WorkflowDAG**: YAML-defined DAG with networkx (nodes: agent_id, type; edges: source, target, condition_expr)
- **PlannerAgent**: LLM plans subtasks `[{task, agent_id, depends_on}]` validated against registered agents
- **TeamRunner**: topological execution, parallel nodes via `asyncio.gather`, condition_expr eval with `outputs` dict
- **ResultAggregator**: LLM synthesis of subtask outputs into structured response
- Shared token budget via GovernanceEngine
- Multi-step agent pipelines with fluent API
- Conditional branching based on step results
- Parallel step execution

### Multi-modal Support
- Image analysis via OpenAI Vision API (GPT-4o)
- PDF text extraction (pure Python, no external libraries)
- Document reading tools (text, markdown, CSV, JSON)
- File upload endpoint for the web UI

### Conversation Branching
- Fork conversations at any point to explore "what if" scenarios
- Switch between branches, compare side-by-side
- Merge insights from multiple branches
- Full branch tree management via API

### Agent Marketplace
- **Package-based**: `PackageManifest` (name, version, entry_point, inputs, outputs, pricing_model)
- **MarketplaceRegistry**: JSON persistence at `~/.agentos/marketplace.json`
- **CLI**: `agentos marketplace publish --manifest agentos-package.yaml`, `agentos marketplace install <name>`
- `GET /marketplace/search?tags=&capability=` for filtered package list
- Publish agent templates for the community
- Search by name, category, tags
- Install agents with one click (wires entry_point into tool registry)
- Rating and review system

### Embeddable Chat Widget & SDK
- White-label chat widget for any website
- Dark/light theme, customisable colours, logo, position
- One script tag to embed — no build step
- Python SDK: `AgentOSClient` with `run()`, `stream()`, `list_agents()`
- WebSocket streaming with automatic HTTP fallback
- CORS enabled for cross-origin embedding

### Pre-built Templates
- Customer Support, Research Assistant, Sales Agent, Code Reviewer
- Ready to deploy or customise

---

## Web Platform

The AgentOS web platform provides a visual interface for everything:

```bash
python examples/run_web_builder.py
# Open http://localhost:8000
```

**Sections:**
- **Agent Builder** — configure and run agents visually
- **Visual Builder** — ReactFlow canvas with AgentNode, ToolNode, RAGNode, TeamNode, ConditionNode, OutputNode; config sidebar, edge condition_expr labels; Save (POST /workflows/), Run (POST /workflows/{id}/run) with live node status via /ws/monitor
- **Templates** — browse and deploy pre-built agents
- **Chat** — real-time streaming conversation
- **Branching** — fork and explore conversation paths
- **Monitor** — live event and cost tracking
- **Analytics** — cost trends, tool usage, model comparison, leaderboard
- **Scheduler** — create and manage scheduled jobs
- **Events** — event bus listeners and triggers
- **A/B Testing** — compare agent variants
- **Multi-modal** — upload images/documents for analysis
- **Account & Usage** — authentication and usage stats
- **Marketplace** — publish, discover, and install agents
- **Embed SDK** — generate embeddable widget snippets

---

## Examples

```bash
# Quick start
python examples/quickstart.py

# Web platform (all features)
python examples/run_web_builder.py

# Simulation sandbox testing
python examples/test_sandbox.py

# RAG ingestion (CLI)
agentos rag ingest --source ./docs --collection my_docs --chunk-strategy sentence

# Marketplace publish/install
agentos marketplace publish --manifest agentos-package.yaml
agentos marketplace install my-package

# Visual Builder (React frontend)
cd frontend && npm install && npm run dev

# Live monitoring dashboard
python examples/run_with_monitor.py

# Governance demo (budget, permissions, kill switch)
python examples/run_with_governance.py

# Full platform demo (everything combined)
python examples/full_demo.py

# WebSocket streaming
python examples/streaming_demo.py

# RAG pipeline
python examples/rag_demo.py

# Agent scheduler
python examples/scheduler_demo.py

# Event-driven agents
python examples/event_demo.py

# Plugin system
python examples/plugin_demo.py

# User authentication
python examples/auth_demo.py

# A/B testing
python examples/ab_test_demo.py

# Multi-step workflows
python examples/workflow_demo.py

# Multi-modal (vision + documents)
python examples/multimodal_demo.py

# Conversation branching
python examples/branching_demo.py

# Agent marketplace
python examples/marketplace_demo.py

# Pre-built templates
python examples/templates_demo.py
```

---

## Docker Deployment

Run the entire platform in a single container:

```bash
# Using docker-compose
docker-compose up -d

# Or use the helper script
./scripts/deploy.sh
```

Open `http://localhost:8000` to access the web UI.

---

## Embed in Your Product

Add an AI agent to any website with two lines:

```html
<script>
  window.AgentOSConfig = {
    baseUrl: "https://your-agentos-server.com",
    agentName: "Support Bot",
    theme: "dark",
    accentColor: "#6c5ce7",
  };
</script>
<script src="https://your-agentos-server.com/embed/chat.js"></script>
```

Or use the Python SDK:

```python
from agentos.embed import AgentOSClient

client = AgentOSClient(base_url="http://localhost:8000")
response = client.run("How can I help?")

# Streaming
for token in client.stream("Tell me a story"):
    print(token, end="", flush=True)
```

---

## Project Structure

```
agentos/
├── src/agentos/
│   ├── core/              # Agent, Tool, Types, Memory, Versioning, A/B Testing
│   │   ├── agent.py       #   Agent with tool calling loop + streaming
│   │   ├── tool.py        #   @tool decorator and Tool class
│   │   ├── types.py       #   Pydantic models (Message, ToolCall, AgentEvent, ToolExecutionContext)
│   │   ├── memory.py      #   Agent memory and fact extraction
│   │   ├── streaming.py   #   StreamingAgent wrapper
│   │   ├── ab_testing.py  #   Agent cloning and A/B testing
│   │   ├── versioning.py  #   Agent version control
│   │   ├── branching.py   #   Conversation branching
│   │   ├── multimodal.py  #   Image/PDF processing utilities
│   │   └── storage.py     #   Persistent key-value storage
│   ├── providers/         # LLM provider integrations
│   ├── sandbox/           # Simulation testing (EvaluationScenario, LLMJudgeScorer, SimulationRunner, ComparisonReport)
│   ├── monitor/           # Real-time event store + dashboard
│   ├── governance/        # Budget, Permissions, Kill Switch, Audit
│   ├── rag/               # RAG pipeline (Chroma/Pinecone/pgvector, HybridRetriever, CrossEncoderReranker, IngestionPipeline)
│   ├── scheduler/         # Agent scheduling (intervals + cron)
│   ├── events/            # Event bus + triggers
│   ├── plugins/           # Plugin system (manager + base class)
│   ├── auth/              # Auth + orgs (Organization, Role, OrgMembership, ApiKey scopes, UsageTrackerAsync)
│   ├── workflows/         # Multi-step workflow engine
│   ├── teams/             # WorkflowDAG, PlannerAgent, TeamRunner, ResultAggregator
│   ├── marketplace/       # PackageManifest, MarketplaceRegistry (publish, search, install)
│   ├── embed/             # Embeddable widget + Python SDK
│   ├── templates/         # Pre-built agent templates
│   ├── tools/             # Built-in tools (calculator, weather, vision, docs, rag_search)
│   └── web/
│       └── app.py         # FastAPI web platform (UI + API)
├── frontend/              # React + ReactFlow Visual Builder (src/pages/VisualBuilder.tsx)
├── plugins/               # User plugins directory
├── examples/              # 20+ runnable demos
├── tests/                 # Unit tests
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## CI/CD

GitHub Actions workflows are included:

- **test.yml** — runs pytest + ruff on every push/PR (Python 3.11 + 3.12 matrix)
- **publish.yml** — builds and publishes to PyPI on GitHub release tags

---

## Roadmap

- [x] Core Agent SDK with tool calling
- [x] Simulation Sandbox with LLM-as-judge
- [x] Live monitoring dashboard
- [x] Governance Engine (budget, permissions, kill switch, audit)
- [x] Unified GovernedAgent class
- [x] WebSocket streaming (real-time token streaming)
- [x] RAG pipeline (document ingestion + vector search)
- [x] Agent scheduler (intervals + cron)
- [x] Event bus (pub/sub with triggers)
- [x] Plugin system (extensible tools + providers)
- [x] User authentication + usage tracking
- [x] A/B testing with statistical significance
- [x] Multi-step workflow engine
- [x] Analytics dashboard (cost trends, tool usage, leaderboards)
- [x] Multi-modal support (vision + document analysis)
- [x] Conversation branching (what-if exploration)
- [x] Agent Marketplace (publish, discover, install)
- [x] Package-based marketplace (manifest, registry, CLI)
- [x] Multi-agent teams (WorkflowDAG, PlannerAgent, TeamRunner)
- [x] RAG: Chroma/Pinecone/pgvector, HybridRetriever, CrossEncoderReranker
- [x] Auth: Organizations, roles, API key scopes, UsageTrackerAsync SQLite
- [x] Visual Builder (ReactFlow workflow editor)
- [x] /ws/monitor for tool and team events
- [x] Embeddable chat widget + SDK
- [x] Docker deployment
- [x] GitHub Actions CI/CD
- [x] Anthropic Claude provider (direct)
- [x] Ollama local model provider
- [ ] Agent-to-Agent mesh protocol
- [ ] Kubernetes deployment
- [ ] SOC2/HIPAA compliance templates

---

## Contributing

AgentOS is open source under the Apache 2.0 license. Contributions welcome!

```bash
git clone https://github.com/sukethrp/agentos.git
cd agentos
pip install -e ".[dev]"
pytest
```

---

## Star this repo if you believe AI agents should be tested before deployed!

Built by [Suketh Reddy Produtoor](https://github.com/sukethrp) | [LinkedIn](https://www.linkedin.com/in/sukethprodutoor/) | [X](https://x.com/SProdutoor45130) | [Reddit](https://www.reddit.com/user/SUKETH_11)
