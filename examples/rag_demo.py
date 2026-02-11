"""AgentOS RAG Demo — Retrieval Augmented Generation pipeline in action."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")

from agentos.rag import RAGPipeline
from agentos.tools.rag_tool import create_rag_tool
from agentos.core.agent import Agent


# ── Create sample documents ──

SAMPLE_DOCS = {
    "product_guide.md": """\
# AgentOS Product Guide

## Getting Started
AgentOS is an AI agent platform that lets you build, deploy, and manage
intelligent agents. Agents can use tools, maintain memory, and work in teams.

## Installation
Install AgentOS via pip:
```
pip install agentos-platform
```

## Creating Your First Agent
```python
from agentos.core.agent import Agent
agent = Agent(name="my-agent", system_prompt="You are helpful.")
agent.run("Hello!")
```

## Configuration
- **Model**: Choose from GPT-4o, GPT-4o-mini, Claude Sonnet, Claude Haiku
- **Temperature**: 0.0 (focused) to 1.0 (creative)
- **Max Iterations**: Default is 10 tool-use loops
- **Memory**: Agents remember past conversations automatically

## Tools
Agents can use built-in tools:
- **Calculator**: Evaluate math expressions
- **Weather**: Get current weather for major cities
- **Web Search**: Search the internet via DuckDuckGo
- **HTTP**: Make REST API calls

## Budget & Governance
Set spending limits per agent per day. The governance module tracks costs
and enforces budgets automatically.
""",

    "faq.txt": """\
Frequently Asked Questions — AgentOS

Q: What models does AgentOS support?
A: AgentOS supports OpenAI models (GPT-4o, GPT-4o-mini), Anthropic models
(Claude Sonnet, Claude Haiku, Claude Opus), and local models via Ollama
(Llama 3.1, Mistral, Gemma 2).

Q: How do I add custom tools?
A: Use the @tool decorator:
    from agentos.core.tool import tool
    @tool(description="My custom tool")
    def my_tool(param: str) -> str:
        return "result"

Q: Is there a web UI?
A: Yes! Run `python examples/run_web_builder.py` to launch the web platform
at http://localhost:8000. It includes an agent builder, chat interface,
template library, and monitoring dashboard.

Q: How does memory work?
A: Agents automatically store conversation history and extract key facts.
The Memory class maintains a sliding window of recent exchanges and a
separate fact store for long-term knowledge.

Q: Can agents work together?
A: Yes. Use the Team and Orchestrator classes to create multi-agent workflows
where agents can delegate tasks to each other.

Q: What about security?
A: AgentOS includes a governance module with budget limits, permission
controls, guardrails for content filtering, and audit logging.

Q: How much does it cost?
A: AgentOS is open source. You pay only for the LLM API calls. GPT-4o-mini
costs about $0.15 per million input tokens and $0.60 per million output tokens.
""",

    "api_reference.md": """\
# AgentOS API Reference

## Agent Class
```python
class Agent:
    def __init__(self, name, model, tools, system_prompt, max_iterations, temperature, memory)
    def run(self, user_input: str, stream: bool = False) -> Message
```

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| name | str | "agent" | Agent identifier |
| model | str | "gpt-4o-mini" | LLM model to use |
| tools | list[Tool] | [] | Available tools |
| system_prompt | str | "You are helpful" | System instructions |
| max_iterations | int | 10 | Max tool-use loops |
| temperature | float | 0.7 | Creativity (0-1) |
| memory | Memory | None | Conversation memory |

## Tool Decorator
```python
@tool(name="my_tool", description="Does something")
def my_tool(param: str) -> str:
    return "result"
```

## RAG Pipeline
```python
from agentos.rag import RAGPipeline

rag = RAGPipeline(chunk_size=512, chunk_overlap=64, top_k=5)
rag.ingest("path/to/file.pdf")
result = rag.query("search question")
print(result.context)
```

## Memory
```python
from agentos.core.memory import Memory

memory = Memory(max_history=20)
memory.add_exchange("user question", "agent answer")
messages = memory.build_messages("system prompt", "new question")
```
""",
}


if __name__ == "__main__":
    print("=" * 60)
    print("🔍 AgentOS — RAG Pipeline Demo")
    print("=" * 60)

    # ── Step 1: Create temp files with sample content ──
    tmpdir = tempfile.mkdtemp(prefix="agentos_rag_")
    print(f"\n📁 Creating sample documents in {tmpdir}")

    for filename, content in SAMPLE_DOCS.items():
        p = Path(tmpdir) / filename
        p.write_text(content)
        print(f"   ✓ {filename} ({len(content):,} chars)")

    # ── Step 2: Initialize the RAG pipeline ──
    print("\n⚙️  Initializing RAG pipeline...")
    rag = RAGPipeline(
        chunk_size=512,
        chunk_overlap=64,
        top_k=3,
    )

    # ── Step 3: Ingest documents ──
    print("\n📥 Ingesting documents...")
    total = rag.ingest_directory(tmpdir, extensions=[".md", ".txt"])
    print(f"   ✓ Ingested {total} chunks from {len(rag.ingested_files)} files")
    print(f"   ✓ Vector store contains {rag.num_chunks} vectors")

    # ── Step 4: Direct RAG queries ──
    print("\n" + "━" * 60)
    print("DEMO 1: Direct RAG Queries")
    print("━" * 60)

    queries = [
        "What models does AgentOS support?",
        "How do I create a custom tool?",
        "What are the Agent class parameters?",
    ]

    for q in queries:
        print(f"\n🔎 Query: {q}")
        result = rag.query(q)
        print(f"   Found {len(result.results)} results")
        if result.results:
            top = result.results[0]
            print(f"   Top match (score: {top.score:.3f}, source: {top.metadata.get('filename', '?')}):")
            print(f"   {top.text[:150]}...")

    # ── Step 5: Agent with RAG tool ──
    print("\n" + "━" * 60)
    print("DEMO 2: Agent with RAG Tool")
    print("━" * 60)

    rag_tool = create_rag_tool(rag, top_k=3)
    agent = Agent(
        name="rag-agent",
        model="gpt-4o-mini",
        tools=[rag_tool],
        system_prompt=(
            "You are a helpful assistant for AgentOS. "
            "Use the search_documents tool to find relevant information from "
            "the knowledge base before answering questions. "
            "Always cite which document your answer comes from."
        ),
        temperature=0.3,
    )

    agent.run("How do I add custom tools to my agent? Give me a code example.")

    # ── Step 6: Second query ──
    print()
    agent.run("What security features does AgentOS have?")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("📊 RAG Pipeline Summary")
    print("=" * 60)
    print(f"   Documents ingested: {len(rag.ingested_files)}")
    print(f"   Total chunks:       {rag.num_chunks}")
    print(f"   Agent tool calls:   {sum(1 for e in agent.events if e.event_type == 'tool_call')}")
    print("=" * 60)
