"""Multi-modal demo — image analysis and document reading with AgentOS.

This demo shows how to:
1. Analyze an image from a URL using the Vision tool
2. Read and summarize a text document using the Document tool
3. Create an agent with both vision and document tools
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from agentos.core.agent import Agent
from agentos.core.multimodal import analyze_image, read_document, extract_text_from_pdf
from agentos.tools.vision_tool import vision_tool
from agentos.tools.document_tool import document_reader_tool, document_qa_tool


def demo_image_analysis():
    """Demo 1: Analyze a public image using the Vision API directly."""
    print("=" * 60)
    print("Demo 1: Direct Image Analysis")
    print("=" * 60)

    # Use a well-known public-domain image
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"

    print(f"\nAnalyzing image: {image_url}")
    print("-" * 40)

    result = analyze_image(
        image_path_or_url=image_url,
        prompt="Describe this image in detail. What do you see?",
    )
    print(result)
    print()


def demo_document_reading():
    """Demo 2: Read and analyze a text document."""
    print("=" * 60)
    print("Demo 2: Document Reading")
    print("=" * 60)

    # Create a sample document
    sample_doc = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, prefix="agentos_demo_"
    )
    sample_doc.write("""# AgentOS Architecture Overview

## Core Components

### Agent Runtime
The Agent Runtime is the heart of AgentOS. It manages the lifecycle of AI agents,
handles tool execution, and coordinates with LLM providers. Each agent runs in an
isolated context with its own memory, tools, and configuration.

### Memory System
AgentOS implements a hierarchical memory system:
- **Short-term memory**: Recent conversation history (last 20 exchanges)
- **Long-term memory**: Extracted facts and user preferences
- **Working memory**: Current task context and intermediate results

### Tool Framework
Tools are the bridge between agents and the external world. AgentOS provides:
- Built-in tools: calculator, weather, web search, news
- Vision tools: image analysis via GPT-4o
- Document tools: read and analyze text, markdown, and PDF files
- Custom tools: define your own with the @tool decorator
- HTTP tools: call any REST API

### Governance Engine
The governance layer ensures agents operate safely:
- Budget guards: prevent cost overruns
- Permission guards: control which tools agents can use
- Audit logging: complete trail of all agent actions
- Kill switch: emergency stop for runaway agents

## Performance Metrics
- Average response time: 1.2 seconds
- Tool call overhead: 50ms per call
- Memory retrieval: O(1) for facts, O(n) for search
- Cost efficiency: $0.002 per typical query

## Deployment
AgentOS supports multiple deployment options:
- Local development with `python examples/run_web_builder.py`
- Docker deployment with `docker-compose up -d`
- Cloud deployment via CI/CD with GitHub Actions
""")
    sample_doc.close()

    print(f"\nReading document: {sample_doc.name}")
    print("-" * 40)

    content = read_document(sample_doc.name)
    print(f"Document length: {len(content):,} characters")
    print(f"Preview:\n{content[:200]}...")
    print()

    # Clean up
    os.unlink(sample_doc.name)
    return sample_doc.name


def demo_agent_with_vision():
    """Demo 3: Agent with vision and document tools."""
    print("=" * 60)
    print("Demo 3: Agent with Vision + Document Tools")
    print("=" * 60)

    agent = Agent(
        name="multimodal-agent",
        model="gpt-4o",
        tools=[vision_tool(), document_reader_tool(), document_qa_tool()],
        system_prompt=(
            "You are a multi-modal AI assistant. You can analyze images and read "
            "documents. When asked about visual content, use the analyze_image tool. "
            "When asked about documents, use read_document or analyze_document. "
            "Provide detailed, helpful responses."
        ),
    )

    # Ask about an image
    print("\nAsking agent to analyze an image from URL...")
    print("-" * 40)

    msg = agent.run(
        "Please analyze this image and describe what you see: "
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
        "PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"
    )

    print(f"\nAgent response: {msg.content[:300]}...")
    total_cost = sum(e.cost_usd for e in agent.events)
    print(f"Total cost: ${total_cost:.4f}")
    print()


def demo_document_agent():
    """Demo 4: Agent that reads a document and answers questions."""
    print("=" * 60)
    print("Demo 4: Document Q&A Agent")
    print("=" * 60)

    # Create a sample document
    doc_path = Path(tempfile.gettempdir()) / "agentos_sample.txt"
    doc_path.write_text(
        "AgentOS Quarterly Report Q4 2025\n\n"
        "Revenue: $2.4M (up 35% QoQ)\n"
        "Active users: 12,500 (up 28%)\n"
        "Agents deployed: 45,000+\n"
        "API calls: 180M per month\n"
        "Uptime: 99.97%\n\n"
        "Key achievements:\n"
        "- Launched multi-model support (GPT-4o, Claude, Ollama)\n"
        "- Released governance engine with budget controls\n"
        "- Added agent A/B testing framework\n"
        "- Introduced workflow system for multi-step pipelines\n"
        "- Docker deployment for one-command setup\n\n"
        "Challenges:\n"
        "- Scaling embedding infrastructure for RAG\n"
        "- Managing costs for high-volume enterprise clients\n"
        "- Complex multi-agent orchestration scenarios\n\n"
        "Q1 2026 Goals:\n"
        "- Launch multi-modal support (images, PDFs)\n"
        "- Enterprise SSO and RBAC\n"
        "- Agent marketplace with revenue sharing\n"
        "- Real-time collaboration features\n"
    )

    agent = Agent(
        name="doc-analyst",
        model="gpt-4o-mini",
        tools=[document_reader_tool(), document_qa_tool()],
        system_prompt=(
            "You are a document analysis assistant. Use the document tools "
            "to read files and answer questions about their content. "
            "Be precise and cite specific data from the documents."
        ),
    )

    print(f"\nAsking agent to analyze: {doc_path}")
    print("-" * 40)

    msg = agent.run(
        f"Read the file at {doc_path} and tell me: "
        "What was the revenue, how many active users are there, "
        "and what are the Q1 2026 goals?"
    )
    print(f"\nAgent response:\n{msg.content}")
    total_cost = sum(e.cost_usd for e in agent.events)
    print(f"\nTotal cost: ${total_cost:.4f}")

    # Clean up
    doc_path.unlink(missing_ok=True)


if __name__ == "__main__":
    print("\n🤖 AgentOS Multi-modal Demo\n")

    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY in your .env file to run this demo.")
        print("   All features require an OpenAI API key.\n")
        sys.exit(1)

    # Run demos — choose which ones to run
    import argparse

    parser = argparse.ArgumentParser(description="AgentOS Multi-modal Demo")
    parser.add_argument(
        "--demo",
        choices=["image", "document", "agent-vision", "agent-doc", "all"],
        default="all",
        help="Which demo to run",
    )
    args = parser.parse_args()

    if args.demo in ("image", "all"):
        demo_image_analysis()

    if args.demo in ("document", "all"):
        demo_document_reading()

    if args.demo in ("agent-vision", "all"):
        demo_agent_with_vision()

    if args.demo in ("agent-doc", "all"):
        demo_document_agent()

    print("\n✅ Demo complete!")
