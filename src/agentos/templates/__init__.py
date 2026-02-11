"""AgentOS Templates — Pre-built agents ready to deploy.

Usage:
    from agentos.templates import list_templates, load_template

    # See all available templates
    list_templates()

    # Load and run a template
    agent = load_template("customer-support")
    agent.run("I need help with my order")
"""

from agentos.templates.customer_support import create_customer_support_agent
from agentos.templates.research_assistant import create_research_agent
from agentos.templates.sales_agent import create_sales_agent
from agentos.templates.code_reviewer import create_code_reviewer_agent

TEMPLATES = {
    "customer-support": {
        "name": "Customer Support Agent",
        "description": "Handles customer inquiries, complaints, and support tickets. Polite, helpful, and solution-oriented.",
        "creator": create_customer_support_agent,
        "category": "support",
        "tools": ["knowledge_base", "ticket_system", "calculator"],
    },
    "research-assistant": {
        "name": "Research Assistant",
        "description": "Researches topics, gathers data, analyzes information, and produces reports.",
        "creator": create_research_agent,
        "category": "research",
        "tools": ["web_search", "news_search", "calculator", "weather"],
    },
    "sales-agent": {
        "name": "Sales Qualification Agent",
        "description": "Qualifies leads, answers product questions, handles objections, and books meetings.",
        "creator": create_sales_agent,
        "category": "sales",
        "tools": ["crm_lookup", "product_catalog", "calculator"],
    },
    "code-reviewer": {
        "name": "Code Review Agent",
        "description": "Reviews code for bugs, security issues, performance problems, and best practices.",
        "creator": create_code_reviewer_agent,
        "category": "engineering",
        "tools": ["code_analyzer"],
    },
}


def list_templates():
    """Print all available agent templates."""
    print(f"\n{'='*60}")
    print(f"📦 AgentOS Template Library")
    print(f"{'='*60}")
    for tid, info in TEMPLATES.items():
        print(f"\n   [{tid}]")
        print(f"   {info['name']}")
        print(f"   {info['description']}")
        print(f"   Category: {info['category']} | Tools: {', '.join(info['tools'])}")
    print(f"\n{'='*60}")
    print(f"   Usage: agent = load_template('customer-support')")
    print(f"{'='*60}")


def load_template(template_id: str, model: str = "gpt-4o-mini", **kwargs):
    """Load a pre-built agent template."""
    if template_id not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Template '{template_id}' not found. Available: {available}")

    creator = TEMPLATES[template_id]["creator"]
    return creator(model=model, **kwargs)