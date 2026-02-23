"""Sales Qualification Agent Template."""

from agentos.core.agent import Agent
from agentos.core.tool import tool
from agentos.tools.http_tool import calculator_tool


@tool(description="Look up a lead or company in the CRM database")
def crm_lookup(company_name: str) -> str:
    crm = {
        "acme corp": "Acme Corp | Size: 500 employees | Industry: Manufacturing | Deal stage: Prospect | Budget: $100K-500K | Contact: john@acme.com",
        "techstart": "TechStart Inc | Size: 50 employees | Industry: SaaS | Deal stage: Demo scheduled | Budget: $10K-50K | Contact: sarah@techstart.io",
        "bigbank": "BigBank Financial | Size: 10,000 employees | Industry: Finance | Deal stage: Enterprise pilot | Budget: $500K+ | Contact: mike@bigbank.com",
    }
    return crm.get(
        company_name.lower(),
        f"No CRM record for '{company_name}'. New lead — needs qualification.",
    )


@tool(description="Search the product catalog for features, pricing, and plans")
def product_catalog(query: str) -> str:
    catalog = {
        "pricing": "AgentOS Pricing: Starter $99/mo (10 agents), Business $499/mo (50 agents), Enterprise $2000+/mo (unlimited). Annual: 20% discount.",
        "features": "Core features: Agent SDK, Simulation Sandbox, Live Dashboard, Governance Engine, Multi-model, Agent Teams, Memory, 50+ integrations.",
        "enterprise": "Enterprise: Custom deployment, SSO, SOC2 compliance, dedicated support, SLA, custom integrations, training. Contact sales.",
        "comparison": "vs LangChain: We have testing sandbox + governance (they don't). vs CrewAI: We have monitoring + governance (they don't). Only platform with all 4.",
    }
    query_lower = query.lower()
    for key, val in catalog.items():
        if key in query_lower:
            return val
    return "Product info: AgentOS is the operating system for AI agents. Visit agentos.dev for details."


def create_sales_agent(model: str = "gpt-4o-mini", **kwargs) -> Agent:
    return Agent(
        name="sales-agent",
        model=model,
        tools=[crm_lookup, product_catalog, calculator_tool()],
        system_prompt="""You are a professional sales qualification agent for AgentOS. Your goals:
1. Understand the prospect's needs and pain points
2. Look up their company in CRM for context
3. Match their needs to AgentOS features
4. Handle objections with facts and comparisons
5. Calculate ROI and pricing for their use case
6. Guide qualified leads toward a demo booking
7. Be consultative, not pushy — focus on solving their problem
8. Always ask about their current solution and what's not working""",
        **kwargs,
    )
