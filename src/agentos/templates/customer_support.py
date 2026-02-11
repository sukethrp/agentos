"""Customer Support Agent Template."""

from agentos.core.agent import Agent
from agentos.core.tool import tool
from agentos.tools.http_tool import calculator_tool


@tool(description="Search the knowledge base for product information, policies, and FAQs")
def knowledge_base(query: str) -> str:
    kb = {
        "refund": "Refund Policy: Full refund within 30 days of purchase. After 30 days, store credit only. Processing time: 5-7 business days.",
        "shipping": "Shipping: Standard (5-7 days, free over $50), Express (2-3 days, $9.99), Overnight ($24.99). International: 10-14 days.",
        "return": "Returns: Items must be unused and in original packaging. Start a return at account.example.com/returns. Print prepaid label.",
        "account": "Account Issues: Reset password at example.com/reset. Contact support for locked accounts. Two-factor auth available in settings.",
        "pricing": "Plans: Free (basic features), Pro ($29/mo, advanced), Enterprise ($99/mo, unlimited). Annual discount: 20% off.",
    }
    query_lower = query.lower()
    for key, val in kb.items():
        if key in query_lower:
            return val
    return f"No specific KB article found for '{query}'. Suggest checking our help center at help.example.com"


@tool(description="Create or look up a support ticket")
def ticket_system(action: str, details: str = "") -> str:
    if "create" in action.lower():
        return f"Ticket #TK-{hash(details) % 10000:04d} created. Customer will receive email confirmation. Expected response: 24 hours."
    elif "status" in action.lower():
        return f"Ticket status: In Progress. Assigned to support team. Last update: 2 hours ago."
    return f"Ticket action '{action}' completed."


def create_customer_support_agent(model: str = "gpt-4o-mini", **kwargs) -> Agent:
    return Agent(
        name="support-agent",
        model=model,
        tools=[knowledge_base, ticket_system, calculator_tool()],
        system_prompt="""You are a friendly and professional customer support agent. Follow these rules:
1. Always greet the customer warmly
2. Listen to their issue carefully
3. Use the knowledge base to find accurate answers
4. If you can't resolve the issue, create a support ticket
5. Always end with "Is there anything else I can help with?"
6. Never make promises you can't keep
7. Be empathetic when customers are frustrated
8. Use the calculator for any pricing or refund calculations""",
        **kwargs,
    )