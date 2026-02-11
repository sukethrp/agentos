"""AgentOS Templates Demo — Pre-built agents ready to deploy."""

import sys
sys.path.insert(0, "src")

from agentos.templates import list_templates, load_template


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AgentOS — Template Library Demo")
    print("=" * 60)

    # Show all templates
    list_templates()

    # ── Customer Support ──
    print("\n" + "━" * 60)
    print("DEMO 1: Customer Support Agent")
    print("━" * 60)
    support = load_template("customer-support")
    support.run("I bought a product 2 weeks ago and want a refund. Order #12345.")

    # ── Sales Agent ──
    print("\n" + "━" * 60)
    print("DEMO 2: Sales Qualification Agent")
    print("━" * 60)
    sales = load_template("sales-agent")
    sales.run("Hi, I'm from Acme Corp. We're looking for an AI agent platform. What's your pricing?")

    # ── Research Assistant ──
    print("\n" + "━" * 60)
    print("DEMO 3: Research Assistant")
    print("━" * 60)
    researcher = load_template("research-assistant")
    researcher.run("What's the current weather in Tokyo and what's 15% of 250?")

    # ── Code Reviewer ──
    print("\n" + "━" * 60)
    print("DEMO 4: Code Review Agent")
    print("━" * 60)
    reviewer = load_template("code-reviewer")
    reviewer.run("""Review this code:
def login(username, password):
    query = "SELECT * FROM users WHERE name='" + username + "' AND pass='" + password + "'"
    result = eval(query)
    api_key = "sk-secret-key-12345"
    print("debug: logged in as", username)
    return result
""")