"""Code Review Agent Template."""

from agentos.core.agent import Agent
from agentos.core.tool import tool


@tool(
    description="Analyze code for bugs, security issues, and best practices. Pass the code as a string."
)
def code_analyzer(code: str) -> str:
    issues = []
    code_lower = code.lower()

    if "eval(" in code_lower:
        issues.append(
            "SECURITY: Use of eval() detected — potential code injection risk"
        )
    if "exec(" in code_lower:
        issues.append(
            "SECURITY: Use of exec() detected — potential code execution risk"
        )
    if "password" in code_lower and ("=" in code_lower):
        issues.append("SECURITY: Hardcoded password detected")
    if "api_key" in code_lower and ("=" in code_lower):
        issues.append("SECURITY: Possible hardcoded API key")
    if "import os" in code_lower and "system(" in code_lower:
        issues.append("SECURITY: os.system() call — potential command injection")
    if "sql" in code_lower and "+" in code_lower:
        issues.append("SECURITY: Possible SQL injection via string concatenation")

    if "except:" in code and "except Exception" not in code:
        issues.append("PRACTICE: Bare except clause — catch specific exceptions")
    if "# TODO" in code or "# FIXME" in code:
        issues.append("PRACTICE: TODO/FIXME comments found — address before merge")
    if "print(" in code_lower and "debug" in code_lower:
        issues.append("PRACTICE: Debug print statements — remove before production")

    lines = code.strip().split("\n")
    if len(lines) > 50:
        issues.append(
            f"QUALITY: Function is {len(lines)} lines — consider breaking into smaller functions"
        )
    long_lines = [i + 1 for i, line in enumerate(lines) if len(line) > 120]
    if long_lines:
        issues.append(f"QUALITY: Lines exceeding 120 chars: {long_lines[:5]}")

    if not issues:
        return "No issues detected. Code looks clean!"

    return "Code Review Results:\n" + "\n".join(issues)


def create_code_reviewer_agent(model: str = "gpt-4o-mini", **kwargs) -> Agent:
    return Agent(
        name="code-reviewer",
        model=model,
        tools=[code_analyzer],
        system_prompt="""You are an expert code reviewer. When reviewing code:
1. First use the code_analyzer tool to check for automated issues
2. Then provide your own expert analysis covering:
   - Logic errors and edge cases
   - Performance concerns
   - Readability and maintainability
   - Security vulnerabilities
   - Suggested improvements
3. Rate the code: Critical / Needs Work / Acceptable / Excellent
4. Be constructive — explain WHY something is an issue and HOW to fix it
5. Praise good patterns you see — positive feedback matters""",
        **kwargs,
    )
