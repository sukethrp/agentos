"""GitHub Plugin for AgentOS — adds tools to interact with the GitHub API.

Class-style plugin that provides:
    - list_repos: list repositories for a user/org
    - create_issue: open an issue on a repo
    - get_pull_requests: list PRs for a repo

Requires:
    GITHUB_TOKEN in .env (or environment variable)

Uses only the standard library (urllib) — no httpx/requests required.
"""

import json
import os
import sys
import urllib.request
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentos.core.tool import Tool
from agentos.plugins.base import BasePlugin, PluginContext

# ── Helpers ──

def _github_request(endpoint: str, method: str = "GET", body: dict | None = None) -> dict | list:
    """Make an authenticated request to the GitHub API."""
    token = os.getenv("GITHUB_TOKEN", "")
    url = f"https://api.github.com{endpoint}"

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "AgentOS-Plugin",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data = json.dumps(body).encode() if body else None

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        return {"error": f"HTTP {e.code}: {e.reason}", "detail": error_body[:500]}
    except Exception as e:
        return {"error": str(e)}


# ── Tools ──

def _list_repos(username: str = "") -> str:
    """List GitHub repositories for a user or the authenticated user.

    If username is empty, lists repos of the authenticated user (requires GITHUB_TOKEN).
    """
    endpoint = f"/users/{username}/repos?per_page=10&sort=updated" if username else "/user/repos?per_page=10&sort=updated"
    result = _github_request(endpoint)

    if isinstance(result, dict) and "error" in result:
        return f"GitHub API error: {result['error']}"

    if not isinstance(result, list):
        return f"Unexpected response: {str(result)[:300]}"

    lines = []
    for repo in result[:10]:
        name = repo.get("full_name", "?")
        desc = repo.get("description", "") or "No description"
        stars = repo.get("stargazers_count", 0)
        lang = repo.get("language", "?")
        lines.append(f"  - {name} ({lang}, ⭐ {stars}): {desc[:80]}")

    if not lines:
        return f"No repositories found for {'user ' + username if username else 'authenticated user'}."

    header = f"Repositories for {'user ' + username if username else 'authenticated user'}:"
    return header + "\n" + "\n".join(lines)


def _create_issue(repo: str, title: str, body: str = "") -> str:
    """Create a GitHub issue on a repository.

    Args:
        repo: Full repo name like 'owner/repo'.
        title: Issue title.
        body: Issue body/description (optional).
    """
    if "/" not in repo:
        return "Error: repo must be in 'owner/repo' format."

    token = os.getenv("GITHUB_TOKEN", "")
    if not token:
        return "Error: GITHUB_TOKEN is required to create issues. Set it in .env."

    result = _github_request(
        f"/repos/{repo}/issues",
        method="POST",
        body={"title": title, "body": body},
    )

    if isinstance(result, dict) and "error" in result:
        return f"Failed to create issue: {result['error']}"

    number = result.get("number", "?")
    url = result.get("html_url", "")
    return f"Issue #{number} created: {title}\nURL: {url}"


def _get_pull_requests(repo: str, state: str = "open") -> str:
    """List pull requests for a GitHub repository.

    Args:
        repo: Full repo name like 'owner/repo'.
        state: Filter by state — 'open', 'closed', or 'all'.
    """
    if "/" not in repo:
        return "Error: repo must be in 'owner/repo' format."

    result = _github_request(f"/repos/{repo}/pulls?state={state}&per_page=10")

    if isinstance(result, dict) and "error" in result:
        return f"GitHub API error: {result['error']}"

    if not isinstance(result, list):
        return f"Unexpected response: {str(result)[:300]}"

    if not result:
        return f"No {state} pull requests found for {repo}."

    lines = []
    for pr in result[:10]:
        number = pr.get("number", "?")
        title = pr.get("title", "?")
        user = pr.get("user", {}).get("login", "?")
        pr_state = pr.get("state", "?")
        lines.append(f"  PR #{number} [{pr_state}] by {user}: {title[:80]}")

    return f"Pull requests for {repo} ({state}):\n" + "\n".join(lines)


# ── Plugin class ──

class GitHubPlugin(BasePlugin):
    name = "github"
    version = "0.1.0"
    description = "GitHub integration — list repos, create issues, get pull requests."
    author = "AgentOS Team"

    def on_load(self, ctx: PluginContext) -> None:
        self.register_tool(ctx, Tool(
            fn=_list_repos,
            name="list_repos",
            description=(
                "List GitHub repositories for a user. "
                "Parameters: username (str, optional — omit for authenticated user). "
                "Returns up to 10 repos sorted by last update."
            ),
        ))
        self.register_tool(ctx, Tool(
            fn=_create_issue,
            name="create_issue",
            description=(
                "Create a GitHub issue. "
                "Parameters: repo (str, 'owner/repo'), title (str), body (str, optional). "
                "Requires GITHUB_TOKEN."
            ),
        ))
        self.register_tool(ctx, Tool(
            fn=_get_pull_requests,
            name="get_pull_requests",
            description=(
                "List pull requests for a GitHub repository. "
                "Parameters: repo (str, 'owner/repo'), state (str: 'open'|'closed'|'all'). "
                "Returns up to 10 PRs."
            ),
        ))

    def on_unload(self) -> None:
        pass  # no persistent resources to clean up
