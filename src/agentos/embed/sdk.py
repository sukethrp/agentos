"""AgentOS Embed SDK — Python client for remote AgentOS instances.

Usage:
    from agentos.embed.sdk import AgentOSClient

    client = AgentOSClient(api_key="ak_...", base_url="http://localhost:8000")
    response = client.run("What is the weather?")
    print(response)

    # Streaming
    for token in client.stream("Tell me a story"):
        print(token, end="", flush=True)

    # List marketplace agents
    agents = client.list_agents()
"""

from __future__ import annotations

import json
from typing import Generator
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


class AgentOSClient:
    """Lightweight Python SDK for interacting with a remote AgentOS server.

    Uses only the standard library (``urllib``) so there are zero extra
    dependencies beyond Python itself.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        system_prompt: str = "You are a helpful assistant.",
        tools: list[str] | None = None,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.timeout = timeout

    # ── Helpers ──────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode()
        req = Request(url, data=body, headers=self._headers(), method="POST")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            error_body = e.read().decode() if e.fp else str(e)
            raise RuntimeError(f"AgentOS API error ({e.code}): {error_body}") from e
        except URLError as e:
            raise ConnectionError(
                f"Cannot reach AgentOS at {self.base_url}: {e}"
            ) from e

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items() if v)
            if qs:
                url += f"?{qs}"
        req = Request(url, headers=self._headers(), method="GET")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            error_body = e.read().decode() if e.fp else str(e)
            raise RuntimeError(f"AgentOS API error ({e.code}): {error_body}") from e
        except URLError as e:
            raise ConnectionError(
                f"Cannot reach AgentOS at {self.base_url}: {e}"
            ) from e

    # ── Core operations ──────────────────────────────────────

    def run(
        self,
        query: str,
        model: str | None = None,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
    ) -> str:
        """Send a query and return the full response text."""
        payload = {
            "query": query,
            "model": model or self.model,
            "system_prompt": system_prompt or self.system_prompt,
            "tools": tools if tools is not None else self.tools,
        }
        data = self._post("/api/run", payload)
        return data.get("response", data.get("message", str(data)))

    def run_full(
        self,
        query: str,
        model: str | None = None,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
    ) -> dict:
        """Send a query and return the full response dict (response, cost, tokens, etc.)."""
        payload = {
            "query": query,
            "model": model or self.model,
            "system_prompt": system_prompt or self.system_prompt,
            "tools": tools if tools is not None else self.tools,
        }
        return self._post("/api/run", payload)

    def stream(
        self,
        query: str,
        model: str | None = None,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
    ) -> Generator[str, None, None]:
        """Stream tokens from the agent via WebSocket.

        Falls back to a single HTTP request if ``websocket-client`` is not
        installed or the WebSocket connection fails.
        """
        ws_url = (
            self.base_url.replace("http://", "ws://").replace("https://", "wss://")
            + "/ws/chat"
        )
        msg = json.dumps(
            {
                "query": query,
                "model": model or self.model,
                "system_prompt": system_prompt or self.system_prompt,
                "tools": tools if tools is not None else self.tools,
            }
        )

        try:
            import websocket  # type: ignore[import-untyped]

            ws = websocket.create_connection(ws_url, timeout=self.timeout)
            ws.send(msg)
            while True:
                raw = ws.recv()
                if not raw:
                    break
                data = json.loads(raw)
                if data.get("type") == "token":
                    yield data["content"]
                elif data.get("type") == "done":
                    break
                elif data.get("type") == "error":
                    raise RuntimeError(data.get("message", "stream error"))
            ws.close()
        except ImportError:
            # No websocket-client — fall back to plain HTTP
            result = self.run(query, model, system_prompt, tools)
            yield result
        except Exception:
            # WebSocket failed — fall back to plain HTTP
            result = self.run(query, model, system_prompt, tools)
            yield result

    # ── Marketplace / discovery ───────────────────────────────

    def list_agents(self, category: str = "", query: str = "") -> list[dict]:
        """List agents from the marketplace."""
        data = self._get("/api/marketplace/list", {"category": category, "q": query})
        return data.get("agents", [])

    def search_agents(self, query: str, category: str = "") -> list[dict]:
        """Search marketplace agents."""
        data = self._get("/api/marketplace/search", {"q": query, "category": category})
        return data.get("agents", [])

    def install_agent(self, agent_id: str) -> dict:
        """Install a marketplace agent (bumps download count, returns config)."""
        return self._post(f"/api/marketplace/install/{agent_id}", {})

    # ── Utility ───────────────────────────────────────────────

    def health(self) -> bool:
        """Check whether the AgentOS server is reachable."""
        try:
            self._get("/docs")
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"AgentOSClient(base_url={self.base_url!r}, model={self.model!r})"
