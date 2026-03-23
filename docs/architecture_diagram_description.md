# AgentOS Architecture Diagram Description

This diagram presents AgentOS as a layered platform with clear boundaries and
data/control flow from user entrypoints down to model providers:

1. **AgentOS CLI (top layer)**  
   Entrypoint for operators and developers (`agentos serve`, `agentos init`,
   `agentos mcp`).

2. **Web Platform (FastAPI)**  
   Product-facing surface with Agent Builder, Chat, Dashboard, and Embed.

3. **Application Modules**  
   Core product capabilities: Agent SDK, Sandbox Testing, Monitor Events, and
   Governance (Budget + Auth).

4. **Core Engine**  
   Shared runtime primitives: Tool Calling, Streaming, Memory, and RAG.

5. **Provider Layer**  
   Model backends: OpenAI, Claude, Ollama, and Mock providers.

6. **MCP Server (bottom layer)**  
   Standards-based integration surface over JSON-RPC/stdio for external tools
   and clients (e.g., Claude Desktop, Cursor).

The visual style uses a dark theme to align with the landing page, with rounded
cards, high-contrast typography, and accent separators for readability.
