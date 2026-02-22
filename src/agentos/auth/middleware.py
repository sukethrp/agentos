from __future__ import annotations
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from agentos.auth.org_store import check_scope


class ScopeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")
        if api_key:
            agent_id = request.path_params.get("agent_id") or request.query_params.get("agent_id")
            tool_name = request.query_params.get("tool_name")
            if agent_id or tool_name:
                if not check_scope(api_key, agent_id, tool_name):
                    from starlette.responses import JSONResponse
                    return JSONResponse({"detail": "API key not scoped for this agent or tool"}, status_code=403)
        return await call_next(request)
