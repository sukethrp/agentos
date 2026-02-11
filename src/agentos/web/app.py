"""AgentOS Web UI — Visual Agent Builder + Marketplace + Dashboard."""

from __future__ import annotations
import asyncio
import json
import queue
import threading
import time as _time
from collections import defaultdict
import os
import tempfile
import uuid as _uuid
from pathlib import Path as _Path
from fastapi import Depends, FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from agentos.monitor.store import store
from agentos.core.types import AgentEvent
from agentos.tools import get_builtin_tools
from agentos.core.multimodal import analyze_image, read_document, encode_image
from agentos.core.branching import ConversationTree, get_or_create_tree, get_tree, list_trees
from agentos.scheduler import AgentScheduler
from agentos.events import event_bus, WebhookTrigger
from agentos.auth import User, create_user, get_current_user, get_user_by_email
from agentos.auth.usage import usage_tracker
from agentos.core.ab_testing import ABTest
from agentos.marketplace import get_marketplace_store
from agentos.embed.widget import generate_widget, generate_widget_js, generate_snippet

load_dotenv()

app = FastAPI(title="AgentOS Platform", version="0.3.0")

# CORS — allow cross-origin embedding of the chat widget
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global scheduler instance
_scheduler = AgentScheduler(max_concurrent=3)
_scheduler.start()

# Global webhook trigger (passive — fires when /api/webhook/ is hit)
_webhook_trigger = WebhookTrigger(name="web-webhook")
_webhook_trigger.start()

# Upload directory for files
_UPLOAD_DIR = _Path(tempfile.gettempdir()) / "agentos_uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class RunRequest(BaseModel):
    name: str = "web-agent"
    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a helpful assistant."
    query: str = ""
    tools: list[str] = []
    temperature: float = 0.7
    budget_limit: float = 5.0


class ChatRequest(BaseModel):
    query: str
    agent_id: str = "default"


class RegisterRequest(BaseModel):
    email: str
    name: str


class LoginRequest(BaseModel):
    email: str


class ABTestAgentConfig(BaseModel):
    name: str = "agent"
    model: str = "gpt-4o-mini"
    system_prompt: str
    temperature: float = 0.7
    tools: list[str] = []


class ABTestRequest(BaseModel):
    agent_a: ABTestAgentConfig
    agent_b: ABTestAgentConfig
    queries: list[str]
    num_runs: int = 5


# ── WebSocket Chat (streaming) ──

@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """WebSocket endpoint that streams tokens in real-time."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        query = data.get("query", "").strip()
        if not query:
            await websocket.send_json({"type": "error", "message": "Empty query"})
            return

        name = data.get("name", "chat-agent")
        model = data.get("model", "gpt-4o-mini")
        system_prompt = data.get("system_prompt", "You are a helpful assistant.")
        tools_list = data.get("tools", [])
        temperature = float(data.get("temperature", 0.7))

        from agentos.core.agent import Agent

        available_tools = get_builtin_tools()
        agent_tools = [available_tools[t] for t in tools_list if t in available_tools]

        agent = Agent(
            name=name,
            model=model,
            tools=agent_tools,
            system_prompt=system_prompt,
            temperature=temperature,
        )

        chunk_queue: queue.Queue = queue.Queue()

        def run_agent_stream():
            try:
                for chunk in agent.run(query, stream=True):
                    chunk_queue.put(("chunk", chunk))
                chunk_queue.put(("done", None))
            except Exception as e:
                chunk_queue.put(("error", str(e)))

        stream_thread = threading.Thread(target=run_agent_stream)
        stream_thread.start()

        full_response = []
        while True:
            try:
                msg_type, data = chunk_queue.get(timeout=0.05)
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            if msg_type == "error":
                await websocket.send_json({"type": "error", "message": data})
                return
            if msg_type == "done":
                break

            if isinstance(data, str):
                full_response.append(data)
                await websocket.send_json({"type": "token", "content": data})

        stream_thread.join(timeout=1.0)

        response_text = "".join(full_response)
        cost = sum(e.cost_usd for e in agent.events)
        tokens = sum(e.tokens_used for e in agent.events)
        tools_used = [e.data.get("tool", "") for e in agent.events if e.event_type == "tool_call"]

        for e in agent.events:
            store.log_event(e)

        await websocket.send_json({
            "type": "done",
            "response": response_text,
            "cost": round(cost, 6),
            "tokens": tokens,
            "tools_used": tools_used,
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ── API Endpoints ──

@app.get("/")
def home():
    return HTMLResponse(WEB_UI_HTML)


@app.get("/api/overview")
def overview():
    return store.get_overview()


@app.get("/api/events")
def get_events(limit: int = 50):
    return store.get_events(limit=limit)


@app.get("/api/templates")
def get_templates():
    return {
        "templates": [
            {"id": "customer-support", "name": "Customer Support", "description": "Handle inquiries, complaints, tickets", "category": "support", "icon": "🎧"},
            {"id": "research-assistant", "name": "Research Assistant", "description": "Research topics, gather data, analyze", "category": "research", "icon": "🔬"},
            {"id": "sales-agent", "name": "Sales Agent", "description": "Qualify leads, answer product questions", "category": "sales", "icon": "💼"},
            {"id": "code-reviewer", "name": "Code Reviewer", "description": "Review code for bugs and security", "category": "engineering", "icon": "👨‍💻"},
            {"id": "custom", "name": "Custom Agent", "description": "Build your own from scratch", "category": "custom", "icon": "🛠️"},
        ]
    }


@app.post("/api/run")
def run_agent(req: RunRequest, current_user: User = Depends(get_current_user)):
    """Run an agent from the web UI."""
    from agentos.core.agent import Agent

    available_tools = get_builtin_tools()
    agent_tools = [available_tools[t] for t in req.tools if t in available_tools]

    agent = Agent(
        name=req.name,
        model=req.model,
        tools=agent_tools,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
    )

    # Capture output
    import io, sys
    old = sys.stdout
    sys.stdout = io.StringIO()
    msg = agent.run(req.query)
    terminal_output = sys.stdout.getvalue()
    sys.stdout = old

    # Log events to store
    for e in agent.events:
        store.log_event(e)

    cost = sum(e.cost_usd for e in agent.events)
    tokens = sum(e.tokens_used for e in agent.events)
    tools_used = [e.data.get("tool","") for e in agent.events if e.event_type == "tool_call"]

    # Track per-user usage
    usage_tracker.log_usage(current_user.id, tokens=tokens, cost=cost)

    return {
        "response": msg.content,
        "cost": round(cost, 6),
        "tokens": tokens,
        "tools_used": tools_used,
        "terminal": terminal_output,
    }


# ── Scheduler API ──

class ScheduleRequest(BaseModel):
    agent_name: str = "scheduled-agent"
    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a helpful assistant."
    query: str
    tools: list[str] = []
    interval: str = ""     # e.g. "5m", "1h", "30s"
    cron: str = ""         # e.g. "0 9 * * *"
    max_executions: int = 0


@app.get("/api/scheduler/jobs")
def list_scheduler_jobs():
    """List all scheduled jobs."""
    return {
        "overview": _scheduler.get_overview(),
        "jobs": [j.to_dict() for j in _scheduler.list_jobs()],
    }


@app.post("/api/scheduler/create")
def create_scheduler_job(req: ScheduleRequest):
    """Create a new scheduled job."""
    available_tools = get_builtin_tools()
    agent_tools = [available_tools[t] for t in req.tools if t in available_tools]

    try:
        job = _scheduler.schedule_from_config(
            agent_name=req.agent_name,
            model=req.model,
            query=req.query,
            tools=agent_tools,
            system_prompt=req.system_prompt,
            interval=req.interval,
            cron=req.cron,
            max_executions=req.max_executions,
        )
        return {"status": "created", "job": job.to_dict()}
    except ValueError as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/scheduler/delete/{job_id}")
def delete_scheduler_job(job_id: str):
    """Delete a scheduled job."""
    if _scheduler.delete_job(job_id):
        return {"status": "deleted", "job_id": job_id}
    return {"status": "error", "message": f"Job {job_id} not found"}


@app.post("/api/scheduler/pause/{job_id}")
def pause_scheduler_job(job_id: str):
    """Pause a scheduled job."""
    if _scheduler.pause_job(job_id):
        return {"status": "paused", "job_id": job_id}
    return {"status": "error", "message": f"Cannot pause job {job_id}"}


@app.post("/api/scheduler/resume/{job_id}")
def resume_scheduler_job(job_id: str):
    """Resume a paused job."""
    if _scheduler.resume_job(job_id):
        return {"status": "resumed", "job_id": job_id}
    return {"status": "error", "message": f"Cannot resume job {job_id}"}


# ── Auth API ──

@app.post("/api/auth/register")
def register_user(req: RegisterRequest):
    """Create a new user and return an API key."""
    try:
        user = create_user(email=req.email, name=req.name)
        return {
            "status": "created",
            "user": user,
            "api_key": user.api_key,
        }
    except ValueError as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/auth/login")
def login_user(req: LoginRequest):
    """Login by email — returns existing API key."""
    user = get_user_by_email(req.email)
    if not user:
        return {"status": "error", "message": "User not found"}
    return {
        "status": "ok",
        "user": user,
        "api_key": user.api_key,
    }


@app.get("/api/auth/usage")
def get_my_usage(period: str = "month", current_user: User = Depends(get_current_user)):
    """Return usage stats for the current user."""
    total = usage_tracker.get_usage(current_user.id).to_dict()
    window = usage_tracker.get_usage_by_period(current_user.id, period=period).to_dict()
    return {
        "status": "ok",
        "user_id": current_user.id,
        "period": period,
        "total": total,
        "window": window,
    }


@app.post("/api/ab-test")
def run_ab_test(req: ABTestRequest, current_user: User = Depends(get_current_user)):
    """Run an A/B test between two agent configs using the Sandbox judge."""
    from agentos.core.agent import Agent

    queries = [q.strip() for q in req.queries if q.strip()]
    if not queries:
        return {"status": "error", "message": "At least one non-empty query is required"}

    available_tools = get_builtin_tools()

    def build_agent(cfg: ABTestAgentConfig) -> Agent:
        agent_tools = [available_tools[t] for t in cfg.tools if t in available_tools]
        return Agent(
            name=cfg.name,
            model=cfg.model,
            tools=agent_tools,
            system_prompt=cfg.system_prompt,
            temperature=cfg.temperature,
        )

    agent_a = build_agent(req.agent_a)
    agent_b = build_agent(req.agent_b)

    tester = ABTest(agent_a, agent_b)
    report = tester.run_test(queries, num_runs=req.num_runs)

    return {"status": "ok", "report": report.model_dump()}


# ── Multi-modal / File Upload API ──

ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
ALLOWED_DOC_EXTS = {".txt", ".md", ".markdown", ".pdf", ".csv", ".json", ".log", ".rst"}
ALLOWED_EXTS = ALLOWED_IMAGE_EXTS | ALLOWED_DOC_EXTS


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload an image or document for analysis."""
    if not file.filename:
        return JSONResponse({"status": "error", "message": "No file provided"}, 400)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        return JSONResponse({
            "status": "error",
            "message": f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTS))}",
        }, 400)

    # Save to temp directory with a unique name
    unique_name = f"{_uuid.uuid4().hex[:8]}_{file.filename}"
    dest = _UPLOAD_DIR / unique_name
    content = await file.read()
    dest.write_bytes(content)

    file_type = "image" if ext in ALLOWED_IMAGE_EXTS else "document"

    return {
        "status": "uploaded",
        "file_path": str(dest),
        "file_name": file.filename,
        "file_type": file_type,
        "size_bytes": len(content),
    }


class AnalyzeFileRequest(BaseModel):
    file_path: str
    question: str = ""
    model: str = "gpt-4o"


@app.post("/api/analyze-file")
def analyze_uploaded_file(req: AnalyzeFileRequest):
    """Analyze an uploaded image or document.

    For images: uses OpenAI Vision API.
    For documents: reads content and uses an agent to answer the question.
    Also accepts image URLs directly.
    """
    question = req.question.strip() or "Describe or summarize this content in detail."

    # Handle URL-based image analysis
    from agentos.core.multimodal import is_url
    if is_url(req.file_path):
        try:
            result = analyze_image(
                image_path_or_url=req.file_path,
                prompt=question,
                model=req.model,
            )
            return {"status": "ok", "type": "image", "analysis": result}
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, 500)

    path = _Path(req.file_path)
    if not path.exists():
        return JSONResponse({"status": "error", "message": "File not found"}, 404)

    ext = path.suffix.lower()

    if ext in ALLOWED_IMAGE_EXTS:
        # Image analysis via Vision API
        try:
            result = analyze_image(
                image_path_or_url=str(path),
                prompt=question,
                model=req.model,
            )
            return {"status": "ok", "type": "image", "analysis": result}
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, 500)
    else:
        # Document analysis — read content, then use an LLM to answer
        try:
            content = read_document(str(path), max_chars=30_000)
            from agentos.core.agent import Agent
            agent = Agent(
                name="doc-analyzer",
                model=req.model if req.model != "gpt-4o" else "gpt-4o-mini",
                system_prompt=(
                    "You are a document analysis assistant. The user has uploaded a document "
                    "and wants you to analyze it. Answer their question based solely on the "
                    "document content provided."
                ),
            )
            import io, sys
            old = sys.stdout
            sys.stdout = io.StringIO()
            msg = agent.run(
                f"Here is the document content:\n\n---\n{content}\n---\n\n"
                f"Question: {question}"
            )
            sys.stdout = old
            for e in agent.events:
                store.log_event(e)
            cost = sum(e.cost_usd for e in agent.events)
            tokens = sum(e.tokens_used for e in agent.events)
            return {
                "status": "ok",
                "type": "document",
                "analysis": msg.content,
                "cost": round(cost, 6),
                "tokens": tokens,
            }
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, 500)


# ── Conversation Branching API ──


class BranchRequest(BaseModel):
    tree_id: str
    at_message_index: int | None = None
    label: str = ""
    source_branch_id: str | None = None


class BranchMessageRequest(BaseModel):
    tree_id: str
    branch_id: str | None = None
    role: str = "user"
    content: str = ""


class CompareBranchesRequest(BaseModel):
    tree_id: str
    branch_a: str
    branch_b: str


class MergeBranchesRequest(BaseModel):
    tree_id: str
    branch_a: str
    branch_b: str
    label: str = "merged"


@app.post("/api/branch")
def create_branch(req: BranchRequest):
    """Create a new conversation branch."""
    tree = get_tree(req.tree_id)
    if not tree:
        return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
    try:
        new_id = tree.branch(
            at_message_index=req.at_message_index,
            label=req.label,
            source_branch_id=req.source_branch_id,
        )
        return {"status": "created", "branch_id": new_id, "tree": tree.to_dict()}
    except (KeyError, IndexError) as e:
        return JSONResponse({"status": "error", "message": str(e)}, 400)


@app.get("/api/branches")
def list_branches(tree_id: str | None = None):
    """List branches. If tree_id provided, show that tree; else list all trees."""
    if tree_id:
        tree = get_tree(tree_id)
        if not tree:
            return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
        return {"status": "ok", "tree": tree.to_dict()}
    return {"status": "ok", "trees": list_trees()}


@app.post("/api/branch/switch")
def switch_branch(tree_id: str, branch_id: str):
    """Switch the active branch for a tree."""
    tree = get_tree(tree_id)
    if not tree:
        return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
    try:
        tree.switch_branch(branch_id)
        return {"status": "ok", "active_branch_id": branch_id}
    except KeyError as e:
        return JSONResponse({"status": "error", "message": str(e)}, 404)


@app.post("/api/branch/message")
def add_branch_message(req: BranchMessageRequest):
    """Add a message to a branch (or the active branch)."""
    tree = get_tree(req.tree_id)
    if not tree:
        return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
    if req.branch_id:
        try:
            tree.switch_branch(req.branch_id)
        except KeyError as e:
            return JSONResponse({"status": "error", "message": str(e)}, 404)
    msg = tree.add_message(req.role, req.content)
    return {"status": "ok", "message": msg.to_dict(), "branch_id": tree.active_branch_id}


@app.post("/api/branch/chat")
def chat_on_branch(req: BranchMessageRequest):
    """Send a user message on a branch and get an agent response.

    This adds the user message, runs the agent with the branch's full
    history, and adds the assistant response.
    """
    tree = get_tree(req.tree_id)
    if not tree:
        return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
    if req.branch_id:
        try:
            tree.switch_branch(req.branch_id)
        except KeyError as e:
            return JSONResponse({"status": "error", "message": str(e)}, 404)

    # Add user message
    tree.add_message("user", req.content)

    # Run agent with branch history
    from agentos.core.agent import Agent

    agent = Agent(
        name="branch-agent",
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant.",
    )
    # Feed the branch history into the agent's messages
    agent.messages = tree.get_messages_openai()
    agent.events = []

    import io, sys
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        msg, event = agent._run_sync_step()
    except AttributeError:
        # Fallback: run full agent
        from agentos.providers.router import call_model as call_llm_direct
        msg_result, event = call_llm_direct(
            model="gpt-4o-mini",
            messages=tree.get_messages_openai(),
            tools=[],
            agent_name="branch-agent",
        )
        response_text = msg_result.content or ""
        tree.add_message("assistant", response_text)
        for e in [event]:
            store.log_event(e)
        sys.stdout = old
        return {
            "status": "ok",
            "response": response_text,
            "cost": round(event.cost_usd, 6),
            "tokens": event.tokens_used,
            "branch_id": tree.active_branch_id,
            "messages": [m.to_dict() for m in tree.get_messages()],
        }
    finally:
        sys.stdout = old


@app.post("/api/branch/compare")
def compare_branches(req: CompareBranchesRequest):
    """Compare two branches side by side."""
    tree = get_tree(req.tree_id)
    if not tree:
        return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
    try:
        result = tree.compare_branches(req.branch_a, req.branch_b)
        return {"status": "ok", "comparison": result}
    except KeyError as e:
        return JSONResponse({"status": "error", "message": str(e)}, 404)


@app.post("/api/branch/merge")
def merge_branches(req: MergeBranchesRequest):
    """Merge two branches into a new combined branch."""
    tree = get_tree(req.tree_id)
    if not tree:
        return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
    try:
        merged_id = tree.merge_branches(req.branch_a, req.branch_b, label=req.label)
        return {"status": "ok", "merged_branch_id": merged_id, "tree": tree.to_dict()}
    except KeyError as e:
        return JSONResponse({"status": "error", "message": str(e)}, 404)


@app.delete("/api/branch/{tree_id}/{branch_id}")
def delete_branch(tree_id: str, branch_id: str):
    """Delete a branch (cannot delete the main branch)."""
    tree = get_tree(tree_id)
    if not tree:
        return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
    if tree.delete_branch(branch_id):
        return {"status": "deleted", "branch_id": branch_id}
    return JSONResponse({"status": "error", "message": "Cannot delete main branch or branch not found"}, 400)


@app.get("/api/branch/messages")
def get_branch_messages(tree_id: str, branch_id: str | None = None):
    """Get all messages for a specific branch (or the active branch)."""
    tree = get_tree(tree_id)
    if not tree:
        return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
    try:
        bid = branch_id or tree.active_branch_id
        msgs = tree.get_messages(bid)
        return {
            "status": "ok",
            "branch_id": bid,
            "messages": [m.to_dict() for m in msgs],
        }
    except KeyError as e:
        return JSONResponse({"status": "error", "message": str(e)}, 404)


@app.post("/api/branch/new-tree")
def create_new_tree():
    """Create a new conversation tree."""
    tree = get_or_create_tree()
    return {"status": "created", "tree": tree.to_dict()}


# ── Event Bus API ──

@app.post("/api/webhook/{event_name}")
async def webhook_receiver(event_name: str, body: dict = {}):
    """Receive a webhook POST and emit it through the event bus.

    Example: POST /api/webhook/deploy.completed  {"repo": "myapp", "status": "success"}
    """
    _webhook_trigger.event_name = f"webhook.{event_name}"
    _webhook_trigger.fire(data=body, source=f"webhook:{event_name}")
    return {
        "status": "emitted",
        "event": f"webhook.{event_name}",
        "listeners_matched": len([
            l for l in event_bus.list_listeners()
            if l.matches(f"webhook.{event_name}")
        ]),
    }


@app.get("/api/events/listeners")
def list_event_listeners():
    """List all registered event listeners."""
    return {
        "overview": event_bus.get_overview(),
        "listeners": [l.to_dict() for l in event_bus.list_listeners()],
    }


@app.get("/api/events/history")
def get_event_history(limit: int = 20):
    """Get recent event emission history."""
    return {
        "history": [log.to_dict() for log in event_bus.get_history(limit=limit)],
    }


@app.post("/api/events/emit")
async def emit_event(body: dict = {}):
    """Manually emit an event through the bus.

    Body: {"event_name": "custom.test", "data": {"key": "value"}}
    """
    event_name = body.get("event_name", "custom.manual")
    data = body.get("data", {})
    log = event_bus.emit(event_name, data=data, source="api:manual")
    return {
        "status": "emitted",
        "event_name": event_name,
        "listeners_triggered": log.listeners_triggered,
    }


# ── Marketplace API ──


class PublishRequest(BaseModel):
    name: str
    description: str = ""
    author: str = "anonymous"
    version: str = "1.0.0"
    category: str = "general"
    icon: str = "🤖"
    tags: list[str] = []
    price: float = 0.0
    config: dict = {}


class ReviewRequest(BaseModel):
    user: str = "anonymous"
    rating: float = 5.0
    comment: str = ""


@app.get("/api/marketplace/list")
def marketplace_list(category: str = "", sort: str = "downloads"):
    """List marketplace agents, optionally filtered by category."""
    mp = get_marketplace_store()
    agents = mp.search(category=category, sort_by=sort)
    return {
        "agents": [a.to_card() for a in agents],
        "categories": mp.get_categories(),
        "stats": mp.stats(),
    }


@app.get("/api/marketplace/search")
def marketplace_search(q: str = "", category: str = "", sort: str = "downloads"):
    """Search the marketplace."""
    mp = get_marketplace_store()
    agents = mp.search(query=q, category=category, sort_by=sort)
    return {"agents": [a.to_card() for a in agents], "query": q}


@app.get("/api/marketplace/trending")
def marketplace_trending():
    mp = get_marketplace_store()
    return {"agents": [a.to_card() for a in mp.get_trending()]}


@app.get("/api/marketplace/top-rated")
def marketplace_top_rated():
    mp = get_marketplace_store()
    return {"agents": [a.to_card() for a in mp.get_top_rated()]}


@app.get("/api/marketplace/{agent_id}")
def marketplace_detail(agent_id: str):
    """Get full details for a marketplace agent, including reviews."""
    mp = get_marketplace_store()
    agent = mp.get(agent_id)
    if not agent:
        return JSONResponse({"status": "error", "message": "Agent not found"}, 404)
    data = agent.model_dump()
    data["status"] = "ok"
    return data


@app.post("/api/marketplace/publish")
def marketplace_publish(req: PublishRequest):
    """Publish a new agent to the marketplace."""
    mp = get_marketplace_store()
    agent = mp.publish(
        name=req.name,
        description=req.description,
        author=req.author,
        version=req.version,
        category=req.category,
        icon=req.icon,
        tags=req.tags,
        price=req.price,
        config=req.config,
    )
    return {"status": "published", "agent": agent.to_card()}


@app.post("/api/marketplace/install/{agent_id}")
def marketplace_install(agent_id: str):
    """Install an agent — increments download counter and returns config."""
    mp = get_marketplace_store()
    agent = mp.install(agent_id)
    if not agent:
        return JSONResponse({"status": "error", "message": "Agent not found"}, 404)
    return {
        "status": "installed",
        "agent": agent.to_card(),
        "config": agent.config.model_dump(),
    }


@app.post("/api/marketplace/review/{agent_id}")
def marketplace_review(agent_id: str, req: ReviewRequest):
    """Leave a review for a marketplace agent."""
    mp = get_marketplace_store()
    review = mp.review(agent_id, user=req.user, rating=req.rating, comment=req.comment)
    if not review:
        return JSONResponse({"status": "error", "message": "Agent not found"}, 404)
    agent = mp.get(agent_id)
    return {
        "status": "reviewed",
        "review": review.model_dump(),
        "new_rating": agent.rating if agent else 0,
        "review_count": agent.review_count if agent else 0,
    }


@app.delete("/api/marketplace/{agent_id}")
def marketplace_delete(agent_id: str):
    mp = get_marketplace_store()
    if mp.delete(agent_id):
        return {"status": "deleted"}
    return JSONResponse({"status": "error", "message": "Agent not found"}, 404)


# ── Embed / Widget API ──


class EmbedConfigRequest(BaseModel):
    agent_name: str = "AgentOS"
    theme: str = "dark"
    position: str = "bottom-right"
    accent_color: str = "#6c5ce7"
    logo: str = ""
    greeting: str = "Hi! How can I help you today?"
    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a helpful assistant."
    tools: list[str] = []
    api_key: str = ""


@app.get("/embed/chat.js")
def embed_chat_js():
    """Serve the embeddable widget JavaScript."""
    js = generate_widget_js()
    return PlainTextResponse(js, media_type="application/javascript")


@app.get("/api/embed/widget")
def embed_widget_get(
    agent_name: str = "AgentOS",
    theme: str = "dark",
    position: str = "bottom-right",
    accent_color: str = "#6c5ce7",
    greeting: str = "Hi! How can I help you today?",
    model: str = "gpt-4o-mini",
):
    """Return a self-contained HTML widget snippet."""
    html = generate_widget(
        agent_name=agent_name,
        base_url="",  # empty = same origin
        theme=theme,
        position=position,
        accent_color=accent_color,
        greeting=greeting,
        model=model,
    )
    return HTMLResponse(html)


@app.post("/api/embed/widget")
def embed_widget_post(req: EmbedConfigRequest):
    """Return a self-contained HTML widget snippet (POST with full config)."""
    html = generate_widget(
        agent_name=req.agent_name,
        base_url="",
        api_key=req.api_key,
        theme=req.theme,
        position=req.position,
        accent_color=req.accent_color,
        greeting=req.greeting,
        model=req.model,
        system_prompt=req.system_prompt,
        tools=req.tools,
    )
    return {"status": "ok", "html": html}


@app.get("/api/embed/snippet")
def embed_snippet(
    base_url: str = "http://localhost:8000",
    agent_name: str = "AgentOS",
    theme: str = "dark",
    position: str = "bottom-right",
    accent_color: str = "#6c5ce7",
    api_key: str = "",
):
    """Return a copy-paste code snippet for embedding."""
    snippet = generate_snippet(
        base_url=base_url,
        api_key=api_key,
        agent_name=agent_name,
        theme=theme,
        position=position,
        accent_color=accent_color,
    )
    return {"status": "ok", "snippet": snippet}


@app.get("/embed/preview")
def embed_preview(
    agent_name: str = "AgentOS",
    theme: str = "dark",
    accent_color: str = "#6c5ce7",
):
    """Render a full HTML page with the widget embedded — handy for previewing."""
    widget_html = generate_widget(
        agent_name=agent_name,
        base_url="",
        theme=theme,
        accent_color=accent_color,
    )
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>AgentOS Widget Preview</title>
<style>body{{margin:0;font-family:system-ui;background:{'#f5f5f5' if theme=='light' else '#1a1a2e'};
color:{'#333' if theme=='light' else '#eee'};display:flex;align-items:center;justify-content:center;min-height:100vh}}
.demo{{text-align:center}}.demo h1{{font-size:28px}}.demo p{{opacity:.6}}</style></head>
<body><div class="demo"><h1>Your Website</h1><p>The AgentOS chat widget is in the corner ↘</p></div>
{widget_html}</body></html>"""
    return HTMLResponse(page)


# ── Analytics API ──

def _bucket_key(ts: float, granularity: str) -> str:
    """Convert a unix timestamp to a bucket key string."""
    import datetime
    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    if granularity == "hour":
        return dt.strftime("%Y-%m-%d %H:00")
    elif granularity == "week":
        # ISO week start (Monday)
        start = dt - datetime.timedelta(days=dt.weekday())
        return start.strftime("%Y-%m-%d")
    else:  # day (default)
        return dt.strftime("%Y-%m-%d")


@app.get("/api/analytics/cost-over-time")
def analytics_cost_over_time(granularity: str = Query("day", pattern="^(hour|day|week)$")):
    """Aggregate cost by time bucket (hour / day / week)."""
    buckets: dict[str, dict] = {}
    for ev in store.events:
        key = _bucket_key(ev["timestamp"], granularity)
        if key not in buckets:
            buckets[key] = {"bucket": key, "cost": 0.0, "tokens": 0, "queries": 0}
        buckets[key]["cost"] += ev.get("cost_usd", 0.0)
        buckets[key]["tokens"] += ev.get("tokens_used", 0)
        if ev.get("event_type") == "llm_call":
            buckets[key]["queries"] += 1
    series = sorted(buckets.values(), key=lambda b: b["bucket"])
    for b in series:
        b["cost"] = round(b["cost"], 6)
    return {"granularity": granularity, "series": series}


@app.get("/api/analytics/popular-tools")
def analytics_popular_tools():
    """Rank tools by usage count."""
    counts: dict[str, dict] = {}
    for ev in store.events:
        if ev.get("event_type") != "tool_call":
            continue
        tool_name = (ev.get("data") or {}).get("tool", "unknown")
        if tool_name not in counts:
            counts[tool_name] = {"tool": tool_name, "count": 0, "total_latency_ms": 0.0, "total_cost": 0.0}
        counts[tool_name]["count"] += 1
        counts[tool_name]["total_latency_ms"] += ev.get("latency_ms", 0.0)
        counts[tool_name]["total_cost"] += ev.get("cost_usd", 0.0)
    ranked = sorted(counts.values(), key=lambda t: t["count"], reverse=True)
    for t in ranked:
        t["avg_latency_ms"] = round(t["total_latency_ms"] / max(t["count"], 1), 1)
        t["total_cost"] = round(t["total_cost"], 6)
    return {"tools": ranked}


@app.get("/api/analytics/model-comparison")
def analytics_model_comparison():
    """Compare models by cost, speed, tokens, and call count."""
    models: dict[str, dict] = {}
    for ev in store.events:
        if ev.get("event_type") != "llm_call":
            continue
        model = (ev.get("data") or {}).get("model", "unknown")
        if model not in models:
            models[model] = {
                "model": model, "calls": 0, "total_cost": 0.0,
                "total_tokens": 0, "total_latency_ms": 0.0,
                "quality_scores": [],
            }
        m = models[model]
        m["calls"] += 1
        m["total_cost"] += ev.get("cost_usd", 0.0)
        m["total_tokens"] += ev.get("tokens_used", 0)
        m["total_latency_ms"] += ev.get("latency_ms", 0.0)
    # Attach quality scores from agent data
    for agent_data in store.agents.values():
        for qs in agent_data.get("quality_scores", []):
            # Attempt to attribute to last-used model (best effort)
            pass
    result = []
    for m in models.values():
        m["avg_cost"] = round(m["total_cost"] / max(m["calls"], 1), 6)
        m["avg_latency_ms"] = round(m["total_latency_ms"] / max(m["calls"], 1), 1)
        m["avg_tokens"] = round(m["total_tokens"] / max(m["calls"], 1))
        m["total_cost"] = round(m["total_cost"], 6)
        result.append(m)
    result.sort(key=lambda x: x["calls"], reverse=True)
    return {"models": result}


@app.get("/api/analytics/agent-leaderboard")
def analytics_agent_leaderboard():
    """Rank agents by quality, cost-efficiency, and usage."""
    leaderboard = []
    for name, a in store.agents.items():
        scores = [s["score"] for s in a.get("quality_scores", [])]
        avg_quality = round(sum(scores) / len(scores), 2) if scores else None
        total_queries = a.get("total_llm_calls", 0)
        cost_per_query = round(a["total_cost"] / max(total_queries, 1), 6)
        leaderboard.append({
            "agent": name,
            "avg_quality": avg_quality,
            "total_cost": round(a["total_cost"], 6),
            "total_tokens": a["total_tokens"],
            "total_queries": total_queries,
            "total_tool_calls": a.get("total_tool_calls", 0),
            "cost_per_query": cost_per_query,
            "total_events": a["total_events"],
        })
    # Sort: quality first (desc), then by queries (desc)
    leaderboard.sort(key=lambda x: (x["avg_quality"] or 0, x["total_queries"]), reverse=True)
    # Compute summary totals
    total_spend = round(sum(a["total_cost"] for a in store.agents.values()), 6)
    total_queries = sum(a.get("total_llm_calls", 0) for a in store.agents.values())
    avg_cost = round(total_spend / max(total_queries, 1), 6)
    return {
        "leaderboard": leaderboard,
        "summary": {
            "total_spend": total_spend,
            "total_queries": total_queries,
            "avg_cost_per_query": avg_cost,
            "total_agents": len(store.agents),
            "total_events": len(store.events),
        },
    }


# ── The Complete Web UI ──

WEB_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgentOS Platform</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#08080f;color:#e0e0e0;min-height:100vh}
.app{display:grid;grid-template-columns:260px 1fr;grid-template-rows:56px 1fr;height:100vh}
.topbar{grid-column:1/-1;background:#0d0d1a;border-bottom:1px solid #1a1a2e;padding:0 24px;display:flex;align-items:center;justify-content:space-between}
.topbar h1{font-size:18px;color:#00d4ff}
.topbar .status{color:#00ff88;font-size:13px;display:flex;align-items:center;gap:8px}
.topbar .status span.small{font-size:11px;color:#888}
.sidebar{background:#0a0a14;border-right:1px solid #1a1a2e;padding:16px;overflow-y:auto}
.sidebar h3{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#555;margin:16px 0 8px}
.nav-item{padding:10px 12px;border-radius:8px;cursor:pointer;font-size:14px;margin-bottom:4px;transition:all 0.2s;display:flex;align-items:center;gap:8px}
.nav-item:hover{background:#12121f}
.nav-item.active{background:#00d4ff15;color:#00d4ff}
.main{padding:24px;overflow-y:auto}
.panel{display:none}
.panel.active{display:block}
.card{background:#0d0d1a;border:1px solid #1a1a2e;border-radius:12px;padding:24px;margin-bottom:16px}
.card h2{font-size:20px;margin-bottom:16px;color:#fff}
label{display:block;font-size:13px;color:#888;margin-bottom:6px;margin-top:16px}
input,textarea,select{width:100%;padding:10px 14px;background:#08080f;border:1px solid #2a2a4a;border-radius:8px;color:#fff;font-size:14px;font-family:inherit}
input:focus,textarea:focus,select:focus{outline:none;border-color:#00d4ff}
textarea{min-height:80px;resize:vertical}
.btn{padding:12px 24px;border-radius:8px;border:none;font-size:14px;font-weight:600;cursor:pointer;transition:all 0.2s}
.btn-primary{background:linear-gradient(135deg,#00d4ff,#0088ff);color:#000}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 4px 20px rgba(0,212,255,0.3)}
.btn-primary:disabled{opacity:0.5;cursor:not-allowed;transform:none}
.btn-secondary{background:#1a1a2e;color:#fff}
.tools-grid{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.tool-tag{padding:6px 14px;border-radius:20px;font-size:13px;cursor:pointer;border:1px solid #2a2a4a;background:#08080f;transition:all 0.2s}
.tool-tag.selected{background:#00d4ff22;border-color:#00d4ff;color:#00d4ff}
.tool-tag:hover{border-color:#00d4ff}
.response-box{background:#08080f;border:1px solid #1a1a2e;border-radius:8px;padding:16px;margin-top:16px;min-height:100px;white-space:pre-wrap;line-height:1.6}
.stats-row{display:flex;gap:12px;margin-top:12px}
.stat-chip{background:#12121f;padding:6px 12px;border-radius:6px;font-size:12px;color:#888}
.stat-chip span{color:#00d4ff;font-weight:600}
.templates-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}
.template-card{background:#0d0d1a;border:1px solid #1a1a2e;border-radius:12px;padding:20px;cursor:pointer;transition:all 0.3s}
.template-card:hover{border-color:#00d4ff33;transform:translateY(-2px)}
.template-card .icon{font-size:28px;margin-bottom:8px}
.template-card h4{color:#fff;margin-bottom:4px}
.template-card p{color:#666;font-size:13px}
.template-card .cat{font-size:11px;color:#00d4ff;margin-top:8px}
.monitor-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
.monitor-card{background:#0d0d1a;border:1px solid #1a1a2e;border-radius:12px;padding:20px}
.monitor-card .label{font-size:11px;color:#666;text-transform:uppercase}
.monitor-card .value{font-size:28px;font-weight:700;margin-top:4px}
.monitor-card .value.blue{color:#00d4ff}
.monitor-card .value.green{color:#00ff88}
.monitor-card .value.yellow{color:#ffaa00}
.event-row{background:#0a0a14;border:1px solid #12121f;padding:10px 14px;margin-bottom:4px;border-radius:6px;display:grid;grid-template-columns:120px 90px 1fr 80px 70px;gap:10px;font-size:13px;align-items:center}
.event-type{padding:3px 8px;border-radius:10px;font-size:11px;font-weight:600;text-align:center}
.event-type.llm_call{background:#00d4ff22;color:#00d4ff}
.event-type.tool_call{background:#ffaa0022;color:#ffaa00}
.loading{display:inline-block;width:16px;height:16px;border:2px solid #333;border-top-color:#00d4ff;border-radius:50%;animation:spin 0.8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
/* ── Analytics charts (pure CSS) ── */
.an-summary{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:20px}
.an-summary .monitor-card .value{font-size:24px}
.an-chart-wrap{position:relative;height:220px;background:#08080f;border:1px solid #1a1a2e;border-radius:10px;padding:16px 16px 32px 48px;margin-bottom:20px;overflow:hidden}
.an-chart-wrap .y-axis{position:absolute;left:0;top:16px;bottom:32px;width:44px;display:flex;flex-direction:column;justify-content:space-between;align-items:flex-end;padding-right:6px;font-size:10px;color:#555}
.an-chart-wrap .x-axis{position:absolute;left:48px;right:16px;bottom:4px;display:flex;justify-content:space-between;font-size:10px;color:#555;overflow:hidden}
.an-line-area{position:absolute;left:48px;top:16px;right:16px;bottom:32px}
.an-line-area svg{width:100%;height:100%;overflow:visible}
.an-line-area svg polyline{fill:none;stroke:#00d4ff;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.an-line-area svg polygon{fill:url(#costGrad);opacity:0.25}
.an-line-area svg circle{fill:#00d4ff;r:3}
.an-bar-wrap{display:flex;align-items:flex-end;gap:10px;height:160px;padding:0 4px}
.an-bar{display:flex;flex-direction:column;align-items:center;flex:1;min-width:0}
.an-bar .bar{width:100%;max-width:56px;border-radius:6px 6px 0 0;transition:height 0.4s ease;position:relative}
.an-bar .bar:hover{opacity:0.85}
.an-bar .bar-label{font-size:10px;color:#888;margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:72px;text-align:center}
.an-bar .bar-value{font-size:11px;color:#fff;font-weight:600;margin-bottom:4px}
.an-table{width:100%;border-collapse:separate;border-spacing:0}
.an-table th{text-align:left;font-size:11px;text-transform:uppercase;color:#555;letter-spacing:0.5px;padding:8px 12px;border-bottom:1px solid #1a1a2e}
.an-table td{padding:10px 12px;font-size:13px;border-bottom:1px solid #0d0d1a}
.an-table tr:hover td{background:#0a0a14}
.an-rank{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;font-size:12px;font-weight:700}
.an-rank.gold{background:#ffaa0033;color:#ffaa00}
.an-rank.silver{background:#88888833;color:#aaa}
.an-rank.bronze{background:#cc774433;color:#cc7744}
.an-rank.normal{background:#1a1a2e;color:#666}
.an-quality-bar{height:6px;border-radius:3px;background:#1a1a2e;overflow:hidden;width:80px;display:inline-block;vertical-align:middle;margin-left:6px}
.an-quality-bar .fill{height:100%;border-radius:3px}
.an-tabs{display:flex;gap:4px;margin-bottom:16px}
.an-tab{padding:6px 14px;border-radius:6px;font-size:12px;cursor:pointer;border:1px solid #2a2a4a;background:#08080f;color:#888;transition:all 0.2s}
.an-tab.active{background:#00d4ff15;border-color:#00d4ff;color:#00d4ff}
/* ── Multi-modal upload ── */
.mm-upload-zone{border:2px dashed #2a2a4a;border-radius:12px;padding:40px 20px;text-align:center;cursor:pointer;transition:all 0.3s;background:#08080f;margin-bottom:16px}
.mm-upload-zone:hover,.mm-upload-zone.dragover{border-color:#00d4ff;background:#00d4ff08}
.mm-upload-zone .icon{font-size:40px;margin-bottom:8px}
.mm-upload-zone p{color:#888;font-size:14px}
.mm-upload-zone .sub{color:#555;font-size:12px;margin-top:4px}
.mm-preview{display:none;margin-bottom:16px;background:#0a0a14;border:1px solid #1a1a2e;border-radius:10px;padding:16px}
.mm-preview img{max-width:100%;max-height:300px;border-radius:8px;margin-bottom:8px}
.mm-preview .file-info{font-size:13px;color:#888}
.mm-result{display:none;background:#08080f;border:1px solid #1a1a2e;border-radius:8px;padding:16px;margin-top:16px;white-space:pre-wrap;line-height:1.6;font-size:14px}
/* ── Branching ── */
.br-bar{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
.br-chip{padding:5px 12px;border-radius:16px;font-size:12px;cursor:pointer;border:1px solid #2a2a4a;background:#08080f;transition:all 0.2s;white-space:nowrap}
.br-chip:hover{border-color:#00d4ff}
.br-chip.active{background:#00d4ff18;border-color:#00d4ff;color:#00d4ff;font-weight:600}
.br-chip .dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px}
.br-msg{padding:8px 14px;border-radius:10px;display:inline-block;max-width:80%;white-space:pre-wrap;line-height:1.5;font-size:14px}
.br-msg.user{background:#00d4ff18;color:#00d4ff;margin-left:auto}
.br-msg.assistant{background:#1a1a2e;color:#e0e0e0}
.br-msg.system{background:#1a0a2e;color:#aa88ff;font-size:12px;font-style:italic}
.br-row{display:flex;margin:6px 0}
.br-row.right{justify-content:flex-end}
.br-idx{font-size:10px;color:#444;width:22px;flex-shrink:0;padding-top:10px;text-align:right;margin-right:6px}
.br-fork-btn{font-size:10px;color:#555;cursor:pointer;margin-left:4px;opacity:0;transition:opacity 0.2s}
.br-row:hover .br-fork-btn{opacity:1}
.br-fork-btn:hover{color:#00d4ff}
.br-compare-col{flex:1;min-width:0;padding:8px;background:#08080f;border-radius:8px;border:1px solid #1a1a2e;max-height:400px;overflow-y:auto}
.br-compare-col h4{font-size:13px;margin-bottom:8px;color:#fff}
</style>
</head>
<body>
<div class="app">
<div class="topbar">
<h1>🤖 AgentOS Platform</h1>
<div class="status">● Online <span class="small" id="user-status">Not logged in</span></div>
</div>
<div class="sidebar">
<h3>Build</h3>
<div class="nav-item active" onclick="showPanel('builder')">🛠️ Agent Builder</div>
<div class="nav-item" onclick="showPanel('templates')">📦 Templates</div>
<h3>Operate</h3>
<div class="nav-item" onclick="showPanel('chat')">💬 Chat</div>
<div class="nav-item" onclick="showPanel('branching')">🌿 Branching</div>
<div class="nav-item" onclick="showPanel('monitor')">📊 Monitor</div>
<div class="nav-item" onclick="showPanel('analytics')">📈 Analytics</div>
<div class="nav-item" onclick="showPanel('scheduler')">⏰ Scheduler</div>
<div class="nav-item" onclick="showPanel('events')">⚡ Events</div>
<div class="nav-item" onclick="showPanel('abtest')">🧪 A/B Testing</div>
<div class="nav-item" onclick="showPanel('multimodal')">👁️ Multi-modal</div>
<h3>Manage</h3>
<div class="nav-item" onclick="showPanel('auth')">🔑 Account & Usage</div>
<div class="nav-item" onclick="showPanel('marketplace')">🏪 Marketplace</div>
<div class="nav-item" onclick="showPanel('embed')">🔌 Embed SDK</div>
</div>
<div class="main">

<!-- AGENT BUILDER -->
<div class="panel active" id="panel-builder">
<div class="card">
<h2>🛠️ Build Your Agent</h2>
<label>Agent Name</label>
<input type="text" id="b-name" value="my-agent" placeholder="my-agent">
<label>Model</label>
<select id="b-model">
<option value="gpt-4o-mini">GPT-4o Mini (cheap + fast)</option>
<option value="gpt-4o">GPT-4o (powerful)</option>
<option value="claude-sonnet">Claude Sonnet (balanced)</option>
<option value="claude-haiku">Claude Haiku (fastest)</option>
</select>
<label>System Prompt (Agent's personality and instructions)</label>
<textarea id="b-prompt" placeholder="You are a helpful assistant...">You are a helpful assistant. Use tools when needed to answer accurately.</textarea>
<label>Tools (click to enable)</label>
<div class="tools-grid">
<div class="tool-tag selected" data-tool="calculator" onclick="toggleTool(this)">🔢 Calculator</div>
<div class="tool-tag" data-tool="weather" onclick="toggleTool(this)">🌤️ Weather</div>
<div class="tool-tag" data-tool="web_search" onclick="toggleTool(this)">🔍 Web Search</div>
<div class="tool-tag" data-tool="analyze_image" onclick="toggleTool(this)">👁️ Vision</div>
<div class="tool-tag" data-tool="read_document" onclick="toggleTool(this)">📄 Doc Reader</div>
<div class="tool-tag" data-tool="analyze_document" onclick="toggleTool(this)">📑 Doc Q&A</div>
</div>
<label>Temperature (creativity: 0=focused, 1=creative)</label>
<input type="range" id="b-temp" min="0" max="1" step="0.1" value="0.7" oninput="document.getElementById('temp-val').textContent=this.value">
<span id="temp-val" style="color:#00d4ff;font-size:13px">0.7</span>
<label>Budget Limit ($/day)</label>
<input type="number" id="b-budget" value="5.00" step="0.5" min="0.5">
<div style="margin-top:24px">
<label>Try Your Agent</label>
<input type="text" id="b-query" placeholder="Ask your agent something..." onkeydown="if(event.key==='Enter')runBuilder()">
<button class="btn btn-primary" style="margin-top:12px;width:100%" onclick="runBuilder()" id="run-btn">▶️ Run Agent</button>
</div>
<div id="b-response" style="display:none">
<label>Response</label>
<div class="response-box" id="b-response-text"></div>
<div class="stats-row" id="b-stats"></div>
</div>
</div>
</div>

<!-- TEMPLATES -->
<div class="panel" id="panel-templates">
<div class="card">
<h2>📦 Agent Templates</h2>
<p style="color:#888;margin-bottom:16px">Pre-built agents ready to deploy. Click one to load it into the builder.</p>
<div class="templates-grid" id="templates-list"></div>
</div>
</div>

<!-- CHAT -->
<div class="panel" id="panel-chat">
<div class="card" style="height:calc(100vh - 140px);display:flex;flex-direction:column">
<h2>💬 Agent Chat</h2>
<div id="chat-messages" style="flex:1;overflow-y:auto;padding:16px 0"></div>
<div style="display:flex;gap:8px">
<input type="text" id="chat-input" placeholder="Type a message..." onkeydown="if(event.key==='Enter')sendChat()" style="flex:1">
<button class="btn btn-primary" onclick="sendChat()">Send</button>
</div>
</div>
</div>

<!-- BRANCHING CHAT -->
<div class="panel" id="panel-branching">
<div class="card" style="height:calc(100vh - 140px);display:flex;flex-direction:column">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
<h2>🌿 Branching Chat</h2>
<button class="btn btn-secondary" style="font-size:12px;padding:6px 12px" onclick="brNewTree()">+ New Conversation</button>
</div>
<p style="color:#888;font-size:13px;margin-bottom:8px">Explore "what if" scenarios by forking conversations. Click the fork icon on any message to create a branch.</p>
<!-- Branch selector bar -->
<div class="br-bar" id="br-bar"><span style="color:#555;font-size:12px">Start a conversation to see branches.</span></div>
<!-- Action buttons -->
<div style="display:flex;gap:6px;margin-bottom:10px" id="br-actions" style="display:none">
<button class="btn btn-secondary" style="font-size:11px;padding:4px 10px" onclick="brCompare()" id="br-compare-btn" title="Select two branches then compare">Compare</button>
<button class="btn btn-secondary" style="font-size:11px;padding:4px 10px" onclick="brMerge()" id="br-merge-btn" title="Merge two branches">Merge</button>
</div>
<!-- Messages area -->
<div id="br-messages" style="flex:1;overflow-y:auto;padding:8px 0"></div>
<!-- Compare results area (hidden by default) -->
<div id="br-compare-area" style="display:none;margin-bottom:8px">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
<span style="color:#888;font-size:12px;font-weight:600">Branch Comparison</span>
<span style="font-size:11px;color:#555;cursor:pointer" onclick="document.getElementById('br-compare-area').style.display='none'">close</span>
</div>
<div style="display:flex;gap:8px" id="br-compare-cols"></div>
</div>
<!-- Input -->
<div style="display:flex;gap:8px">
<input type="text" id="br-input" placeholder="Type a message... (fork icon on messages to branch)" onkeydown="if(event.key==='Enter')brSend()" style="flex:1">
<button class="btn btn-primary" onclick="brSend()">Send</button>
</div>
</div>
</div>

<!-- MONITOR -->
<div class="panel" id="panel-monitor">
<div class="monitor-grid" id="mon-overview">
<div class="monitor-card"><div class="label">Agents</div><div class="value blue" id="m-agents">0</div></div>
<div class="monitor-card"><div class="label">Events</div><div class="value green" id="m-events">0</div></div>
<div class="monitor-card"><div class="label">Cost</div><div class="value yellow" id="m-cost">$0</div></div>
<div class="monitor-card"><div class="label">Status</div><div class="value blue" id="m-status">Ready</div></div>
</div>
<div class="card">
<h2>Live Events</h2>
<div id="mon-events"></div>
</div>
</div>

<!-- ANALYTICS -->
<div class="panel" id="panel-analytics">
<!-- Summary cards -->
<div class="an-summary" id="an-summary">
<div class="monitor-card"><div class="label">Total Spend</div><div class="value yellow" id="an-total-spend">$0</div></div>
<div class="monitor-card"><div class="label">Total Queries</div><div class="value blue" id="an-total-queries">0</div></div>
<div class="monitor-card"><div class="label">Avg Cost / Query</div><div class="value green" id="an-avg-cost">$0</div></div>
<div class="monitor-card"><div class="label">Total Events</div><div class="value purple" id="an-total-events">0</div></div>
<div class="monitor-card"><div class="label">Active Agents</div><div class="value blue" id="an-total-agents">0</div></div>
</div>

<!-- Cost over time chart -->
<div class="card">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
<h2>Cost Over Time</h2>
<div class="an-tabs">
<div class="an-tab active" onclick="switchCostGranularity('hour',this)">Hour</div>
<div class="an-tab" onclick="switchCostGranularity('day',this)">Day</div>
<div class="an-tab" onclick="switchCostGranularity('week',this)">Week</div>
</div>
</div>
<div class="an-chart-wrap" id="an-cost-chart">
<div class="y-axis" id="an-cost-y"></div>
<div class="an-line-area" id="an-cost-area">
<svg viewBox="0 0 100 100" preserveAspectRatio="none">
<defs><linearGradient id="costGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#00d4ff"/><stop offset="100%" stop-color="#00d4ff" stop-opacity="0"/></linearGradient></defs>
<polygon id="an-cost-fill" points="0,100 100,100"/>
<polyline id="an-cost-line" points=""/>
<g id="an-cost-dots"></g>
</svg>
</div>
<div class="x-axis" id="an-cost-x"></div>
</div>
</div>

<!-- Popular tools bar chart -->
<div class="card">
<h2>Most Used Tools</h2>
<div class="an-bar-wrap" id="an-tools-bars"></div>
<p style="color:#555;font-size:12px;text-align:center;margin-top:8px" id="an-tools-empty">No tool usage recorded yet. Run an agent with tools enabled.</p>
</div>

<!-- Model comparison table -->
<div class="card">
<h2>Model Comparison</h2>
<div style="overflow-x:auto">
<table class="an-table" id="an-model-table">
<thead><tr><th>Model</th><th>Calls</th><th>Total Cost</th><th>Avg Cost</th><th>Avg Latency</th><th>Avg Tokens</th><th>Total Tokens</th></tr></thead>
<tbody id="an-model-tbody"><tr><td colspan="7" style="color:#555;text-align:center;padding:20px">No model data yet.</td></tr></tbody>
</table>
</div>
</div>

<!-- Agent leaderboard -->
<div class="card">
<h2>Agent Leaderboard</h2>
<div style="overflow-x:auto">
<table class="an-table" id="an-leader-table">
<thead><tr><th>#</th><th>Agent</th><th>Quality</th><th>Queries</th><th>Tool Calls</th><th>Total Cost</th><th>Cost/Query</th><th>Events</th></tr></thead>
<tbody id="an-leader-tbody"><tr><td colspan="8" style="color:#555;text-align:center;padding:20px">No agent data yet.</td></tr></tbody>
</table>
</div>
</div>
</div>

<!-- MARKETPLACE -->
<div class="panel" id="panel-marketplace">
<!-- Stats + search -->
<div class="monitor-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:16px">
<div class="monitor-card"><div class="label">Agents</div><div class="value blue" id="mp-stat-agents">0</div></div>
<div class="monitor-card"><div class="label">Downloads</div><div class="value green" id="mp-stat-downloads">0</div></div>
<div class="monitor-card"><div class="label">Reviews</div><div class="value yellow" id="mp-stat-reviews">0</div></div>
<div class="monitor-card"><div class="label">Free</div><div class="value purple" id="mp-stat-free">0</div></div>
</div>
<div class="card">
<div style="display:flex;gap:8px;align-items:center;margin-bottom:12px">
<input type="text" id="mp-search" placeholder="Search agents..." style="flex:1" onkeydown="if(event.key==='Enter')mpSearch()">
<select id="mp-cat" onchange="mpSearch()" style="width:140px">
<option value="">All categories</option>
</select>
<select id="mp-sort" onchange="mpSearch()" style="width:130px">
<option value="downloads">Most popular</option>
<option value="rating">Top rated</option>
<option value="newest">Newest</option>
</select>
<button class="btn btn-primary" style="white-space:nowrap" onclick="mpSearch()">Search</button>
</div>
<div class="templates-grid" id="mp-grid"><p style="color:#555;padding:20px;text-align:center">Loading marketplace...</p></div>
</div>
<!-- Publish form -->
<div class="card">
<h2>➕ Publish Your Agent</h2>
<p style="color:#888;margin-bottom:12px">Share your agent configuration with the community.</p>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
<div>
<label>Agent Name</label>
<input type="text" id="mp-pub-name" placeholder="My Awesome Agent">
<label>Description</label>
<textarea id="mp-pub-desc" placeholder="What does your agent do?"></textarea>
<label>Author</label>
<input type="text" id="mp-pub-author" placeholder="Your name">
</div>
<div>
<label>Category</label>
<select id="mp-pub-cat">
<option value="general">General</option>
<option value="support">Support</option>
<option value="research">Research</option>
<option value="sales">Sales</option>
<option value="engineering">Engineering</option>
<option value="writing">Writing</option>
<option value="data">Data</option>
<option value="productivity">Productivity</option>
</select>
<label>Icon (emoji)</label>
<input type="text" id="mp-pub-icon" value="🤖" style="width:60px">
<label>Tags (comma-separated)</label>
<input type="text" id="mp-pub-tags" placeholder="ai, chatbot, support">
<label>Price ($0 = free)</label>
<input type="number" id="mp-pub-price" value="0" min="0" step="1">
<label>Version</label>
<input type="text" id="mp-pub-ver" value="1.0.0">
</div>
</div>
<label style="margin-top:8px">System Prompt</label>
<textarea id="mp-pub-prompt" placeholder="You are a helpful assistant...">You are a helpful assistant.</textarea>
<label>Model</label>
<select id="mp-pub-model">
<option value="gpt-4o-mini">GPT-4o Mini</option>
<option value="gpt-4o">GPT-4o</option>
</select>
<label>Tools</label>
<div class="tools-grid" id="mp-pub-tools">
<div class="tool-tag" data-tool="calculator" onclick="toggleTool(this)">🔢 Calculator</div>
<div class="tool-tag" data-tool="weather" onclick="toggleTool(this)">🌤️ Weather</div>
<div class="tool-tag" data-tool="web_search" onclick="toggleTool(this)">🔍 Web Search</div>
<div class="tool-tag" data-tool="analyze_image" onclick="toggleTool(this)">👁️ Vision</div>
<div class="tool-tag" data-tool="read_document" onclick="toggleTool(this)">📄 Doc Reader</div>
</div>
<button class="btn btn-primary" style="margin-top:16px;width:100%" onclick="mpPublish()">🚀 Publish to Marketplace</button>
<div id="mp-pub-status" style="margin-top:8px;font-size:13px;color:#888"></div>
</div>
<!-- Detail / review overlay (hidden) -->
<div class="card" id="mp-detail" style="display:none">
<div style="display:flex;justify-content:space-between;align-items:center">
<h2 id="mp-detail-title">Agent Details</h2>
<span style="font-size:12px;color:#555;cursor:pointer" onclick="document.getElementById('mp-detail').style.display='none'">close</span>
</div>
<div id="mp-detail-body"></div>
<div style="margin-top:16px;border-top:1px solid #1a1a2e;padding-top:12px">
<h3 style="font-size:14px;margin-bottom:8px;color:#fff">Leave a Review</h3>
<div style="display:flex;gap:8px;align-items:end">
<div style="flex:1"><label>Comment</label><input type="text" id="mp-rev-comment" placeholder="Great agent!"></div>
<div style="width:80px"><label>Rating</label><select id="mp-rev-rating"><option>5</option><option>4</option><option>3</option><option>2</option><option>1</option></select></div>
<button class="btn btn-primary" style="padding:10px 16px" onclick="mpReview()">Submit</button>
</div>
<div id="mp-reviews"></div>
</div>
</div>
</div>

<!-- EMBED SDK -->
<div class="panel" id="panel-embed">
<div class="card">
<h2>🔌 Embeddable Chat Widget</h2>
<p style="color:#888;margin-bottom:16px">Generate an embeddable widget so other companies can add AgentOS agents to their websites with a single script tag.</p>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
<div>
<label>Widget Name</label>
<input type="text" id="emb-name" value="Support Bot">
<label>Theme</label>
<select id="emb-theme"><option value="dark">Dark</option><option value="light">Light</option></select>
<label>Position</label>
<select id="emb-pos"><option value="bottom-right">Bottom-right</option><option value="bottom-left">Bottom-left</option></select>
<label>Accent Colour</label>
<div style="display:flex;gap:8px;align-items:center">
<input type="color" id="emb-color" value="#6c5ce7" style="width:44px;height:36px;padding:2px;border-radius:4px;border:1px solid #1e1e3a;background:#0a0a14;cursor:pointer">
<input type="text" id="emb-color-hex" value="#6c5ce7" style="flex:1" oninput="document.getElementById('emb-color').value=this.value">
</div>
</div>
<div>
<label>Greeting Message</label>
<input type="text" id="emb-greet" value="Hi! How can I help you today?">
<label>Logo URL (optional)</label>
<input type="text" id="emb-logo" placeholder="https://example.com/logo.png">
<label>Model</label>
<select id="emb-model"><option value="gpt-4o-mini">GPT-4o Mini</option><option value="gpt-4o">GPT-4o</option></select>
<label>System Prompt</label>
<textarea id="emb-prompt" rows="2">You are a helpful assistant.</textarea>
</div>
</div>

<label style="margin-top:8px">Server URL</label>
<input type="text" id="emb-url" value="http://localhost:8000">

<div style="display:flex;gap:8px;margin-top:16px">
<button class="btn btn-primary" style="flex:1" onclick="embGenerate()">Generate Snippet</button>
<button class="btn btn-secondary" style="flex:1" onclick="embPreview()">Live Preview ↗</button>
</div>
</div>

<div class="card" id="emb-output" style="display:none">
<h2>📋 Embed Code</h2>
<p style="color:#888;margin-bottom:8px">Copy this snippet and paste it into any HTML page, just before <code>&lt;/body&gt;</code>:</p>
<pre id="emb-snippet" style="background:#0a0a14;padding:16px;border-radius:8px;font-size:12px;overflow-x:auto;white-space:pre-wrap;word-break:break-all;border:1px solid #1e1e3a;color:#10b981;max-height:260px;overflow-y:auto"></pre>
<button class="btn btn-secondary" style="margin-top:8px" onclick="embCopy()">📋 Copy to Clipboard</button>
<span id="emb-copy-status" style="margin-left:8px;font-size:13px;color:#888"></span>
</div>

<div class="card">
<h2>🐍 Python SDK</h2>
<p style="color:#888;margin-bottom:8px">Use the Python SDK to integrate AgentOS into any backend:</p>
<pre style="background:#0a0a14;padding:16px;border-radius:8px;font-size:13px;overflow-x:auto;white-space:pre-wrap;border:1px solid #1e1e3a;color:#e0e0e0">from agentos.embed import AgentOSClient

client = AgentOSClient(
    base_url="http://localhost:8000",
    api_key="your-api-key",
)

# Single response
response = client.run("How can I help?")
print(response)

# Streaming
for token in client.stream("Tell me a story"):
    print(token, end="", flush=True)

# Browse marketplace
agents = client.list_agents()
for a in agents:
    print(a["name"], a["downloads"])</pre>
</div>
</div>

<!-- SCHEDULER -->
<div class="panel" id="panel-scheduler">
<div class="card">
<h2>⏰ Agent Scheduler</h2>
<p style="color:#888;margin-bottom:16px">Schedule agents to run automatically at intervals or cron times.</p>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
<div>
<label>Agent Name</label>
<input type="text" id="sc-name" value="scheduled-agent" placeholder="scheduled-agent">
<label>Model</label>
<select id="sc-model">
<option value="gpt-4o-mini">GPT-4o Mini</option>
<option value="gpt-4o">GPT-4o</option>
</select>
<label>Query / Task</label>
<textarea id="sc-query" placeholder="What should the agent do each run?">Check the weather in Tokyo and summarize it.</textarea>
</div>
<div>
<label>Schedule Type</label>
<select id="sc-type" onchange="document.getElementById('sc-interval').style.display=this.value==='interval'?'block':'none';document.getElementById('sc-cron').style.display=this.value==='cron'?'block':'none'">
<option value="interval">Interval (every N minutes)</option>
<option value="cron">Cron Expression</option>
</select>
<div id="sc-interval">
<label>Interval</label>
<select id="sc-interval-val">
<option value="30s">Every 30 seconds</option>
<option value="1m">Every 1 minute</option>
<option value="5m" selected>Every 5 minutes</option>
<option value="15m">Every 15 minutes</option>
<option value="1h">Every 1 hour</option>
<option value="1d">Every 1 day</option>
</select>
</div>
<div id="sc-cron" style="display:none">
<label>Cron Expression</label>
<input type="text" id="sc-cron-val" value="0 9 * * *" placeholder="min hour dom month dow">
<span style="font-size:11px;color:#555">e.g. 0 9 * * * = 9am daily</span>
</div>
<label>Tools</label>
<div class="tools-grid">
<div class="tool-tag" data-tool="calculator" onclick="toggleTool(this)">🔢 Calculator</div>
<div class="tool-tag selected" data-tool="weather" onclick="toggleTool(this)">🌤️ Weather</div>
<div class="tool-tag" data-tool="web_search" onclick="toggleTool(this)">🔍 Web Search</div>
<div class="tool-tag" data-tool="analyze_image" onclick="toggleTool(this)">👁️ Vision</div>
<div class="tool-tag" data-tool="read_document" onclick="toggleTool(this)">📄 Doc Reader</div>
</div>
<label>Max Executions (0 = unlimited)</label>
<input type="number" id="sc-max" value="0" min="0">
</div>
</div>
<button class="btn btn-primary" style="margin-top:16px;width:100%" onclick="createScheduledJob()">⏰ Create Scheduled Job</button>
</div>
<div class="card">
<h2>Active Jobs</h2>
<div id="sc-jobs"><p style="color:#555">No scheduled jobs. Create one above.</p></div>
</div>
</div>

<!-- EVENTS -->
<div class="panel" id="panel-events">
<div class="card">
<h2>⚡ Event Bus</h2>
<p style="color:#888;margin-bottom:16px">Fire events and see which agents react. Supports webhooks, timers, agent-to-agent chains, and custom events.</p>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
<div>
<label>Event Name</label>
<input type="text" id="ev-name" value="custom.test" placeholder="webhook.received, agent.completed, custom.*">
<label>Event Data (JSON)</label>
<textarea id="ev-data" style="min-height:80px" placeholder='{"key": "value"}'>{"message": "Hello from the event bus!"}</textarea>
</div>
<div>
<label>Quick Events</label>
<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px">
<button class="btn btn-secondary" style="font-size:12px;padding:6px 12px" onclick="document.getElementById('ev-name').value='webhook.received'">webhook.received</button>
<button class="btn btn-secondary" style="font-size:12px;padding:6px 12px" onclick="document.getElementById('ev-name').value='agent.completed'">agent.completed</button>
<button class="btn btn-secondary" style="font-size:12px;padding:6px 12px" onclick="document.getElementById('ev-name').value='file.changed'">file.changed</button>
<button class="btn btn-secondary" style="font-size:12px;padding:6px 12px" onclick="document.getElementById('ev-name').value='schedule.triggered'">schedule.triggered</button>
<button class="btn btn-secondary" style="font-size:12px;padding:6px 12px" onclick="document.getElementById('ev-name').value='custom.test'">custom.test</button>
</div>
<label>Webhook URL</label>
<div style="background:#08080f;border:1px solid #2a2a4a;border-radius:8px;padding:10px 14px;font-size:13px;color:#888;margin-top:4px;word-break:break-all">
POST <span style="color:#00d4ff">/api/webhook/{event_name}</span>
<br><span style="font-size:11px">Send JSON body to fire events from external services</span>
</div>
</div>
</div>
<button class="btn btn-primary" style="margin-top:16px;width:100%" onclick="emitEvent()">⚡ Emit Event</button>
<div id="ev-result" style="display:none;margin-top:12px;background:#08080f;border:1px solid #1a1a2e;border-radius:8px;padding:12px;font-size:13px"></div>
</div>
<div class="card">
<h2>Registered Listeners</h2>
<div id="ev-listeners"><p style="color:#555">No listeners registered. Use the Python API to register agents.</p></div>
</div>
<div class="card">
<h2>Event History</h2>
<div id="ev-history"><p style="color:#555">No events emitted yet.</p></div>
</div>
</div>

<!-- AUTH / ACCOUNT -->
<div class="panel" id="panel-auth">
<div class="card">
<h2>🔑 Account</h2>
<p style="color:#888;margin-bottom:12px">Register or log in to get an API key. This key is used to track your usage.</p>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
<div>
<label>Email</label>
<input type="email" id="auth-email" placeholder="you@example.com">
<label>Name</label>
<input type="text" id="auth-name" placeholder="Your name">
<div style="display:flex;gap:8px;margin-top:16px">
<button class="btn btn-primary" style="flex:1" onclick="registerUser()">Register</button>
<button class="btn btn-secondary" style="flex:1" onclick="loginUser()">Login</button>
</div>
</div>
<div>
<label>Your API Key</label>
<input type="text" id="auth-apikey" readonly placeholder="Not logged in">
<button class="btn btn-secondary" style="margin-top:8px;width:100%" onclick="copyApiKey()">Copy API Key</button>
<p style="font-size:11px;color:#555;margin-top:8px">API key is stored locally in your browser (localStorage) and sent as <code>X-API-Key</code> for API calls.</p>
</div>
</div>
<div id="auth-message" style="margin-top:12px;font-size:13px;color:#888"></div>
</div>
<div class="card">
<h2>📊 Your Usage</h2>
<div id="auth-usage"><p style="color:#555">Log in to see your usage.</p></div>
</div>
</div>

<!-- A/B TESTING -->
<div class="panel" id="panel-abtest">
<div class="card">
<h2>🧪 A/B Testing</h2>
<p style="color:#888;margin-bottom:16px">Compare two agent configurations on the same set of queries using an LLM-as-judge.</p>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
<div>
<h3 style="font-size:14px;margin-bottom:8px;color:#fff">Agent A</h3>
<label>Name</label>
<input type="text" id="ab-a-name" value="agent-a">
<label>Model</label>
<select id="ab-a-model">
<option value="gpt-4o-mini">GPT-4o Mini</option>
<option value="gpt-4o">GPT-4o</option>
</select>
<label>System Prompt</label>
<textarea id="ab-a-prompt">You are a helpful, concise assistant.</textarea>
<label>Temperature</label>
<input type="number" id="ab-a-temp" value="0.7" step="0.1" min="0" max="2">
</div>
<div>
<h3 style="font-size:14px;margin-bottom:8px;color:#fff">Agent B</h3>
<label>Name</label>
<input type="text" id="ab-b-name" value="agent-b">
<label>Model</label>
<select id="ab-b-model">
<option value="gpt-4o-mini">GPT-4o Mini</option>
<option value="gpt-4o">GPT-4o</option>
</select>
<label>System Prompt</label>
<textarea id="ab-b-prompt">You are a creative assistant. Provide richer, more detailed answers.</textarea>
<label>Temperature</label>
<input type="number" id="ab-b-temp" value="1.0" step="0.1" min="0" max="2">
</div>
</div>
<label style="margin-top:16px">Tools (applied to both agents)</label>
<div class="tools-grid" id="ab-tools">
<div class="tool-tag selected" data-tool="calculator" onclick="toggleTool(this)">🔢 Calculator</div>
<div class="tool-tag" data-tool="weather" onclick="toggleTool(this)">🌤️ Weather</div>
<div class="tool-tag" data-tool="web_search" onclick="toggleTool(this)">🔍 Web Search</div>
<div class="tool-tag" data-tool="analyze_image" onclick="toggleTool(this)">👁️ Vision</div>
<div class="tool-tag" data-tool="read_document" onclick="toggleTool(this)">📄 Doc Reader</div>
</div>
<label style="margin-top:16px">Test Queries (one per line)</label>
<textarea id="ab-queries" style="min-height:100px">Summarize the benefits of AgentOS in one paragraph.
Explain the difference between GPT-4o and GPT-4o-mini.
Give three ideas for onboarding flows for a SaaS dashboard.
Help me debug why my Python script might be slow.
Write a short product description for an AI agent platform.</textarea>
<label style="margin-top:16px">Number of runs (repeats full query set)</label>
<input type="number" id="ab-runs" value="3" min="1" max="10">
<button class="btn btn-primary" style="margin-top:16px;width:100%" onclick="runAbTest()">🧪 Run A/B Test</button>
<div id="ab-status" style="margin-top:8px;font-size:13px;color:#888"></div>
</div>
<div class="card">
<h2>Results</h2>
<div id="ab-results"><p style="color:#555">No A/B test run yet.</p></div>
</div>
</div>

<!-- MULTI-MODAL -->
<div class="panel" id="panel-multimodal">
<div class="card">
<h2>👁️ Multi-modal Analysis</h2>
<p style="color:#888;margin-bottom:16px">Upload an image or document and ask questions about it. Supports images (PNG, JPG, GIF, WebP) and documents (TXT, MD, PDF, CSV, JSON).</p>
<div class="mm-upload-zone" id="mm-dropzone" onclick="document.getElementById('mm-file-input').click()" ondragover="event.preventDefault();this.classList.add('dragover')" ondragleave="this.classList.remove('dragover')" ondrop="handleDrop(event)">
<div class="icon">📁</div>
<p>Click to upload or drag &amp; drop</p>
<p class="sub">Images: PNG, JPG, GIF, WebP &middot; Documents: TXT, MD, PDF, CSV, JSON</p>
</div>
<input type="file" id="mm-file-input" style="display:none" accept=".png,.jpg,.jpeg,.gif,.webp,.txt,.md,.markdown,.pdf,.csv,.json,.log,.rst" onchange="handleFileSelect(this)">
<div class="mm-preview" id="mm-preview">
<div id="mm-preview-content"></div>
<div class="file-info" id="mm-file-info"></div>
<button class="btn btn-secondary" style="margin-top:8px;font-size:12px" onclick="clearUpload()">Remove file</button>
</div>
<div style="display:grid;grid-template-columns:1fr auto;gap:8px;align-items:end">
<div>
<label>Ask a question about your file</label>
<input type="text" id="mm-question" placeholder="Describe this image... / Summarize the key points... / What is the main topic?" value="" onkeydown="if(event.key==='Enter')analyzeFile()">
</div>
<div>
<label>Model</label>
<select id="mm-model" style="width:140px">
<option value="gpt-4o">GPT-4o (vision)</option>
<option value="gpt-4o-mini">GPT-4o Mini</option>
</select>
</div>
</div>
<button class="btn btn-primary" style="margin-top:16px;width:100%" onclick="analyzeFile()" id="mm-analyze-btn">👁️ Analyze</button>
<div id="mm-status" style="margin-top:8px;font-size:13px;color:#888"></div>
<div class="mm-result" id="mm-result"></div>
</div>
<div class="card">
<h2>Or Analyze by URL</h2>
<p style="color:#888;margin-bottom:12px">Paste a public image URL to analyze without uploading.</p>
<label>Image URL</label>
<input type="text" id="mm-url" placeholder="https://example.com/photo.jpg">
<label>Question</label>
<input type="text" id="mm-url-question" placeholder="What is in this image?" onkeydown="if(event.key==='Enter')analyzeUrl()">
<button class="btn btn-primary" style="margin-top:12px;width:100%" onclick="analyzeUrl()">👁️ Analyze URL</button>
<div id="mm-url-status" style="margin-top:8px;font-size:13px;color:#888"></div>
<div class="mm-result" id="mm-url-result"></div>
</div>
</div>

</div>
</div>

<script>
function showPanel(id){
document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
document.getElementById('panel-'+id).classList.add('active');
event.target.classList.add('active');
if(id==='monitor')refreshMonitor();
if(id==='templates')loadTemplates();
if(id==='analytics')refreshAnalytics();
if(id==='branching')brRefresh();
if(id==='events')refreshEvents();
if(id==='auth')refreshAuthUsage();
if(id==='abtest'){}  // A/B panel is static; no periodic refresh needed
if(id==='marketplace')mpRefresh();
}

function toggleTool(el){el.classList.toggle('selected')}

function getSelectedTools(){
return [...document.querySelectorAll('.tool-tag.selected')].map(t=>t.dataset.tool);
}

async function runBuilder(){
const btn=document.getElementById('run-btn');
btn.disabled=true;btn.innerHTML='<span class="loading"></span> Running...';
const body={
name:document.getElementById('b-name').value,
model:document.getElementById('b-model').value,
system_prompt:document.getElementById('b-prompt').value,
query:document.getElementById('b-query').value,
tools:getSelectedTools(),
temperature:parseFloat(document.getElementById('b-temp').value),
budget_limit:parseFloat(document.getElementById('b-budget').value),
};
try{
const headers={'Content-Type':'application/json'};
const apiKey=localStorage.getItem('agentos_api_key');
if(apiKey)headers['X-API-Key']=apiKey;
const r=await fetch('/api/run',{method:'POST',headers:headers,body:JSON.stringify(body)});
const d=await r.json();
document.getElementById('b-response').style.display='block';
document.getElementById('b-response-text').textContent=d.response||'No response';
document.getElementById('b-stats').innerHTML=`
<div class="stat-chip">Cost: <span>$${d.cost.toFixed(4)}</span></div>
<div class="stat-chip">Tokens: <span>${d.tokens}</span></div>
<div class="stat-chip">Tools: <span>${d.tools_used.join(', ')||'none'}</span></div>`;
}catch(e){
document.getElementById('b-response').style.display='block';
document.getElementById('b-response-text').textContent='Error: '+e.message;
}
btn.disabled=false;btn.innerHTML='▶️ Run Agent';
}

async function sendChat(){
const input=document.getElementById('chat-input');
const q=input.value.trim();if(!q)return;
input.value='';
const msgs=document.getElementById('chat-messages');
msgs.innerHTML+=`<div style="text-align:right;margin:8px 0"><span style="background:#00d4ff22;color:#00d4ff;padding:8px 14px;border-radius:12px;display:inline-block">${q}</span></div>`;
msgs.innerHTML+=`<div style="margin:8px 0" id="chat-response"><span style="background:#1a1a2e;padding:8px 14px;border-radius:12px;display:inline-block;max-width:80%;white-space:pre-wrap;line-height:1.5"><span id="chat-streaming"><span class="loading"></span> Thinking...</span><span id="chat-content"></span><span style="font-size:11px;color:#555;display:none" id="chat-stats"></span></span></div>`;
msgs.scrollTop=msgs.scrollHeight;
const wsProto=location.protocol==='https:'?'wss:':'ws:';
const wsUrl=`${wsProto}//${location.host}/ws/chat`;
try{
const ws=new WebSocket(wsUrl);
ws.onmessage=function(ev){
const d=JSON.parse(ev.data);
if(d.type==='token'){
document.getElementById('chat-streaming').style.display='none';
document.getElementById('chat-content').textContent+=d.content;
msgs.scrollTop=msgs.scrollHeight;
}else if(d.type==='done'){
document.getElementById('chat-streaming').style.display='none';
document.getElementById('chat-stats').style.display='block';
document.getElementById('chat-stats').textContent=`$${d.cost.toFixed(4)} · ${d.tokens} tokens`;
}else if(d.type==='error'){
document.getElementById('chat-streaming').innerHTML='<span style="color:#ff4444">Error: '+d.message+'</span>';
document.getElementById('chat-streaming').style.display='inline';
}
};
ws.onerror=function(){
document.getElementById('chat-streaming').innerHTML='<span style="color:#ff4444">WebSocket error</span>';
document.getElementById('chat-streaming').style.display='inline';
ws.close();
};
ws.onclose=function(){
};
await new Promise((resolve,reject)=>{
ws.onopen=resolve;
ws.onerror=()=>reject(new Error('WebSocket failed'));
setTimeout(()=>reject(new Error('timeout')),5000);
});
ws.send(JSON.stringify({
query:q,
name:document.getElementById('b-name').value||'chat-agent',
model:document.getElementById('b-model').value||'gpt-4o-mini',
system_prompt:document.getElementById('b-prompt').value||'You are a helpful assistant.',
tools:getSelectedTools(),
temperature:parseFloat(document.getElementById('b-temp').value)||0.7
}));
}catch(e){
sendChatHttp(q,msgs);
}
}
async function sendChatHttp(q,msgs){
try{
const headers={'Content-Type':'application/json'};
const apiKey=localStorage.getItem('agentos_api_key');
if(apiKey)headers['X-API-Key']=apiKey;
const r=await fetch('/api/run',{method:'POST',headers:headers,body:JSON.stringify({
name:document.getElementById('b-name').value||'chat-agent',
model:document.getElementById('b-model').value||'gpt-4o-mini',
system_prompt:document.getElementById('b-prompt').value||'You are a helpful assistant.',
query:q,tools:getSelectedTools(),temperature:0.7
})});
const d=await r.json();
document.getElementById('chat-streaming').style.display='none';
document.getElementById('chat-content').textContent=d.response||'';
document.getElementById('chat-stats').style.display='block';
document.getElementById('chat-stats').textContent=`$${d.cost.toFixed(4)} · ${d.tokens} tokens`;
msgs.scrollTop=msgs.scrollHeight;
}catch(err){
document.getElementById('chat-streaming').innerHTML='<span style="color:#ff4444">Error: '+err.message+'</span>';
}
}

async function loadTemplates(){
try{
const r=await fetch('/api/templates');
const d=await r.json();
let h='';
d.templates.forEach(t=>{
h+=`<div class="template-card" onclick="loadTemplate('${t.id}')">
<div class="icon">${t.icon}</div><h4>${t.name}</h4><p>${t.description}</p><div class="cat">${t.category}</div></div>`;
});
document.getElementById('templates-list').innerHTML=h;
}catch(e){console.log(e)}
}

function loadTemplate(id){
const prompts={
'customer-support':'You are a friendly customer support agent. Be helpful, empathetic, and solution-oriented. Use the knowledge base for accurate answers.',
'research-assistant':'You are a thorough research assistant. Search for current information, cross-reference sources, and present findings clearly.',
'sales-agent':'You are a professional sales agent. Understand prospect needs, present solutions, handle objections, and guide toward next steps.',
'code-reviewer':'You are an expert code reviewer. Analyze code for bugs, security issues, and best practices. Be constructive.',
'custom':'You are a helpful assistant. Use tools when needed.'
};
document.getElementById('b-name').value=id;
document.getElementById('b-prompt').value=prompts[id]||prompts['custom'];
showPanel('builder');
document.querySelector('.nav-item').classList.add('active');
}

async function refreshMonitor(){
try{
const r=await fetch('/api/overview');
const d=await r.json();
document.getElementById('m-agents').textContent=d.total_agents;
document.getElementById('m-events').textContent=d.total_events;
document.getElementById('m-cost').textContent='$'+d.total_cost.toFixed(4);
document.getElementById('m-status').textContent=d.total_agents>0?'Active':'Ready';
const er=await fetch('/api/events?limit=15');
const events=await er.json();
let h='';
events.reverse().forEach(e=>{
const t=new Date(e.timestamp*1000).toLocaleTimeString();
const cls=e.event_type;
const info=e.event_type==='tool_call'?`${e.data.tool||''}(${JSON.stringify(e.data.args||{}).slice(0,50)})`:
`model:${e.data.model||''} tokens:${(e.data.prompt_tokens||0)+(e.data.completion_tokens||0)}`;
h+=`<div class="event-row"><span style="color:#aa88ff">${e.agent_name}</span><span class="event-type ${cls}">${e.event_type}</span><span>${info}</span><span style="color:#00ff88;text-align:right">$${(e.cost_usd||0).toFixed(4)}</span><span style="color:#666;text-align:right">${(e.latency_ms||0).toFixed(0)}ms</span></div>`;
});
document.getElementById('mon-events').innerHTML=h||'<p style="color:#555;padding:16px">No events yet. Run an agent to see events here.</p>';
}catch(e){console.log(e)}
}

loadTemplates();
setInterval(()=>{if(document.getElementById('panel-monitor').classList.contains('active'))refreshMonitor()},3000);
setInterval(()=>{if(document.getElementById('panel-scheduler').classList.contains('active'))refreshScheduler()},2000);

async function createScheduledJob(){
const type=document.getElementById('sc-type').value;
const body={
agent_name:document.getElementById('sc-name').value,
model:document.getElementById('sc-model').value,
query:document.getElementById('sc-query').value,
tools:[...document.querySelectorAll('#panel-scheduler .tool-tag.selected')].map(t=>t.dataset.tool),
interval:type==='interval'?document.getElementById('sc-interval-val').value:'',
cron:type==='cron'?document.getElementById('sc-cron-val').value:'',
max_executions:parseInt(document.getElementById('sc-max').value)||0,
};
try{
const r=await fetch('/api/scheduler/create',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
const d=await r.json();
if(d.status==='created'){refreshScheduler();}
else{alert('Error: '+(d.message||'Unknown'));}
}catch(e){alert('Error: '+e.message);}
}

async function refreshScheduler(){
try{
const r=await fetch('/api/scheduler/jobs');
const d=await r.json();
let h='';
if(!d.jobs||d.jobs.length===0){
h='<p style="color:#555">No scheduled jobs. Create one above.</p>';
}else{
h+=`<div style="margin-bottom:12px;display:flex;gap:12px">
<div class="stat-chip">Active: <span>${d.overview.active_jobs}</span></div>
<div class="stat-chip">Total Runs: <span>${d.overview.total_executions}</span></div>
<div class="stat-chip">Total Cost: <span>$${d.overview.total_cost.toFixed(4)}</span></div>
</div>`;
d.jobs.forEach(j=>{
const status=j.status==='running'?'🟢 Running':j.status==='pending'?'🔵 Pending':j.status==='paused'?'⏸️ Paused':j.status==='completed'?'✅ Done':'⭕ '+j.status;
const next=j.next_run?new Date(j.next_run*1000).toLocaleTimeString():'—';
const last=j.last_run?new Date(j.last_run*1000).toLocaleTimeString():'never';
const sched=j.interval_seconds>0?(j.interval_seconds<60?j.interval_seconds+'s':j.interval_seconds<3600?Math.round(j.interval_seconds/60)+'m':Math.round(j.interval_seconds/3600)+'h'):j.cron_expression;
h+=`<div style="background:#0a0a14;border:1px solid #1a1a2e;border-radius:8px;padding:14px;margin-bottom:8px">
<div style="display:flex;justify-content:space-between;align-items:center">
<div><strong style="color:#fff">${j.agent_name}</strong> <span style="color:#555;font-size:12px">· ${j.job_id}</span></div>
<div style="display:flex;gap:6px">
<button onclick="fetch('/api/scheduler/${j.status==='paused'?'resume':'pause'}/${j.job_id}',{method:'POST'}).then(()=>refreshScheduler())" style="background:#1a1a2e;border:1px solid #2a2a4a;color:#fff;padding:4px 10px;border-radius:6px;cursor:pointer;font-size:12px">${j.status==='paused'?'▶️ Resume':'⏸️ Pause'}</button>
<button onclick="if(confirm('Delete this job?'))fetch('/api/scheduler/delete/${j.job_id}',{method:'DELETE'}).then(()=>refreshScheduler())" style="background:#2a1a1a;border:1px solid #4a2a2a;color:#ff6666;padding:4px 10px;border-radius:6px;cursor:pointer;font-size:12px">🗑️ Delete</button>
</div>
</div>
<div style="color:#888;font-size:13px;margin-top:6px">${j.query}</div>
<div style="display:flex;gap:16px;margin-top:8px;font-size:12px;color:#666">
<span>${status}</span>
<span>⏱️ ${sched}</span>
<span>Runs: ${j.execution_count}${j.max_executions>0?'/'+j.max_executions:''}</span>
<span>Next: ${next}</span>
<span>Last: ${last}</span>
</div>`;
if(j.history&&j.history.length>0){
h+=`<div style="margin-top:8px;font-size:11px;color:#555">`;
j.history.slice(-3).reverse().forEach(e=>{
const t=new Date(e.started_at*1000).toLocaleTimeString();
const st=e.status==='completed'?'✅':'❌';
h+=`<div style="padding:3px 0;border-top:1px solid #12121f">${st} ${t} · ${e.result.slice(0,100)}${e.result.length>100?'...':''} · $${e.cost_usd.toFixed(4)} · ${e.duration_ms.toFixed(0)}ms</div>`;
});
h+=`</div>`;}
h+=`</div>`;
});
}
document.getElementById('sc-jobs').innerHTML=h;
}catch(e){console.log(e)}
}

async function emitEvent(){
const evName=document.getElementById('ev-name').value.trim();
let evData={};
try{evData=JSON.parse(document.getElementById('ev-data').value);}catch(e){evData={raw:document.getElementById('ev-data').value};}
try{
const r=await fetch('/api/events/emit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({event_name:evName,data:evData})});
const d=await r.json();
const el=document.getElementById('ev-result');
el.style.display='block';
el.innerHTML=`<span style="color:#00ff88">✓ Emitted</span> <strong>${d.event_name}</strong> — ${d.listeners_triggered} listener(s) triggered`;
refreshEvents();
}catch(e){
document.getElementById('ev-result').style.display='block';
document.getElementById('ev-result').innerHTML='<span style="color:#ff4444">Error: '+e.message+'</span>';
}
}

async function refreshEvents(){
try{
const lr=await fetch('/api/events/listeners');
const ld=await lr.json();
let lh='';
if(!ld.listeners||ld.listeners.length===0){
lh='<p style="color:#555">No listeners registered. Use the Python API to register agents.</p>';
}else{
lh+=`<div style="margin-bottom:12px;display:flex;gap:12px">
<div class="stat-chip">Listeners: <span>${ld.overview.total_listeners}</span></div>
<div class="stat-chip">Events Emitted: <span>${ld.overview.total_events_emitted}</span></div>
<div class="stat-chip">Total Executions: <span>${ld.overview.total_executions}</span></div>
</div>`;
ld.listeners.forEach(l=>{
const last=l.last_triggered?new Date(l.last_triggered*1000).toLocaleTimeString():'never';
lh+=`<div style="background:#0a0a14;border:1px solid #1a1a2e;border-radius:8px;padding:12px;margin-bottom:6px;display:grid;grid-template-columns:1fr 1fr 80px 80px;gap:10px;font-size:13px;align-items:center">
<div><span style="color:#00d4ff;font-weight:600">${l.event_pattern}</span></div>
<div><strong style="color:#fff">${l.agent_name}</strong> <span style="color:#555;font-size:11px">· ${l.listener_id}</span></div>
<div style="text-align:right;color:#00ff88">${l.execution_count} runs</div>
<div style="text-align:right;color:#666">${last}</div>
</div>`;
});
}
document.getElementById('ev-listeners').innerHTML=lh;

const hr=await fetch('/api/events/history?limit=15');
const hd=await hr.json();
let hh='';
if(!hd.history||hd.history.length===0){
hh='<p style="color:#555">No events emitted yet.</p>';
}else{
hd.history.reverse().forEach(h=>{
const t=new Date(h.event.timestamp*1000).toLocaleTimeString();
const results=h.results.map(r=>`<span style="color:${r.status==='completed'?'#00ff88':'#ff4444'}">${r.agent_name}: ${(r.result||r.error||'').slice(0,80)}</span>`).join('<br>');
hh+=`<div style="background:#0a0a14;border:1px solid #12121f;border-radius:6px;padding:10px 14px;margin-bottom:4px;font-size:13px">
<div style="display:flex;justify-content:space-between;align-items:center">
<span style="color:#00d4ff;font-weight:600">${h.event.name}</span>
<span style="color:#666;font-size:11px">${t}</span>
</div>
<div style="color:#888;font-size:12px;margin-top:4px">${h.listeners_triggered} listener(s) · source: ${h.event.source||'—'}</div>
${results?'<div style="margin-top:6px;font-size:11px;border-top:1px solid #1a1a2e;padding-top:6px">'+results+'</div>':''}
</div>`;
});
}
document.getElementById('ev-history').innerHTML=hh;
}catch(e){console.log(e)}
}

setInterval(()=>{if(document.getElementById('panel-events').classList.contains('active'))refreshEvents()},3000);

// ── Analytics ──

let _anGranularity='hour';

function switchCostGranularity(g,el){
_anGranularity=g;
document.querySelectorAll('#panel-analytics .an-tab').forEach(t=>t.classList.remove('active'));
el.classList.add('active');
refreshAnalytics();
}

async function refreshAnalytics(){
try{
// Fetch all analytics endpoints in parallel
const [costR,toolsR,modelsR,leaderR]=await Promise.all([
fetch('/api/analytics/cost-over-time?granularity='+_anGranularity),
fetch('/api/analytics/popular-tools'),
fetch('/api/analytics/model-comparison'),
fetch('/api/analytics/agent-leaderboard'),
]);
const costD=await costR.json();
const toolsD=await toolsR.json();
const modelsD=await modelsR.json();
const leaderD=await leaderR.json();

// ── Summary cards ──
const s=leaderD.summary||{};
document.getElementById('an-total-spend').textContent='$'+(s.total_spend||0).toFixed(4);
document.getElementById('an-total-queries').textContent=(s.total_queries||0).toLocaleString();
document.getElementById('an-avg-cost').textContent='$'+(s.avg_cost_per_query||0).toFixed(4);
document.getElementById('an-total-events').textContent=(s.total_events||0).toLocaleString();
document.getElementById('an-total-agents').textContent=s.total_agents||0;

// ── Cost line chart ──
renderCostChart(costD.series||[]);

// ── Tools bar chart ──
renderToolsBars(toolsD.tools||[]);

// ── Model comparison table ──
renderModelTable(modelsD.models||[]);

// ── Agent leaderboard ──
renderLeaderboard(leaderD.leaderboard||[]);

}catch(e){console.log('Analytics refresh error:',e);}
}

function renderCostChart(series){
const line=document.getElementById('an-cost-line');
const fill=document.getElementById('an-cost-fill');
const dotsG=document.getElementById('an-cost-dots');
const yAxis=document.getElementById('an-cost-y');
const xAxis=document.getElementById('an-cost-x');
dotsG.innerHTML='';
if(!series.length){
line.setAttribute('points','');
fill.setAttribute('points','0,100 100,100');
yAxis.innerHTML='<span>0</span><span>0</span>';
xAxis.innerHTML='<span>No data</span>';
return;
}
const maxCost=Math.max(...series.map(s=>s.cost),0.0001);
const n=series.length;
let pts=[];
for(let i=0;i<n;i++){
const x=n===1?50:(i/(n-1))*100;
const y=100-(series[i].cost/maxCost)*100;
pts.push(x.toFixed(2)+','+y.toFixed(2));
dotsG.innerHTML+=`<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="2.5"><title>$${series[i].cost.toFixed(4)} — ${series[i].bucket}</title></circle>`;
}
line.setAttribute('points',pts.join(' '));
fill.setAttribute('points','0,100 '+pts.join(' ')+' 100,100');
// Y axis labels (5 ticks)
let yH='';
for(let i=0;i<5;i++){
const val=(maxCost*(4-i)/4);
yH+=`<span>$${val<0.01?val.toFixed(4):val.toFixed(2)}</span>`;
}
yAxis.innerHTML=yH;
// X axis labels (max 8)
let xH='';
const step=Math.max(1,Math.floor(n/8));
for(let i=0;i<n;i+=step){
const lbl=series[i].bucket;
const short=_anGranularity==='hour'?lbl.slice(11):_anGranularity==='week'?'Wk '+lbl.slice(5):lbl.slice(5);
xH+=`<span>${short}</span>`;
}
xAxis.innerHTML=xH;
}

function renderToolsBars(tools){
const wrap=document.getElementById('an-tools-bars');
const empty=document.getElementById('an-tools-empty');
if(!tools.length){wrap.innerHTML='';empty.style.display='block';return;}
empty.style.display='none';
const maxCount=Math.max(...tools.map(t=>t.count),1);
const colors=['#00d4ff','#00ff88','#ffaa00','#aa88ff','#ff6688','#66ddaa','#dd88ff','#ff8844'];
let h='';
tools.slice(0,10).forEach((t,i)=>{
const pct=Math.max(8,(t.count/maxCount)*100);
const c=colors[i%colors.length];
h+=`<div class="an-bar">
<div class="bar-value">${t.count}</div>
<div class="bar" style="height:${pct}%;background:${c}" title="${t.tool}: ${t.count} calls, avg ${t.avg_latency_ms}ms"></div>
<div class="bar-label">${t.tool}</div>
</div>`;
});
wrap.innerHTML=h;
}

function renderModelTable(models){
const tbody=document.getElementById('an-model-tbody');
if(!models.length){
tbody.innerHTML='<tr><td colspan="7" style="color:#555;text-align:center;padding:20px">No model data yet. Run an agent to see comparisons.</td></tr>';
return;
}
let h='';
models.forEach(m=>{
h+=`<tr>
<td style="color:#00d4ff;font-weight:600">${m.model}</td>
<td>${m.calls}</td>
<td style="color:#ffaa00">$${m.total_cost.toFixed(4)}</td>
<td>$${m.avg_cost.toFixed(6)}</td>
<td>${m.avg_latency_ms.toFixed(0)}ms</td>
<td>${m.avg_tokens.toLocaleString()}</td>
<td>${m.total_tokens.toLocaleString()}</td>
</tr>`;
});
tbody.innerHTML=h;
}

function renderLeaderboard(agents){
const tbody=document.getElementById('an-leader-tbody');
if(!agents.length){
tbody.innerHTML='<tr><td colspan="8" style="color:#555;text-align:center;padding:20px">No agent data yet.</td></tr>';
return;
}
let h='';
agents.forEach((a,i)=>{
const rankCls=i===0?'gold':i===1?'silver':i===2?'bronze':'normal';
const qStr=a.avg_quality!==null?a.avg_quality.toFixed(1):'—';
const qColor=a.avg_quality===null?'#555':a.avg_quality>=8?'#00ff88':a.avg_quality>=6?'#ffaa00':'#ff6666';
const qPct=a.avg_quality!==null?Math.min(100,a.avg_quality*10):0;
h+=`<tr>
<td><span class="an-rank ${rankCls}">${i+1}</span></td>
<td style="color:#fff;font-weight:600">${a.agent}</td>
<td><span style="color:${qColor}">${qStr}</span><span class="an-quality-bar"><span class="fill" style="width:${qPct}%;background:${qColor}"></span></span></td>
<td>${a.total_queries}</td>
<td>${a.total_tool_calls}</td>
<td style="color:#ffaa00">$${a.total_cost.toFixed(4)}</td>
<td>$${a.cost_per_query.toFixed(6)}</td>
<td>${a.total_events}</td>
</tr>`;
});
tbody.innerHTML=h;
}

setInterval(()=>{if(document.getElementById('panel-analytics').classList.contains('active'))refreshAnalytics()},5000);

// ── Embed SDK helpers ──

function embGenerate(){
  const baseUrl=document.getElementById('emb-url').value.trim()||'http://localhost:8000';
  const name=document.getElementById('emb-name').value.trim()||'AgentOS';
  const theme=document.getElementById('emb-theme').value;
  const pos=document.getElementById('emb-pos').value;
  const color=document.getElementById('emb-color').value;
  const greet=document.getElementById('emb-greet').value;
  const logo=document.getElementById('emb-logo').value;
  const model=document.getElementById('emb-model').value;
  const prompt=document.getElementById('emb-prompt').value;
  const snippet=`<script>\n  window.AgentOSConfig = {\n    baseUrl: "${baseUrl}",\n    agentName: "${name}",\n    theme: "${theme}",\n    position: "${pos}",\n    accentColor: "${color}",\n    greeting: "${greet}",${logo?'\n    logo: "'+logo+'",'  :''}\n    model: "${model}",\n    systemPrompt: "${prompt.replace(/"/g,'\\"')}",\n  };\n<\\/script>\n<script src="${baseUrl}/embed/chat.js"><\\/script>`;
  document.getElementById('emb-snippet').textContent=snippet;
  document.getElementById('emb-output').style.display='block';
  document.getElementById('emb-output').scrollIntoView({behavior:'smooth'});
}

function embCopy(){
  const text=document.getElementById('emb-snippet').textContent;
  navigator.clipboard.writeText(text).then(()=>{
    document.getElementById('emb-copy-status').textContent='Copied!';
    setTimeout(()=>{document.getElementById('emb-copy-status').textContent='';},2000);
  });
}

function embPreview(){
  const name=document.getElementById('emb-name').value.trim()||'AgentOS';
  const theme=document.getElementById('emb-theme').value;
  const color=encodeURIComponent(document.getElementById('emb-color').value);
  window.open('/embed/preview?agent_name='+encodeURIComponent(name)+'&theme='+theme+'&accent_color='+color,'_blank');
}

// ── Marketplace helpers ──

let _mpAgents=[];
let _mpDetailId=null;

async function mpRefresh(){
  try{
    const r=await fetch('/api/marketplace/list');
    const d=await r.json();
    _mpAgents=d.agents||[];
    // stats
    const s=d.stats||{};
    document.getElementById('mp-stat-agents').textContent=s.total_agents||0;
    document.getElementById('mp-stat-downloads').textContent=s.total_downloads||0;
    document.getElementById('mp-stat-reviews').textContent=s.total_reviews||0;
    document.getElementById('mp-stat-free').textContent=s.free_count||0;
    // categories
    const sel=document.getElementById('mp-cat');
    const prev=sel.value;
    sel.innerHTML='<option value="">All categories</option>';
    (s.categories||[]).forEach(c=>{sel.innerHTML+='<option value="'+c+'">'+c+'</option>';});
    sel.value=prev;
    mpRenderGrid(_mpAgents);
  }catch(e){console.error('mpRefresh',e);}
}

function mpRenderGrid(agents){
  const g=document.getElementById('mp-grid');
  if(!agents.length){g.innerHTML='<p style="color:#555;padding:20px;text-align:center">No agents found. Be the first to publish!</p>';return;}
  g.innerHTML=agents.map(a=>{
    const stars='★'.repeat(Math.round(a.rating))+'☆'.repeat(5-Math.round(a.rating));
    const priceLabel=a.price===0?'Free':'$'+a.price;
    return '<div class="template-card" style="cursor:pointer" onclick="mpShowDetail(\''+a.id+'\')">'+
      '<div class="icon">'+a.icon+'</div>'+
      '<h4>'+a.name+'</h4>'+
      '<p>'+a.description+'</p>'+
      '<div style="color:#f0b429;font-size:13px;margin:4px 0">'+stars+' <span style="color:#666">('+a.review_count+')</span></div>'+
      '<div class="cat">by '+a.author+' · '+priceLabel+' · ↓'+a.downloads+'</div>'+
      '<button class="btn btn-primary" style="width:100%;margin-top:8px;padding:6px" onclick="event.stopPropagation();mpInstall(\''+a.id+'\')">Install</button>'+
    '</div>';
  }).join('');
}

async function mpSearch(){
  const q=document.getElementById('mp-search').value;
  const cat=document.getElementById('mp-cat').value;
  const sort=document.getElementById('mp-sort').value;
  try{
    const r=await fetch('/api/marketplace/search?q='+encodeURIComponent(q)+'&category='+encodeURIComponent(cat)+'&sort='+sort);
    const d=await r.json();
    mpRenderGrid(d.agents||[]);
  }catch(e){console.error(e);}
}

async function mpInstall(id){
  if(!confirm('Install this agent?'))return;
  try{
    const r=await fetch('/api/marketplace/install/'+id,{method:'POST'});
    const d=await r.json();
    if(d.status==='installed'){
      alert('Installed '+d.agent.name+'! Downloads: '+d.agent.downloads);
      mpRefresh();
    }else{alert(d.message||'Install failed');}
  }catch(e){alert('Install error: '+e);}
}

async function mpShowDetail(id){
  _mpDetailId=id;
  try{
    const r=await fetch('/api/marketplace/'+id);
    const d=await r.json();
    if(d.status==='error'){alert(d.message);return;}
    const det=document.getElementById('mp-detail');
    document.getElementById('mp-detail-title').textContent=d.icon+' '+d.name+' v'+d.version;
    let html='<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px">';
    html+='<div><span style="color:#888">Author:</span> '+d.author+'</div>';
    html+='<div><span style="color:#888">Category:</span> '+d.category+'</div>';
    html+='<div><span style="color:#888">Price:</span> '+(d.price===0?'Free':'$'+d.price)+'</div>';
    html+='<div><span style="color:#888">Downloads:</span> '+d.downloads+'</div>';
    html+='<div><span style="color:#888">Rating:</span> '+d.rating+' ('+d.review_count+' reviews)</div>';
    html+='<div><span style="color:#888">Tags:</span> '+(d.tags||[]).join(', ')+'</div>';
    html+='</div>';
    html+='<p>'+d.description+'</p>';
    html+='<div style="margin-top:12px"><strong>Config:</strong><pre style="background:#0a0a14;padding:8px;border-radius:4px;font-size:12px;max-height:200px;overflow:auto">'+JSON.stringify(d.config||{},null,2)+'</pre></div>';
    document.getElementById('mp-detail-body').innerHTML=html;
    // reviews
    let rhtml='';
    (d.reviews||[]).forEach(rv=>{
      const st='★'.repeat(Math.round(rv.rating))+'☆'.repeat(5-Math.round(rv.rating));
      rhtml+='<div style="padding:8px 0;border-bottom:1px solid #1a1a2e"><span style="color:#f0b429">'+st+'</span> <strong>'+rv.user+'</strong> — '+rv.comment+'</div>';
    });
    document.getElementById('mp-reviews').innerHTML=rhtml;
    det.style.display='block';
    det.scrollIntoView({behavior:'smooth'});
  }catch(e){console.error(e);}
}

async function mpReview(){
  if(!_mpDetailId)return;
  const comment=document.getElementById('mp-rev-comment').value;
  const rating=parseInt(document.getElementById('mp-rev-rating').value)||5;
  try{
    const r=await fetch('/api/marketplace/review/'+_mpDetailId,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user:'anonymous',rating,comment})});
    const d=await r.json();
    if(d.status==='reviewed'){
      document.getElementById('mp-rev-comment').value='';
      mpShowDetail(_mpDetailId);
      mpRefresh();
    }else{alert(d.message||'Review failed');}
  }catch(e){alert('Review error: '+e);}
}

async function mpPublish(){
  const name=document.getElementById('mp-pub-name').value.trim();
  if(!name){alert('Agent name is required');return;}
  const tools=[...document.querySelectorAll('#mp-pub-tools .tool-tag.selected')].map(t=>t.dataset.tool);
  const body={
    name,
    description:document.getElementById('mp-pub-desc').value,
    author:document.getElementById('mp-pub-author').value||'anonymous',
    category:document.getElementById('mp-pub-cat').value,
    icon:document.getElementById('mp-pub-icon').value||'🤖',
    tags:(document.getElementById('mp-pub-tags').value||'').split(',').map(s=>s.trim()).filter(Boolean),
    price:parseFloat(document.getElementById('mp-pub-price').value)||0,
    version:document.getElementById('mp-pub-ver').value||'1.0.0',
    config:{
      name:name,
      model:document.getElementById('mp-pub-model').value,
      system_prompt:document.getElementById('mp-pub-prompt').value,
      tools:tools,
      temperature:0.7,
      max_iterations:10,
    }
  };
  document.getElementById('mp-pub-status').textContent='Publishing...';
  try{
    const r=await fetch('/api/marketplace/publish',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(d.status==='published'){
      document.getElementById('mp-pub-status').innerHTML='<span style="color:#10b981">Published! '+d.agent.name+' is now live.</span>';
      mpRefresh();
    }else{document.getElementById('mp-pub-status').textContent='Error: '+(d.message||'unknown');}
  }catch(e){document.getElementById('mp-pub-status').textContent='Error: '+e;}
}

// ── Branching helpers ──

let _brTreeId=null;
let _brActiveBranch=null;
let _brBranches=[];

async function brNewTree(){
try{
const r=await fetch('/api/branch/new-tree',{method:'POST'});
const d=await r.json();
_brTreeId=d.tree.tree_id;
_brActiveBranch=d.tree.active_branch_id;
_brBranches=d.tree.branches;
brRenderBar();
brRenderMessages([]);
}catch(e){console.log('brNewTree error:',e);}
}

function brRenderBar(){
const bar=document.getElementById('br-bar');
if(!_brBranches.length){bar.innerHTML='<span style="color:#555;font-size:12px">Start a conversation.</span>';return;}
const colors=['#00d4ff','#00ff88','#ffaa00','#aa88ff','#ff6688','#66ddaa'];
let h='';
_brBranches.forEach((b,i)=>{
const c=colors[i%colors.length];
const cls=b.branch_id===_brActiveBranch?'active':'';
const icon=b.is_main?'●':'◆';
h+=`<div class="br-chip ${cls}" onclick="brSwitch('${b.branch_id}')" title="${b.message_count} messages">`;
h+=`<span class="dot" style="background:${c}"></span>${icon} ${b.label} (${b.message_count})</div>`;
});
bar.innerHTML=h;
}

async function brSwitch(branchId){
if(!_brTreeId)return;
try{
const r=await fetch('/api/branch/switch?tree_id='+encodeURIComponent(_brTreeId)+'&branch_id='+encodeURIComponent(branchId),{method:'POST'});
const d=await r.json();
if(d.status==='ok'){
_brActiveBranch=branchId;
await brRefresh();
}
}catch(e){console.log('brSwitch error:',e);}
}

async function brRefresh(){
if(!_brTreeId)return;
try{
const r=await fetch('/api/branches?tree_id='+encodeURIComponent(_brTreeId));
const d=await r.json();
if(d.status==='ok'){
_brBranches=d.tree.branches;
_brActiveBranch=d.tree.active_branch_id;
brRenderBar();
// Fetch messages for active branch
const msgs=_brBranches.find(b=>b.branch_id===_brActiveBranch);
// We need to get messages from the branch — use the compare trick or add_message list
// Just re-render from branch data
await brLoadMessages();
}
}catch(e){console.log('brRefresh error:',e);}
}

async function brLoadMessages(){
if(!_brTreeId||!_brActiveBranch)return;
try{
const r=await fetch('/api/branch/messages?tree_id='+encodeURIComponent(_brTreeId)+'&branch_id='+encodeURIComponent(_brActiveBranch));
const d=await r.json();
if(d.status==='ok'){
brRenderMessages(d.messages);
}
}catch(e){console.log('brLoadMessages error:',e);}
}

function brRenderMessages(msgs){
const el=document.getElementById('br-messages');
if(!msgs||msgs.length===0){
el.innerHTML='<div style="color:#555;text-align:center;padding:40px">No messages yet. Type below to start the conversation.</div>';
return;
}
let h='';
msgs.forEach((m,i)=>{
const align=m.role==='user'?'right':'';
const cls=m.role;
const forkBtn=`<span class="br-fork-btn" onclick="brForkAt(${i})" title="Fork conversation here">🌿 fork</span>`;
h+=`<div class="br-row ${align}">`;
h+=`<span class="br-idx">${i}</span>`;
h+=`<span class="br-msg ${cls}">${escHtml(m.content)}</span>`;
h+=forkBtn;
h+=`</div>`;
});
el.innerHTML=h;
el.scrollTop=el.scrollHeight;
}

function escHtml(s){
if(!s)return'';
return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function brForkAt(msgIndex){
if(!_brTreeId)return;
const label=prompt('Label for the new branch:','branch-'+((_brBranches||[]).length));
if(label===null)return;
try{
const r=await fetch('/api/branch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
tree_id:_brTreeId,at_message_index:msgIndex,label:label,source_branch_id:_brActiveBranch
})});
const d=await r.json();
if(d.status==='created'){
_brActiveBranch=d.branch_id;
_brBranches=d.tree.branches;
brRenderBar();
await brLoadMessages();
}
}catch(e){console.log('brForkAt error:',e);}
}

async function brSend(){
const input=document.getElementById('br-input');
const q=input.value.trim();
if(!q)return;
input.value='';

// Auto-create tree if none exists
if(!_brTreeId){
await brNewTree();
}
try{
const r=await fetch('/api/branch/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
tree_id:_brTreeId,branch_id:_brActiveBranch,role:'user',content:q
})});
const d=await r.json();
if(d.status==='ok'){
await brRefresh();
}else{
console.log('brSend error:',d.message);
}
}catch(e){console.log('brSend error:',e);}
}

async function brCompare(){
if(!_brTreeId||_brBranches.length<2){alert('Need at least 2 branches to compare.');return;}
const ids=_brBranches.map(b=>b.branch_id);
const aLabel=_brBranches[0].label;
const bLabel=_brBranches.length>1?_brBranches[1].label:aLabel;
const a=prompt('Branch A ID or label:',ids[0]);
const b=prompt('Branch B ID or label:',ids.length>1?ids[1]:ids[0]);
if(!a||!b)return;
try{
const r=await fetch('/api/branch/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
tree_id:_brTreeId,branch_a:a,branch_b:b
})});
const d=await r.json();
if(d.status==='ok'){
const c=d.comparison;
const area=document.getElementById('br-compare-area');
area.style.display='block';
let h=`<div class="br-compare-col"><h4>${c.branch_a.label} (${c.branch_a.total_messages} msgs)</h4>`;
c.branch_a_unique.forEach(m=>{
h+=`<div style="margin:4px 0;font-size:12px"><span style="color:${m.role==='user'?'#00d4ff':'#888'}">${m.role}:</span> ${escHtml((m.content||'').slice(0,200))}</div>`;
});
if(!c.branch_a_unique.length)h+='<div style="color:#555;font-size:12px">No unique messages.</div>';
h+=`</div>`;
h+=`<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:0 4px"><div style="font-size:11px;color:#555">${c.shared_count} shared</div><div style="font-size:16px;color:#555">⇔</div></div>`;
h+=`<div class="br-compare-col"><h4>${c.branch_b.label} (${c.branch_b.total_messages} msgs)</h4>`;
c.branch_b_unique.forEach(m=>{
h+=`<div style="margin:4px 0;font-size:12px"><span style="color:${m.role==='user'?'#00d4ff':'#888'}">${m.role}:</span> ${escHtml((m.content||'').slice(0,200))}</div>`;
});
if(!c.branch_b_unique.length)h+='<div style="color:#555;font-size:12px">No unique messages.</div>';
h+=`</div>`;
document.getElementById('br-compare-cols').innerHTML=h;
}
}catch(e){console.log('brCompare error:',e);}
}

async function brMerge(){
if(!_brTreeId||_brBranches.length<2){alert('Need at least 2 branches to merge.');return;}
const ids=_brBranches.map(b=>b.branch_id);
const a=prompt('Branch A ID:',ids[0]);
const b=prompt('Branch B ID:',ids.length>1?ids[1]:ids[0]);
const label=prompt('Label for merged branch:','merged');
if(!a||!b||!label)return;
try{
const r=await fetch('/api/branch/merge',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
tree_id:_brTreeId,branch_a:a,branch_b:b,label:label
})});
const d=await r.json();
if(d.status==='ok'){
_brActiveBranch=d.merged_branch_id;
_brBranches=d.tree.branches;
brRenderBar();
await brLoadMessages();
}
}catch(e){console.log('brMerge error:',e);}
}

// ── Multi-modal helpers ──

let _mmFilePath=null;
let _mmFileType=null;

function handleDrop(e){
e.preventDefault();
e.currentTarget.classList.remove('dragover');
if(e.dataTransfer.files.length>0)uploadFile(e.dataTransfer.files[0]);
}

function handleFileSelect(input){
if(input.files.length>0)uploadFile(input.files[0]);
}

async function uploadFile(file){
const statusEl=document.getElementById('mm-status');
statusEl.style.color='#888';
statusEl.textContent='Uploading...';
const formData=new FormData();
formData.append('file',file);
try{
const r=await fetch('/api/upload',{method:'POST',body:formData});
const d=await r.json();
if(d.status!=='uploaded'){
statusEl.style.color='#ff6666';
statusEl.textContent='Error: '+(d.message||'Upload failed');
return;
}
_mmFilePath=d.file_path;
_mmFileType=d.file_type;
statusEl.style.color='#00ff88';
statusEl.textContent='Uploaded: '+d.file_name+' ('+formatBytes(d.size_bytes)+')';
// Show preview
const previewEl=document.getElementById('mm-preview');
const contentEl=document.getElementById('mm-preview-content');
const infoEl=document.getElementById('mm-file-info');
previewEl.style.display='block';
if(d.file_type==='image'){
const url=URL.createObjectURL(file);
contentEl.innerHTML='<img src="'+url+'" alt="Preview">';
document.getElementById('mm-question').placeholder='What is in this image? Describe the details...';
}else{
contentEl.innerHTML='<div style="font-size:36px;text-align:center;padding:20px">📄</div>';
document.getElementById('mm-question').placeholder='Summarize this document... / What are the key points?';
}
infoEl.textContent=d.file_name+' · '+d.file_type+' · '+formatBytes(d.size_bytes);
}catch(e){
statusEl.style.color='#ff6666';
statusEl.textContent='Error: '+e.message;
}
}

function clearUpload(){
_mmFilePath=null;
_mmFileType=null;
document.getElementById('mm-preview').style.display='none';
document.getElementById('mm-preview-content').innerHTML='';
document.getElementById('mm-file-info').textContent='';
document.getElementById('mm-status').textContent='';
document.getElementById('mm-result').style.display='none';
document.getElementById('mm-file-input').value='';
}

function formatBytes(b){
if(b<1024)return b+' B';
if(b<1048576)return (b/1024).toFixed(1)+' KB';
return (b/1048576).toFixed(1)+' MB';
}

async function analyzeFile(){
if(!_mmFilePath){
document.getElementById('mm-status').style.color='#ff6666';
document.getElementById('mm-status').textContent='Please upload a file first.';
return;
}
const btn=document.getElementById('mm-analyze-btn');
btn.disabled=true;btn.innerHTML='<span class="loading"></span> Analyzing...';
const statusEl=document.getElementById('mm-status');
const resultEl=document.getElementById('mm-result');
statusEl.style.color='#888';
statusEl.textContent='Analyzing...';
try{
const r=await fetch('/api/analyze-file',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
file_path:_mmFilePath,
question:document.getElementById('mm-question').value||'',
model:document.getElementById('mm-model').value
})});
const d=await r.json();
if(d.status==='ok'){
statusEl.style.color='#00ff88';
let info=_mmFileType==='image'?'Image analysis complete':'Document analysis complete';
if(d.cost)info+=' · $'+d.cost.toFixed(4);
if(d.tokens)info+=' · '+d.tokens+' tokens';
statusEl.textContent=info;
resultEl.style.display='block';
resultEl.textContent=d.analysis||'No result';
}else{
statusEl.style.color='#ff6666';
statusEl.textContent='Error: '+(d.message||'Analysis failed');
}
}catch(e){
statusEl.style.color='#ff6666';
statusEl.textContent='Error: '+e.message;
}
btn.disabled=false;btn.innerHTML='👁️ Analyze';
}

async function analyzeUrl(){
const url=document.getElementById('mm-url').value.trim();
if(!url){
document.getElementById('mm-url-status').style.color='#ff6666';
document.getElementById('mm-url-status').textContent='Please enter an image URL.';
return;
}
const statusEl=document.getElementById('mm-url-status');
const resultEl=document.getElementById('mm-url-result');
statusEl.style.color='#888';
statusEl.textContent='Analyzing image from URL...';
try{
const r=await fetch('/api/analyze-file',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
file_path:url,
question:document.getElementById('mm-url-question').value||'Describe this image in detail.',
model:'gpt-4o'
})});
const d=await r.json();
if(d.status==='ok'){
statusEl.style.color='#00ff88';
statusEl.textContent='Analysis complete';
resultEl.style.display='block';
resultEl.textContent=d.analysis||'No result';
}else{
statusEl.style.color='#ff6666';
statusEl.textContent='Error: '+(d.message||'Analysis failed');
}
}catch(e){
statusEl.style.color='#ff6666';
statusEl.textContent='Error: '+e.message;
}
}

// ── Auth helpers ──

function setUserStatus(email){
const el=document.getElementById('user-status');
if(!el)return;
if(email){
el.textContent='Logged in as '+email;
}else{
el.textContent='Not logged in';
}
}

async function registerUser(){
const email=document.getElementById('auth-email').value.trim();
const name=document.getElementById('auth-name').value.trim();
const msgEl=document.getElementById('auth-message');
msgEl.style.color='#888';
msgEl.textContent='Registering...';
try{
const r=await fetch('/api/auth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email,name})});
const d=await r.json();
if(d.status==='created'){
localStorage.setItem('agentos_api_key',d.api_key);
document.getElementById('auth-apikey').value=d.api_key;
setUserStatus(d.user.email);
msgEl.style.color='#00ff88';
msgEl.textContent='User created. API key stored in this browser.';
refreshAuthUsage();
}else{
msgEl.style.color='#ff6666';
msgEl.textContent='Error: '+(d.message||'Unable to register');
}
}catch(e){
msgEl.style.color='#ff6666';
msgEl.textContent='Error: '+e.message;
}
}

async function loginUser(){
const email=document.getElementById('auth-email').value.trim();
const msgEl=document.getElementById('auth-message');
msgEl.style.color='#888';
msgEl.textContent='Logging in...';
try{
const r=await fetch('/api/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email})});
const d=await r.json();
if(d.status==='ok'){
localStorage.setItem('agentos_api_key',d.api_key);
document.getElementById('auth-apikey').value=d.api_key;
setUserStatus(d.user.email);
msgEl.style.color='#00ff88';
msgEl.textContent='Logged in. API key stored in this browser.';
refreshAuthUsage();
}else{
msgEl.style.color='#ff6666';
msgEl.textContent='Error: '+(d.message||'Unable to login');
}
}catch(e){
msgEl.style.color='#ff6666';
msgEl.textContent='Error: '+e.message;
}
}

function copyApiKey(){
const el=document.getElementById('auth-apikey');
if(!el.value)return;
navigator.clipboard.writeText(el.value).then(()=>{
const msgEl=document.getElementById('auth-message');
msgEl.style.color='#00d4ff';
msgEl.textContent='API key copied to clipboard.';
});
}

async function refreshAuthUsage(){
const apiKey=localStorage.getItem('agentos_api_key');
const usageEl=document.getElementById('auth-usage');
if(!apiKey){
setUserStatus(null);
usageEl.innerHTML='<p style="color:#555">Log in to see your usage.</p>';
return;
}
document.getElementById('auth-apikey').value=apiKey;
try{
const r=await fetch('/api/auth/usage?period=month',{headers:{'X-API-Key':apiKey}});
const d=await r.json();
if(d.status!=='ok'){
usageEl.innerHTML='<p style="color:#ff6666">Error: '+(d.message||'Unable to load usage')+'</p>';
return;
}
setUserStatus(d.total && d.total.user_id ? d.total.user_id : 'user');
const total=d.total;
const window=d.window;
usageEl.innerHTML=`
<div class="stats-row">
<div class="stat-chip">Total Queries: <span>${total.queries}</span></div>
<div class="stat-chip">Total Tokens: <span>${total.tokens}</span></div>
<div class="stat-chip">Total Cost: <span>$${total.cost.toFixed(4)}</span></div>
</div>
<div class="stats-row" style="margin-top:8px">
<div class="stat-chip">Last ${d.period} · Queries: <span>${window.queries}</span></div>
<div class="stat-chip">Last ${d.period} · Tokens: <span>${window.tokens}</span></div>
<div class="stat-chip">Last ${d.period} · Cost: <span>$${window.cost.toFixed(4)}</span></div>
</div>`;
}catch(e){
usageEl.innerHTML='<p style="color:#ff6666">Error: '+e.message+'</p>';
}
}

async function runAbTest(){
const statusEl=document.getElementById('ab-status');
const resultsEl=document.getElementById('ab-results');
statusEl.style.color='#888';
statusEl.textContent='Running A/B test... this may take a while.';
resultsEl.innerHTML='<p style="color:#888">Running A/B test...</p>';
const tools=[...document.querySelectorAll('#ab-tools .tool-tag.selected')].map(t=>t.dataset.tool);
const queriesRaw=document.getElementById('ab-queries').value.split('\\n').map(q=>q.trim()).filter(q=>q);
const runs=parseInt(document.getElementById('ab-runs').value)||3;
if(!queriesRaw.length){
statusEl.style.color='#ff6666';
statusEl.textContent='Please enter at least one test query.';
return;
}
const body={
agent_a:{
name:document.getElementById('ab-a-name').value||'agent-a',
model:document.getElementById('ab-a-model').value||'gpt-4o-mini',
system_prompt:document.getElementById('ab-a-prompt').value||'You are a helpful, concise assistant.',
temperature:parseFloat(document.getElementById('ab-a-temp').value)||0.7,
tools:tools,
},
agent_b:{
name:document.getElementById('ab-b-name').value||'agent-b',
model:document.getElementById('ab-b-model').value||'gpt-4o-mini',
system_prompt:document.getElementById('ab-b-prompt').value||'You are a creative assistant.',
temperature:parseFloat(document.getElementById('ab-b-temp').value)||1.0,
tools:tools,
},
queries:queriesRaw,
num_runs:runs,
};
try{
const headers={'Content-Type':'application/json'};
const apiKey=localStorage.getItem('agentos_api_key');
if(apiKey)headers['X-API-Key']=apiKey;
const r=await fetch('/api/ab-test',{method:'POST',headers:headers,body:JSON.stringify(body)});
const d=await r.json();
if(d.status!=='ok'){
statusEl.style.color='#ff6666';
statusEl.textContent='Error: '+(d.message||'A/B test failed');
return;
}
const rep=d.report;
statusEl.style.color='#00ff88';
statusEl.textContent=`Winner: ${rep.winner} (confidence ${(rep.confidence*100).toFixed(1)}%)`;
const a=rep.scores.agent_a;
const b=rep.scores.agent_b;
let h='';
h+=`<div class="stats-row">
<div class="stat-chip">Winner: <span>${rep.winner}</span></div>
<div class="stat-chip">Confidence: <span>${(rep.confidence*100).toFixed(1)}%</span></div>
</div>`;
h+=`<div class="stats-row" style="margin-top:8px">
<div class="stat-chip">A · Avg Overall: <span>${a.avg_overall.toFixed(2)}</span></div>
<div class="stat-chip">A · Win Rate: <span>${(a.win_rate*100).toFixed(1)}%</span></div>
<div class="stat-chip">A · Pass Rate: <span>${a.pass_rate.toFixed(1)}%</span></div>
</div>`;
h+=`<div class="stats-row" style="margin-top:4px">
<div class="stat-chip">B · Avg Overall: <span>${b.avg_overall.toFixed(2)}</span></div>
<div class="stat-chip">B · Win Rate: <span>${(b.win_rate*100).toFixed(1)}%</span></div>
<div class="stat-chip">B · Pass Rate: <span>${b.pass_rate.toFixed(1)}%</span></div>
</div>`;
if(rep.per_query&&rep.per_query.length){
h+=`<div style="margin-top:12px;font-size:13px">
<h3 style="font-size:14px;margin-bottom:6px">Per-query breakdown</h3>`;
rep.per_query.forEach((r,i)=>{
const icon=r.winner==='agent_a'?'A':r.winner==='agent_b'?'B':'=';
h+=`<div style="background:#0a0a14;border:1px solid #1a1a2e;border-radius:8px;padding:10px 12px;margin-bottom:4px">
<div style="display:flex;justify-content:space-between;font-size:13px">
<span><strong>Q${i+1}</strong> ${r.query}</span>
<span style="color:#00d4ff">Winner: ${icon}</span>
</div>
<div style="font-size:12px;color:#888;margin-top:4px">
A: ${r.score_a.toFixed(1)} · B: ${r.score_b.toFixed(1)}
</div>
</div>`;
});
h+='</div>';
}
resultsEl.innerHTML=h;
}catch(e){
statusEl.style.color='#ff6666';
statusEl.textContent='Error: '+e.message;
resultsEl.innerHTML='<p style="color:#ff6666">Error: '+e.message+'</p>';
}
}

// Initialize API key if already stored
(function initAuth(){
const apiKey=localStorage.getItem('agentos_api_key');
if(apiKey){
document.getElementById('auth-apikey').value=apiKey;
setUserStatus('user');
refreshAuthUsage();
}
})();

</script>
</body>
</html>
"""