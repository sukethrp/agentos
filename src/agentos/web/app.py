"""AgentOS Web UI — Visual Agent Builder + Marketplace + Dashboard."""

from __future__ import annotations
import asyncio
import json
import queue
import threading
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from agentos.monitor.store import store
from agentos.core.types import AgentEvent
from agentos.tools import get_builtin_tools
from agentos.scheduler import AgentScheduler
from agentos.events import event_bus, WebhookTrigger
from agentos.auth import User, create_user, get_current_user, get_user_by_email
from agentos.auth.usage import usage_tracker
from agentos.core.ab_testing import ABTest

load_dotenv()

app = FastAPI(title="AgentOS Platform", version="0.1.0")

# Global scheduler instance
_scheduler = AgentScheduler(max_concurrent=3)
_scheduler.start()

# Global webhook trigger (passive — fires when /api/webhook/ is hit)
_webhook_trigger = WebhookTrigger(name="web-webhook")
_webhook_trigger.start()


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
<div class="nav-item" onclick="showPanel('monitor')">📊 Monitor</div>
<div class="nav-item" onclick="showPanel('scheduler')">⏰ Scheduler</div>
<div class="nav-item" onclick="showPanel('events')">⚡ Events</div>
<div class="nav-item" onclick="showPanel('abtest')">🧪 A/B Testing</div>
<h3>Manage</h3>
<div class="nav-item" onclick="showPanel('auth')">🔑 Account & Usage</div>
<div class="nav-item" onclick="showPanel('marketplace')">🏪 Marketplace</div>
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

<!-- MARKETPLACE -->
<div class="panel" id="panel-marketplace">
<div class="card">
<h2>🏪 Agent Marketplace</h2>
<p style="color:#888;margin-bottom:24px">Share and discover agent templates from the community. Coming soon — be the first to publish!</p>
<div class="templates-grid">
<div class="template-card" style="border-style:dashed;text-align:center;padding:40px">
<div style="font-size:40px;margin-bottom:8px">➕</div>
<h4>Publish Your Agent</h4>
<p>Share your agent template with the community and earn revenue.</p>
<button class="btn btn-secondary" style="margin-top:12px">Coming Soon</button>
</div>
<div class="template-card">
<div class="icon">🎧</div>
<h4>Customer Support Pro</h4>
<p>Advanced support with ticket management and escalation.</p>
<div class="cat">by AgentOS Team · Free</div>
</div>
<div class="template-card">
<div class="icon">📊</div>
<h4>Data Analyst</h4>
<p>Analyze datasets, create reports, and visualize trends.</p>
<div class="cat">by community · $29</div>
</div>
<div class="template-card">
<div class="icon">✍️</div>
<h4>Content Writer</h4>
<p>Write blog posts, emails, and social media content.</p>
<div class="cat">by community · $19</div>
</div>
</div>
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
if(id==='events')refreshEvents();
if(id==='auth')refreshAuthUsage();
if(id==='abtest'){}  // A/B panel is static; no periodic refresh needed
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