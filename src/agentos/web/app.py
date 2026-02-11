"""AgentOS Web UI — Visual Agent Builder + Marketplace + Dashboard."""

from __future__ import annotations
import asyncio
import json
import queue
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from agentos.monitor.store import store
from agentos.core.types import AgentEvent
from agentos.tools import get_builtin_tools
from agentos.scheduler import AgentScheduler

load_dotenv()

app = FastAPI(title="AgentOS Platform", version="0.1.0")

# Global scheduler instance
_scheduler = AgentScheduler(max_concurrent=3)
_scheduler.start()


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
def run_agent(req: RunRequest):
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
.topbar .status{color:#00ff88;font-size:13px}
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
<div class="status">● Online</div>
</div>
<div class="sidebar">
<h3>Build</h3>
<div class="nav-item active" onclick="showPanel('builder')">🛠️ Agent Builder</div>
<div class="nav-item" onclick="showPanel('templates')">📦 Templates</div>
<h3>Operate</h3>
<div class="nav-item" onclick="showPanel('chat')">💬 Chat</div>
<div class="nav-item" onclick="showPanel('monitor')">📊 Monitor</div>
<div class="nav-item" onclick="showPanel('scheduler')">⏰ Scheduler</div>
<h3>Manage</h3>
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
const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
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
const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
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
</script>
</body>
</html>
"""