from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from agentos.monitor.store import store
from agentos.core.types import AgentEvent

app = FastAPI(title="AgentOS Monitor", version="0.3.0")


@app.get("/")
def dashboard():
    """Serve the live dashboard."""
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/api/overview")
def overview():
    """Get overview of all agents."""
    return store.get_overview()


@app.get("/api/agents/{name}")
def get_agent(name: str):
    """Get details for one agent."""
    agent = store.get_agent(name)
    if not agent:
        return {"error": "Agent not found"}
    return agent


@app.get("/api/events")
def get_events(agent: str | None = None, limit: int = 50):
    """Get recent events."""
    return store.get_events(agent_name=agent, limit=limit)


@app.post("/api/events")
def log_event(event: AgentEvent):
    """Log an agent event."""
    store.log_event(event)
    return {"status": "ok"}


@app.get("/api/drift/{name}")
def check_drift(name: str):
    """Check quality drift for an agent."""
    drift = store.detect_drift(name)
    return drift or {"status": "no drift detected"}


# ── The entire dashboard as inline HTML ──

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgentOS Monitor</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0f; color: #e0e0e0; }
  .header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px 32px; border-bottom: 1px solid #2a2a4a; display: flex; justify-content: space-between; align-items: center; }
  .header h1 { font-size: 22px; color: #00d4ff; }
  .header .status { color: #00ff88; font-size: 14px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; padding: 24px 32px; }
  .card { background: #12121f; border: 1px solid #2a2a4a; border-radius: 12px; padding: 20px; }
  .card .label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
  .card .value { font-size: 32px; font-weight: 700; margin-top: 8px; }
  .card .value.blue { color: #00d4ff; }
  .card .value.green { color: #00ff88; }
  .card .value.yellow { color: #ffaa00; }
  .card .value.purple { color: #aa88ff; }
  .agents-section { padding: 0 32px 24px; }
  .agents-section h2 { font-size: 18px; color: #ccc; margin-bottom: 16px; }
  .agent-card { background: #12121f; border: 1px solid #2a2a4a; border-radius: 12px; padding: 20px; margin-bottom: 12px; display: grid; grid-template-columns: 2fr 1fr 1fr 1fr 1fr 1fr; align-items: center; gap: 16px; }
  .agent-name { font-size: 16px; font-weight: 600; color: #00d4ff; }
  .agent-stat { text-align: center; }
  .agent-stat .num { font-size: 20px; font-weight: 700; }
  .agent-stat .lbl { font-size: 11px; color: #888; margin-top: 4px; }
  .status-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
  .status-running { background: #00ff8822; color: #00ff88; }
  .events-section { padding: 0 32px 32px; }
  .events-section h2 { font-size: 18px; color: #ccc; margin-bottom: 16px; }
  .event-row { background: #12121f; border: 1px solid #1a1a2e; padding: 12px 16px; margin-bottom: 4px; border-radius: 8px; display: grid; grid-template-columns: 140px 100px 1fr 80px 80px; gap: 12px; font-size: 13px; align-items: center; }
  .event-type { padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; text-align: center; }
  .event-type.llm_call { background: #00d4ff22; color: #00d4ff; }
  .event-type.tool_call { background: #ffaa0022; color: #ffaa00; }
  .event-agent { color: #aa88ff; }
  .event-cost { color: #00ff88; text-align: right; }
  .event-latency { color: #888; text-align: right; }
  .refresh-note { text-align: center; padding: 16px; color: #555; font-size: 13px; }
</style>
</head>
<body>

<div class="header">
  <h1>🤖 AgentOS Monitor</h1>
  <div class="status">● Live — Auto-refreshing every 3s</div>
</div>

<div class="grid" id="overview">
  <div class="card"><div class="label">Agents</div><div class="value blue" id="total-agents">-</div></div>
  <div class="card"><div class="label">Total Events</div><div class="value green" id="total-events">-</div></div>
  <div class="card"><div class="label">Total Cost</div><div class="value yellow" id="total-cost">-</div></div>
  <div class="card"><div class="label">Status</div><div class="value purple" id="system-status">Ready</div></div>
</div>

<div class="agents-section">
  <h2>Agents</h2>
  <div id="agents-list"><div style="color:#555;padding:20px;">No agents running yet. Run an agent to see it here.</div></div>
</div>

<div class="events-section">
  <h2>Live Event Stream</h2>
  <div id="events-list"><div style="color:#555;padding:20px;">No events yet.</div></div>
</div>

<div class="refresh-note">Dashboard auto-refreshes every 3 seconds</div>

<script>
async function refresh() {
  try {
    const ov = await (await fetch('/api/overview')).json();
    document.getElementById('total-agents').textContent = ov.total_agents;
    document.getElementById('total-events').textContent = ov.total_events.toLocaleString();
    document.getElementById('total-cost').textContent = '$' + ov.total_cost.toFixed(4);
    document.getElementById('system-status').textContent = ov.total_agents > 0 ? 'Active' : 'Ready';

    let agentsHtml = '';
    for (const a of ov.agents) {
      agentsHtml += `
        <div class="agent-card">
          <div><span class="agent-name">${a.name}</span><br><span class="status-badge status-running">running</span></div>
          <div class="agent-stat"><div class="num" style="color:#00d4ff">${a.total_llm_calls}</div><div class="lbl">LLM Calls</div></div>
          <div class="agent-stat"><div class="num" style="color:#ffaa00">${a.total_tool_calls}</div><div class="lbl">Tool Calls</div></div>
          <div class="agent-stat"><div class="num" style="color:#00ff88">${a.total_tokens.toLocaleString()}</div><div class="lbl">Tokens</div></div>
          <div class="agent-stat"><div class="num" style="color:#ffaa00">$${a.total_cost.toFixed(4)}</div><div class="lbl">Cost</div></div>
          <div class="agent-stat"><div class="num" style="color:#aa88ff">${a.total_events}</div><div class="lbl">Events</div></div>
        </div>`;
    }
    document.getElementById('agents-list').innerHTML = agentsHtml || '<div style="color:#555;padding:20px;">No agents yet.</div>';

    const events = await (await fetch('/api/events?limit=20')).json();
    let eventsHtml = '';
    for (const e of events.reverse()) {
      const time = new Date(e.timestamp * 1000).toLocaleTimeString();
      const typeClass = e.event_type;
      const data = e.data || {};
      let info = '';
      if (e.event_type === 'tool_call') info = `${data.tool || ''}(${JSON.stringify(data.args || {}).slice(0,60)})`;
      else if (e.event_type === 'llm_call') info = `model: ${data.model || ''} | tokens: ${data.prompt_tokens || 0}+${data.completion_tokens || 0}`;
      else info = JSON.stringify(data).slice(0, 80);

      eventsHtml += `
        <div class="event-row">
          <span class="event-agent">${e.agent_name}</span>
          <span class="event-type ${typeClass}">${e.event_type}</span>
          <span>${info}</span>
          <span class="event-cost">$${(e.cost_usd || 0).toFixed(4)}</span>
          <span class="event-latency">${(e.latency_ms || 0).toFixed(0)}ms</span>
        </div>`;
    }
    document.getElementById('events-list').innerHTML = eventsHtml || '<div style="color:#555;padding:20px;">No events yet.</div>';
  } catch(err) { console.log('Refresh error:', err); }
}

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""
