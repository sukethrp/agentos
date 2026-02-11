"""Embeddable Chat Widget — generates self-contained HTML/JS/CSS snippets.

Usage:
    from agentos.embed.widget import generate_widget, generate_widget_js

    # Full inline widget (single HTML block):
    html = generate_widget(agent_name="Support Bot", theme="dark")

    # External JS approach (served from /embed/chat.js):
    js = generate_widget_js()
"""

from __future__ import annotations

import json
from typing import Any


# ── Colour palettes ──────────────────────────────────────────────────────────

_THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#0d0d1a",
        "card": "#12122a",
        "border": "#1e1e3a",
        "text": "#e0e0e0",
        "muted": "#888",
        "accent": "#6c5ce7",
        "accent_hover": "#7c6cf7",
        "user_bg": "#1a1a3e",
        "bot_bg": "#0a0a14",
        "input_bg": "#0a0a14",
        "shadow": "rgba(0,0,0,.45)",
    },
    "light": {
        "bg": "#ffffff",
        "card": "#f8f9fa",
        "border": "#e0e0e0",
        "text": "#1a1a2e",
        "muted": "#666",
        "accent": "#6c5ce7",
        "accent_hover": "#5a4bd6",
        "user_bg": "#eef0ff",
        "bot_bg": "#f3f3f8",
        "input_bg": "#ffffff",
        "shadow": "rgba(0,0,0,.15)",
    },
}


def _get_theme(name: str) -> dict[str, str]:
    return _THEMES.get(name, _THEMES["dark"])


# ── Widget JavaScript (standalone, served at /embed/chat.js) ─────────────────

def generate_widget_js() -> str:
    """Return the full JS source for the embeddable chat widget.

    The script reads its configuration from ``window.AgentOSConfig`` which the
    host page must set *before* loading the script::

        <script>
          window.AgentOSConfig = {
            apiKey: "ak_...",
            baseUrl: "http://localhost:8000",
            agentName: "Support Bot",
            theme: "dark",          // "dark" | "light"
            position: "bottom-right", // "bottom-right" | "bottom-left"
            accentColor: "#6c5ce7",
            logo: "",               // optional URL
            greeting: "Hi! How can I help you today?",
          };
        </script>
        <script src="http://localhost:8000/embed/chat.js"></script>
    """
    return r"""
(function(){
"use strict";

/* ── Read host config ─────────────────────────────────────────── */
var C = window.AgentOSConfig || {};
var BASE   = (C.baseUrl   || "").replace(/\/+$/,"");
var API_KEY= C.apiKey     || "";
var AGENT  = C.agentName  || "AgentOS";
var THEME  = C.theme      || "dark";
var POS    = C.position   || "bottom-right";
var ACCENT = C.accentColor|| "#6c5ce7";
var LOGO   = C.logo       || "";
var GREET  = C.greeting   || "Hi! How can I help you today?";
var MODEL  = C.model      || "gpt-4o-mini";
var PROMPT = C.systemPrompt || "You are a helpful assistant.";
var TOOLS  = C.tools      || [];

/* ── Theme palettes ───────────────────────────────────────────── */
var themes = {
  dark: {bg:"#0d0d1a",card:"#12122a",border:"#1e1e3a",text:"#e0e0e0",muted:"#888",
         userBg:"#1a1a3e",botBg:"#0a0a14",inputBg:"#0a0a14",shadow:"rgba(0,0,0,.45)"},
  light:{bg:"#ffffff",card:"#f8f9fa",border:"#e0e0e0",text:"#1a1a2e",muted:"#666",
         userBg:"#eef0ff",botBg:"#f3f3f8",inputBg:"#ffffff",shadow:"rgba(0,0,0,.15)"}
};
var T = themes[THEME] || themes.dark;

/* ── Helpers ──────────────────────────────────────────────────── */
function el(tag,attrs,children){
  var e=document.createElement(tag);
  if(attrs) Object.keys(attrs).forEach(function(k){
    if(k==="style"&&typeof attrs[k]==="object"){Object.assign(e.style,attrs[k]);}
    else if(k.indexOf("on")===0){e.addEventListener(k.slice(2),attrs[k]);}
    else e.setAttribute(k,attrs[k]);
  });
  (children||[]).forEach(function(c){
    if(typeof c==="string") e.appendChild(document.createTextNode(c));
    else if(c) e.appendChild(c);
  });
  return e;
}

/* ── Build DOM ────────────────────────────────────────────────── */
var isLeft = POS==="bottom-left";

// FAB
var fab = el("div",{style:{
  position:"fixed",bottom:"24px",[isLeft?"left":"right"]:"24px",
  width:"56px",height:"56px",borderRadius:"50%",background:ACCENT,
  display:"flex",alignItems:"center",justifyContent:"center",
  cursor:"pointer",boxShadow:"0 4px 16px "+T.shadow,zIndex:"99999",
  transition:"transform .2s",fontSize:"26px",color:"#fff",userSelect:"none"
},onclick:toggle},["💬"]);
document.body.appendChild(fab);

// Chat window
var win = el("div",{style:{
  position:"fixed",bottom:"92px",[isLeft?"left":"right"]:"24px",
  width:"380px",maxHeight:"520px",borderRadius:"16px",overflow:"hidden",
  background:T.card,border:"1px solid "+T.border,
  boxShadow:"0 8px 32px "+T.shadow,zIndex:"99998",
  display:"none",flexDirection:"column",fontFamily:"'Inter',system-ui,sans-serif",
  transition:"opacity .25s,transform .25s",opacity:"0",transform:"translateY(12px)"
}});

// Header
var hdr = el("div",{style:{
  padding:"14px 18px",background:ACCENT,display:"flex",alignItems:"center",gap:"10px"
}},[
  LOGO ? el("img",{src:LOGO,style:{width:"28px",height:"28px",borderRadius:"6px"}}) : null,
  el("span",{style:{flex:"1",fontWeight:"600",fontSize:"15px",color:"#fff"}},[AGENT]),
  el("span",{style:{cursor:"pointer",fontSize:"18px",color:"rgba(255,255,255,.7)",lineHeight:"1"},onclick:toggle},["✕"])
]);
win.appendChild(hdr);

// Messages area
var msgs = el("div",{style:{
  flex:"1",overflowY:"auto",padding:"14px",minHeight:"280px",maxHeight:"340px",
  display:"flex",flexDirection:"column",gap:"10px",background:T.bg
}});
win.appendChild(msgs);

// Input bar
var inputBar = el("div",{style:{
  display:"flex",gap:"8px",padding:"10px 14px",borderTop:"1px solid "+T.border,background:T.card
}});
var inp = el("input",{type:"text",placeholder:"Type a message...",style:{
  flex:"1",padding:"10px 14px",borderRadius:"8px",border:"1px solid "+T.border,
  background:T.inputBg,color:T.text,fontSize:"14px",outline:"none"
},onkeydown:function(e){if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}}});
var sendBtn = el("button",{style:{
  padding:"10px 16px",borderRadius:"8px",border:"none",background:ACCENT,color:"#fff",
  fontSize:"14px",cursor:"pointer",fontWeight:"500",whiteSpace:"nowrap"
},onclick:send},["Send"]);
inputBar.appendChild(inp);
inputBar.appendChild(sendBtn);
win.appendChild(inputBar);
document.body.appendChild(win);

// Greeting bubble
addBubble("bot",GREET);

/* ── State ────────────────────────────────────────────────────── */
var open=false, busy=false;

function toggle(){
  open=!open;
  if(open){
    win.style.display="flex";
    setTimeout(function(){win.style.opacity="1";win.style.transform="translateY(0)";},10);
    fab.style.transform="scale(.85)";
    inp.focus();
  } else {
    win.style.opacity="0";win.style.transform="translateY(12px)";
    setTimeout(function(){win.style.display="none";},250);
    fab.style.transform="scale(1)";
  }
}

function addBubble(role,text){
  var isUser = role==="user";
  var bub = el("div",{style:{
    padding:"10px 14px",borderRadius:"12px",maxWidth:"85%",fontSize:"14px",lineHeight:"1.5",
    wordBreak:"break-word",whiteSpace:"pre-wrap",
    background:isUser?T.userBg:T.botBg,color:T.text,
    alignSelf:isUser?"flex-end":"flex-start"
  }},[text]);
  msgs.appendChild(bub);
  msgs.scrollTop=msgs.scrollHeight;
  return bub;
}

function addTyping(){
  var dot = el("div",{class:"aos-typing",style:{
    padding:"10px 14px",borderRadius:"12px",background:T.botBg,alignSelf:"flex-start",
    display:"flex",gap:"5px",alignItems:"center"
  }},[
    el("span",{style:{width:"7px",height:"7px",borderRadius:"50%",background:T.muted,animation:"aosDot .6s ease-in-out infinite"}}),
    el("span",{style:{width:"7px",height:"7px",borderRadius:"50%",background:T.muted,animation:"aosDot .6s ease-in-out .15s infinite"}}),
    el("span",{style:{width:"7px",height:"7px",borderRadius:"50%",background:T.muted,animation:"aosDot .6s ease-in-out .3s infinite"}})
  ]);
  msgs.appendChild(dot);
  msgs.scrollTop=msgs.scrollHeight;
  return dot;
}

/* ── Inject keyframe animation ─────────────────────────────── */
var style=document.createElement("style");
style.textContent="@keyframes aosDot{0%,80%,100%{opacity:.3;transform:scale(.8)}40%{opacity:1;transform:scale(1.1)}}";
document.head.appendChild(style);

/* ── Send message ─────────────────────────────────────────── */
function send(){
  var q=inp.value.trim();
  if(!q||busy) return;
  inp.value="";
  addBubble("user",q);
  busy=true;
  sendBtn.textContent="...";
  var typing=addTyping();

  // Try WebSocket first, fall back to HTTP
  var wsProto = BASE.replace(/^http/,"ws");
  var ws;
  try{ ws = new WebSocket(wsProto+"/ws/chat"); } catch(e){ ws=null; }

  if(ws){
    var tokens=[];
    var botBub=null;
    ws.onopen=function(){
      ws.send(JSON.stringify({query:q,model:MODEL,system_prompt:PROMPT,tools:TOOLS}));
    };
    ws.onmessage=function(ev){
      var d=JSON.parse(ev.data);
      if(d.type==="token"){
        if(!botBub){if(typing.parentNode)typing.parentNode.removeChild(typing);botBub=addBubble("bot","");}
        tokens.push(d.content);
        botBub.textContent=tokens.join("");
        msgs.scrollTop=msgs.scrollHeight;
      } else if(d.type==="done"){
        if(!botBub){if(typing.parentNode)typing.parentNode.removeChild(typing);botBub=addBubble("bot",d.response||"");}
        else botBub.textContent=d.response||tokens.join("");
        busy=false;sendBtn.textContent="Send";ws.close();
      } else if(d.type==="error"){
        if(typing.parentNode)typing.parentNode.removeChild(typing);
        addBubble("bot","Error: "+(d.message||"unknown"));
        busy=false;sendBtn.textContent="Send";ws.close();
      }
    };
    ws.onerror=function(){
      ws.close();
      httpFallback(q,typing);
    };
  } else {
    httpFallback(q,typing);
  }
}

function httpFallback(q,typing){
  var hdrs={"Content-Type":"application/json"};
  if(API_KEY) hdrs["X-API-Key"]=API_KEY;
  fetch(BASE+"/api/run",{method:"POST",headers:hdrs,body:JSON.stringify({
    query:q,model:MODEL,system_prompt:PROMPT,tools:TOOLS
  })}).then(function(r){return r.json();}).then(function(d){
    if(typing.parentNode)typing.parentNode.removeChild(typing);
    addBubble("bot",d.response||d.message||"No response");
    busy=false;sendBtn.textContent="Send";
  }).catch(function(e){
    if(typing.parentNode)typing.parentNode.removeChild(typing);
    addBubble("bot","Error: "+e.message);
    busy=false;sendBtn.textContent="Send";
  });
}

})();
"""


# ── Inline widget generator ─────────────────────────────────────────────────

def generate_widget(
    agent_name: str = "AgentOS",
    base_url: str = "http://localhost:8000",
    api_key: str = "",
    theme: str = "dark",
    position: str = "bottom-right",
    accent_color: str = "#6c5ce7",
    logo: str = "",
    greeting: str = "Hi! How can I help you today?",
    model: str = "gpt-4o-mini",
    system_prompt: str = "You are a helpful assistant.",
    tools: list[str] | None = None,
) -> str:
    """Return a self-contained ``<script>`` block that renders the chat widget.

    Embed this in any HTML page — no extra files needed.
    """
    config = {
        "baseUrl": base_url,
        "apiKey": api_key,
        "agentName": agent_name,
        "theme": theme,
        "position": position,
        "accentColor": accent_color,
        "logo": logo,
        "greeting": greeting,
        "model": model,
        "systemPrompt": system_prompt,
        "tools": tools or [],
    }
    js_body = generate_widget_js()
    return (
        "<script>\n"
        f"window.AgentOSConfig = {json.dumps(config)};\n"
        f"{js_body}\n"
        "</script>"
    )


# ── Snippet helpers for the embed preview / docs ────────────────────────────

def generate_snippet(
    base_url: str = "http://localhost:8000",
    api_key: str = "",
    agent_name: str = "AgentOS",
    theme: str = "dark",
    position: str = "bottom-right",
    accent_color: str = "#6c5ce7",
) -> str:
    """Return a short HTML snippet that loads the widget via the hosted JS file."""
    config_lines = [
        f'    baseUrl: "{base_url}",',
        f'    apiKey: "{api_key}",' if api_key else "",
        f'    agentName: "{agent_name}",',
        f'    theme: "{theme}",',
        f'    position: "{position}",',
        f'    accentColor: "{accent_color}",',
    ]
    config_block = "\n".join(l for l in config_lines if l)
    return (
        "<script>\n"
        "  window.AgentOSConfig = {\n"
        f"{config_block}\n"
        "  };\n"
        "</script>\n"
        f'<script src="{base_url}/embed/chat.js"></script>'
    )
