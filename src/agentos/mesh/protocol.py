"""Mesh Protocol — message format, handshake, and negotiation primitives.

Every inter-agent message is a :class:`MeshMessage` with a cryptographic
signature so the recipient can verify authenticity.  The protocol defines
a small set of message *types* that cover the full lifecycle:

    ping → pong → handshake → negotiate → transact → verify → ack

All payloads are plain JSON dicts — no binary, no protobuf.
"""

from __future__ import annotations

import threading
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Message types ────────────────────────────────────────────────────────────

class MessageType(str, Enum):
    PING = "ping"
    PONG = "pong"
    HANDSHAKE = "handshake"
    HANDSHAKE_ACK = "handshake_ack"
    NEGOTIATE = "negotiate"
    NEGOTIATE_RESPONSE = "negotiate_response"
    TRANSACT = "transact"
    TRANSACT_RESULT = "transact_result"
    VERIFY = "verify"
    VERIFY_RESULT = "verify_result"
    ACK = "ack"
    ERROR = "error"


class NegotiationStatus(str, Enum):
    PROPOSED = "proposed"
    COUNTER = "counter"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


# ── Identity ─────────────────────────────────────────────────────────────────

class MeshIdentity(BaseModel):
    """Public identity card of a mesh agent."""

    mesh_id: str                       # e.g. "sales-bot@acme.com"
    display_name: str = ""
    public_key: str = ""               # base64-encoded HMAC key (shared-secret simplified model)
    capabilities: list[str] = Field(default_factory=list)  # ["negotiate", "quote", "transact"]
    endpoint_url: str = ""             # "http://acme.com:9000/mesh"
    organisation: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Core message envelope ────────────────────────────────────────────────────

class MeshMessage(BaseModel):
    """Signed JSON envelope exchanged between mesh agents."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    type: MessageType
    sender: str                        # mesh_id
    recipient: str                     # mesh_id
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    signature: str = ""                # HMAC-SHA256 hex digest
    reply_to: str | None = None        # id of the message being replied to
    conversation_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])

    def summary(self) -> str:
        return f"[{self.type.value}] {self.sender} → {self.recipient} ({self.id})"


# ── Negotiation helpers ──────────────────────────────────────────────────────

class NegotiationProposal(BaseModel):
    """Structured proposal payload for a negotiate message."""

    proposal_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    description: str = ""
    terms: dict[str, Any] = Field(default_factory=dict)  # price, quantity, SLA, etc.
    status: NegotiationStatus = NegotiationStatus.PROPOSED
    round: int = 1
    max_rounds: int = 5
    deadline: float | None = None      # unix timestamp


class NegotiationResponse(BaseModel):
    """Response to a negotiation proposal."""

    proposal_id: str
    status: NegotiationStatus
    counter_terms: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    round: int = 1


# ── Helpers for building messages ────────────────────────────────────────────

def make_ping(sender: str, recipient: str) -> MeshMessage:
    return MeshMessage(type=MessageType.PING, sender=sender, recipient=recipient)


def make_pong(sender: str, ping: MeshMessage) -> MeshMessage:
    return MeshMessage(
        type=MessageType.PONG, sender=sender, recipient=ping.sender,
        reply_to=ping.id, conversation_id=ping.conversation_id,
    )


def make_handshake(identity: MeshIdentity, recipient: str) -> MeshMessage:
    return MeshMessage(
        type=MessageType.HANDSHAKE,
        sender=identity.mesh_id,
        recipient=recipient,
        payload=identity.model_dump(),
    )


def make_handshake_ack(identity: MeshIdentity, handshake: MeshMessage) -> MeshMessage:
    return MeshMessage(
        type=MessageType.HANDSHAKE_ACK,
        sender=identity.mesh_id,
        recipient=handshake.sender,
        payload=identity.model_dump(),
        reply_to=handshake.id,
        conversation_id=handshake.conversation_id,
    )


def make_negotiate(
    sender: str,
    recipient: str,
    proposal: NegotiationProposal,
    conversation_id: str = "",
) -> MeshMessage:
    return MeshMessage(
        type=MessageType.NEGOTIATE,
        sender=sender,
        recipient=recipient,
        payload=proposal.model_dump(),
        conversation_id=conversation_id or uuid.uuid4().hex[:12],
    )


def make_negotiate_response(
    sender: str,
    negotiate_msg: MeshMessage,
    response: NegotiationResponse,
) -> MeshMessage:
    return MeshMessage(
        type=MessageType.NEGOTIATE_RESPONSE,
        sender=sender,
        recipient=negotiate_msg.sender,
        payload=response.model_dump(),
        reply_to=negotiate_msg.id,
        conversation_id=negotiate_msg.conversation_id,
    )


def make_error(sender: str, recipient: str, error: str, reply_to: str = "") -> MeshMessage:
    return MeshMessage(
        type=MessageType.ERROR,
        sender=sender,
        recipient=recipient,
        payload={"error": error},
        reply_to=reply_to or None,
    )


# ── In-process delegation protocol ───────────────────────────────────────────


class AgentMessage(BaseModel):
    """A message passed between agents in a mesh."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    sender: str
    receiver: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_run_id: str | None = None
    timestamp: float = Field(default_factory=time.time)


class DelegationRequest(BaseModel):
    """A request from one agent to delegate a subtask to another."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    from_agent: str
    to_agent: str
    task: str
    context: dict[str, Any] = Field(default_factory=dict)
    parent_run_id: str | None = None
    timestamp: float = Field(default_factory=time.time)


class DelegationResult(BaseModel):
    """The result of a delegated subtask."""

    request_id: str
    from_agent: str
    to_agent: str
    result: str
    cost_usd: float = 0.0
    tokens_used: int = 0
    latency_ms: float = 0.0
    success: bool = True


class SharedContext:
    """Thread-safe shared key-value store for agents in a mesh.

    All agents in a mesh share a single ``SharedContext`` instance.
    They can read/write facts that are visible to every other agent,
    enabling implicit coordination without direct message passing.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._history: list[dict[str, Any]] = []

    def set(self, key: str, value: Any, *, author: str = "") -> None:
        with self._lock:
            self._store[key] = value
            self._history.append(
                {"key": key, "value": value, "author": author, "ts": time.time()}
            )

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._store.get(key, default)

    def get_all(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._store)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def to_prompt_fragment(self) -> str:
        """Render shared context as a text block for system-prompt injection."""
        with self._lock:
            if not self._store:
                return ""
            lines = ["[SHARED CONTEXT from other agents]"]
            for k, v in self._store.items():
                lines.append(f"- {k}: {v}")
            return "\n".join(lines)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        return f"SharedContext({len(self)} keys)"
