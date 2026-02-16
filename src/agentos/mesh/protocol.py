"""Mesh Protocol — message format, handshake, and negotiation primitives.

Every inter-agent message is a :class:`MeshMessage` with a cryptographic
signature so the recipient can verify authenticity.  The protocol defines
a small set of message *types* that cover the full lifecycle:

    ping → pong → handshake → negotiate → transact → verify → ack

All payloads are plain JSON dicts — no binary, no protobuf.
"""

from __future__ import annotations

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
