"""Mesh Transactions — verified outcomes, receipts, and ledger.

A *transaction* represents a completed agreement between two agents.
Each side signs a receipt so there is non-repudiable proof of what
was agreed and what the outcome was.

The :class:`TransactionLedger` keeps an in-memory audit trail.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentos.mesh.auth import sign_message
from agentos.mesh.protocol import (
    MeshMessage,
    MessageType,
)


# ── Transaction models ───────────────────────────────────────────────────────

class TransactionStatus(str, Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    DISPUTED = "disputed"


class TransactionRequest(BaseModel):
    """Payload for a TRANSACT message."""

    transaction_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    agreed_terms: dict[str, Any] = Field(default_factory=dict)
    negotiation_id: str = ""         # links back to the negotiation conversation
    initiator: str = ""              # mesh_id
    counterparty: str = ""           # mesh_id
    created_at: float = Field(default_factory=time.time)


class TransactionReceipt(BaseModel):
    """Immutable receipt signed by both parties."""

    transaction_id: str
    status: TransactionStatus = TransactionStatus.COMPLETED
    outcome: dict[str, Any] = Field(default_factory=dict)
    initiator: str = ""
    counterparty: str = ""
    initiator_signature: str = ""
    counterparty_signature: str = ""
    completed_at: float = Field(default_factory=time.time)
    receipt_hash: str = ""           # SHA-256 of the canonical receipt

    def compute_hash(self) -> str:
        """Compute a tamper-evident hash of the receipt contents."""
        obj = {
            "transaction_id": self.transaction_id,
            "status": self.status.value,
            "outcome": self.outcome,
            "initiator": self.initiator,
            "counterparty": self.counterparty,
            "completed_at": self.completed_at,
        }
        canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def finalise(self) -> None:
        """Stamp the receipt hash (call after both signatures are set)."""
        self.receipt_hash = self.compute_hash()

    def verify_integrity(self) -> bool:
        """Check that the receipt hash still matches."""
        return self.receipt_hash == self.compute_hash()


# ── Message builders ─────────────────────────────────────────────────────────

def make_transact(
    sender: str,
    recipient: str,
    request: TransactionRequest,
    conversation_id: str = "",
) -> MeshMessage:
    return MeshMessage(
        type=MessageType.TRANSACT,
        sender=sender,
        recipient=recipient,
        payload=request.model_dump(),
        conversation_id=conversation_id or uuid.uuid4().hex[:12],
    )


def make_transact_result(
    sender: str,
    transact_msg: MeshMessage,
    receipt: TransactionReceipt,
) -> MeshMessage:
    return MeshMessage(
        type=MessageType.TRANSACT_RESULT,
        sender=sender,
        recipient=transact_msg.sender,
        payload=receipt.model_dump(),
        reply_to=transact_msg.id,
        conversation_id=transact_msg.conversation_id,
    )


def make_verify(
    sender: str,
    recipient: str,
    transaction_id: str,
    conversation_id: str = "",
) -> MeshMessage:
    return MeshMessage(
        type=MessageType.VERIFY,
        sender=sender,
        recipient=recipient,
        payload={"transaction_id": transaction_id},
        conversation_id=conversation_id,
    )


def make_verify_result(
    sender: str,
    verify_msg: MeshMessage,
    receipt: TransactionReceipt | None,
) -> MeshMessage:
    if receipt:
        payload = {"found": True, "receipt": receipt.model_dump(), "integrity": receipt.verify_integrity()}
    else:
        payload = {"found": False}
    return MeshMessage(
        type=MessageType.VERIFY_RESULT,
        sender=sender,
        recipient=verify_msg.sender,
        payload=payload,
        reply_to=verify_msg.id,
        conversation_id=verify_msg.conversation_id,
    )


# ── Ledger ───────────────────────────────────────────────────────────────────

class TransactionLedger:
    """In-memory audit trail of all transactions."""

    def __init__(self) -> None:
        self._transactions: dict[str, TransactionRequest] = {}
        self._receipts: dict[str, TransactionReceipt] = {}

    def record_request(self, req: TransactionRequest) -> None:
        self._transactions[req.transaction_id] = req

    def record_receipt(self, receipt: TransactionReceipt) -> None:
        receipt.finalise()
        self._receipts[receipt.transaction_id] = receipt

    def get_request(self, transaction_id: str) -> TransactionRequest | None:
        return self._transactions.get(transaction_id)

    def get_receipt(self, transaction_id: str) -> TransactionReceipt | None:
        return self._receipts.get(transaction_id)

    def list_transactions(self) -> list[dict]:
        out = []
        for tid, req in self._transactions.items():
            r = self._receipts.get(tid)
            out.append({
                "transaction_id": tid,
                "description": req.description,
                "initiator": req.initiator,
                "counterparty": req.counterparty,
                "status": r.status.value if r else "pending",
                "has_receipt": r is not None,
            })
        return out

    def verify(self, transaction_id: str) -> dict:
        """Verify a transaction's receipt integrity."""
        receipt = self._receipts.get(transaction_id)
        if not receipt:
            return {"found": False, "transaction_id": transaction_id}
        return {
            "found": True,
            "transaction_id": transaction_id,
            "status": receipt.status.value,
            "integrity": receipt.verify_integrity(),
            "receipt_hash": receipt.receipt_hash,
        }

    def stats(self) -> dict:
        return {
            "total_transactions": len(self._transactions),
            "total_receipts": len(self._receipts),
            "completed": sum(1 for r in self._receipts.values() if r.status == TransactionStatus.COMPLETED),
            "failed": sum(1 for r in self._receipts.values() if r.status == TransactionStatus.FAILED),
        }


# ── Default singleton ────────────────────────────────────────────────────────

_default_ledger: TransactionLedger | None = None


def get_ledger() -> TransactionLedger:
    global _default_ledger
    if _default_ledger is None:
        _default_ledger = TransactionLedger()
    return _default_ledger
