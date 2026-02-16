"""Mesh Auth — keypair generation, message signing, and verification.

Uses HMAC-SHA256 with a shared secret per agent identity.  This is a
*simplified* model suitable for demos and internal meshes.  For real
cross-organisation federation you would swap in RSA/Ed25519 asymmetric
signatures — the API surface stays the same.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from typing import Any

from agentos.mesh.protocol import MeshMessage


# ── Key generation ───────────────────────────────────────────────────────────

def generate_keypair() -> tuple[str, str]:
    """Return ``(private_key, public_key)`` hex strings.

    In the HMAC model, the *private key* is the full secret and the
    *public key* is a truncated fingerprint that acts as an identifier
    but cannot be used to forge signatures.
    """
    private = secrets.token_hex(32)
    public = hashlib.sha256(private.encode()).hexdigest()[:32]
    return private, public


# ── Signing ──────────────────────────────────────────────────────────────────

def _canonical(msg: MeshMessage) -> bytes:
    """Deterministic byte representation of the signable fields."""
    obj = {
        "id": msg.id,
        "type": msg.type.value if hasattr(msg.type, "value") else str(msg.type),
        "sender": msg.sender,
        "recipient": msg.recipient,
        "payload": msg.payload,
        "timestamp": msg.timestamp,
        "reply_to": msg.reply_to or "",
        "conversation_id": msg.conversation_id,
    }
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sign_message(msg: MeshMessage, private_key: str) -> MeshMessage:
    """Sign a message **in-place** and return it (for chaining)."""
    canonical = _canonical(msg)
    sig = hmac.new(private_key.encode(), canonical, hashlib.sha256).hexdigest()
    msg.signature = sig
    return msg


def verify_signature(msg: MeshMessage, private_key: str) -> bool:
    """Verify the HMAC signature on a message.

    The *private_key* of the **sender** is required because this is
    HMAC (symmetric).  In a production system, you'd verify against
    the sender's *public* RSA/Ed25519 key.
    """
    canonical = _canonical(msg)
    expected = hmac.new(private_key.encode(), canonical, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, msg.signature)


# ── Challenge-response helpers (handshake) ───────────────────────────────────

def generate_challenge() -> str:
    """Random nonce for a handshake challenge."""
    return secrets.token_hex(16)


def solve_challenge(challenge: str, private_key: str) -> str:
    """Prove knowledge of the private key by HMACing the challenge nonce."""
    return hmac.new(private_key.encode(), challenge.encode(), hashlib.sha256).hexdigest()


def verify_challenge(challenge: str, solution: str, private_key: str) -> bool:
    expected = solve_challenge(challenge, private_key)
    return hmac.compare_digest(expected, solution)


# ── Shared-secret derivation (for two-party HMAC) ───────────────────────────

def derive_shared_secret(key_a: str, key_b: str) -> str:
    """Derive a symmetric shared secret from two public keys.

    Both parties call this with the same two public keys (in any
    order).  The inputs are sorted so the result is commutative:
    ``derive(pub_A, pub_B) == derive(pub_B, pub_A)``.

    This is a simplified stand-in for Diffie-Hellman key agreement.
    """
    combined = "".join(sorted([key_a, key_b]))
    return hashlib.sha256(combined.encode()).hexdigest()
