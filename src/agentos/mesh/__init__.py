"""AgentOS Mesh — Agent-to-Agent Protocol.

Enables agents from different organisations to:
- **Discover** each other via a registry (DNS-like lookup)
- **Authenticate** with HMAC-signed messages
- **Communicate** using a structured JSON protocol
- **Negotiate** terms (price, SLA, scope) with counter-offers
- **Transact** with verified outcomes and tamper-evident receipts
"""

from agentos.mesh.protocol import (
    MeshIdentity,
    MeshMessage,
    MessageType,
    NegotiationProposal,
    NegotiationResponse,
    NegotiationStatus,
    make_error,
    make_handshake,
    make_handshake_ack,
    make_negotiate,
    make_negotiate_response,
    make_ping,
    make_pong,
)
from agentos.mesh.auth import (
    derive_shared_secret,
    generate_challenge,
    generate_keypair,
    sign_message,
    solve_challenge,
    verify_challenge,
    verify_signature,
)
from agentos.mesh.discovery import MeshRegistry, get_registry
from agentos.mesh.transaction import (
    TransactionLedger,
    TransactionReceipt,
    TransactionRequest,
    TransactionStatus,
    get_ledger,
    make_transact,
    make_transact_result,
    make_verify,
    make_verify_result,
)
from agentos.mesh.server import MeshNode, mesh_app, init_node, get_node, handle_message, run_mesh_server
from agentos.mesh.mesh_router import MeshRouter, MeshMessage as MeshRouterMessage, get_mesh_router

__all__ = [
    # Protocol
    "MeshIdentity",
    "MeshMessage",
    "MessageType",
    "NegotiationProposal",
    "NegotiationResponse",
    "NegotiationStatus",
    "make_ping",
    "make_pong",
    "make_handshake",
    "make_handshake_ack",
    "make_negotiate",
    "make_negotiate_response",
    "make_error",
    # Auth
    "generate_keypair",
    "sign_message",
    "verify_signature",
    "generate_challenge",
    "solve_challenge",
    "verify_challenge",
    "derive_shared_secret",
    # Discovery
    "MeshRegistry",
    "get_registry",
    # Transactions
    "TransactionLedger",
    "TransactionReceipt",
    "TransactionRequest",
    "TransactionStatus",
    "get_ledger",
    "make_transact",
    "make_transact_result",
    "make_verify",
    "make_verify_result",
    # Server
    "MeshNode",
    "mesh_app",
    "init_node",
    "get_node",
    "handle_message",
    "run_mesh_server",
    "MeshRouter",
    "MeshRouterMessage",
    "get_mesh_router",
]
