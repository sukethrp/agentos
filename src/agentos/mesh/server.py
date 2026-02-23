"""Mesh Server — FastAPI endpoints that expose an agent as a mesh node.

Can be run standalone (``python -m agentos.mesh.server``) or mounted
as a sub-application inside the main AgentOS web server.

Endpoints:

    POST /mesh/message         — receive any mesh message
    GET  /mesh/identity        — return this node's public identity
    GET  /mesh/registry        — list all registered agents
    GET  /mesh/registry/search — search the registry
    POST /mesh/register        — register an external agent
    POST /mesh/deregister      — remove an agent from registry
    GET  /mesh/transactions    — list transactions
    GET  /mesh/verify/{tx_id}  — verify a transaction receipt
    GET  /mesh/stats           — registry + ledger stats
"""

from __future__ import annotations


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agentos.mesh.auth import (
    derive_shared_secret,
    generate_keypair,
    sign_message,
)
from agentos.mesh.discovery import MeshRegistry, get_registry
from agentos.mesh.protocol import (
    MeshIdentity,
    MeshMessage,
    MessageType,
    NegotiationResponse,
    NegotiationStatus,
    make_error,
    make_handshake_ack,
    make_negotiate_response,
    make_pong,
)
from agentos.mesh.transaction import (
    TransactionLedger,
    TransactionReceipt,
    TransactionRequest,
    TransactionStatus,
    get_ledger,
    make_transact_result,
    make_verify_result,
)


# ── Node state ───────────────────────────────────────────────────────────────


class MeshNode:
    """Represents **this** server's identity and state."""

    def __init__(
        self,
        mesh_id: str = "node@localhost",
        display_name: str = "AgentOS Node",
        organisation: str = "local",
        capabilities: list[str] | None = None,
        endpoint_url: str = "http://localhost:9100/mesh",
    ) -> None:
        self.private_key, self.public_key = generate_keypair()
        self.identity = MeshIdentity(
            mesh_id=mesh_id,
            display_name=display_name,
            public_key=self.public_key,
            capabilities=capabilities or ["ping", "negotiate", "transact"],
            endpoint_url=endpoint_url,
            organisation=organisation,
        )
        self.known_keys: dict[str, str] = {}  # mesh_id → shared_secret
        self.conversations: dict[str, list[MeshMessage]] = {}

    def sign(self, msg: MeshMessage) -> MeshMessage:
        return sign_message(msg, self.private_key)

    def remember(self, msg: MeshMessage) -> None:
        self.conversations.setdefault(msg.conversation_id, []).append(msg)


# ── Default node singleton ───────────────────────────────────────────────────

_node: MeshNode | None = None


def get_node() -> MeshNode:
    global _node
    if _node is None:
        _node = MeshNode()
    return _node


def init_node(
    mesh_id: str,
    display_name: str = "AgentOS Node",
    organisation: str = "local",
    capabilities: list[str] | None = None,
    endpoint_url: str = "http://localhost:9100/mesh",
) -> MeshNode:
    global _node
    _node = MeshNode(
        mesh_id=mesh_id,
        display_name=display_name,
        organisation=organisation,
        capabilities=capabilities,
        endpoint_url=endpoint_url,
    )
    registry = get_registry()
    registry.register(_node.identity)
    return _node


# ── Negotiation handler (auto-accept demo logic) ────────────────────────────


def _auto_negotiate(node: MeshNode, msg: MeshMessage) -> MeshMessage:
    """Simple auto-negotiation strategy: accept if price ≤ budget."""
    proposal = msg.payload
    terms = proposal.get("terms", {})
    price = terms.get("price", 0)

    budget = 10000  # default budget for demo

    if price <= budget:
        resp = NegotiationResponse(
            proposal_id=proposal.get("proposal_id", ""),
            status=NegotiationStatus.ACCEPTED,
            reason=f"Price ${price:,} is within our budget of ${budget:,}.",
            round=proposal.get("round", 1),
        )
    elif proposal.get("round", 1) < proposal.get("max_rounds", 5):
        counter_price = int(price * 0.8)
        resp = NegotiationResponse(
            proposal_id=proposal.get("proposal_id", ""),
            status=NegotiationStatus.COUNTER,
            counter_terms={**terms, "price": counter_price},
            reason=f"${price:,} is over budget. We can do ${counter_price:,}.",
            round=proposal.get("round", 1) + 1,
        )
    else:
        resp = NegotiationResponse(
            proposal_id=proposal.get("proposal_id", ""),
            status=NegotiationStatus.REJECTED,
            reason=f"Cannot agree on price after {proposal.get('round', 1)} rounds.",
            round=proposal.get("round", 1),
        )

    return make_negotiate_response(node.identity.mesh_id, msg, resp)


# ── Message router ───────────────────────────────────────────────────────────


def handle_message(
    msg: MeshMessage,
    node: MeshNode | None = None,
    registry: MeshRegistry | None = None,
    ledger: TransactionLedger | None = None,
) -> MeshMessage:
    """Route an incoming message and produce a response."""
    node = node or get_node()
    registry = registry or get_registry()
    ledger = ledger or get_ledger()

    node.remember(msg)

    if msg.type == MessageType.PING:
        resp = make_pong(node.identity.mesh_id, msg)

    elif msg.type == MessageType.HANDSHAKE:
        sender_identity = MeshIdentity(**msg.payload)
        registry.register(sender_identity)
        if sender_identity.public_key:
            shared = derive_shared_secret(node.public_key, sender_identity.public_key)
            node.known_keys[sender_identity.mesh_id] = shared
        resp = make_handshake_ack(node.identity, msg)

    elif msg.type == MessageType.NEGOTIATE:
        resp = _auto_negotiate(node, msg)

    elif msg.type == MessageType.TRANSACT:
        req = TransactionRequest(**msg.payload)
        ledger.record_request(req)
        receipt = TransactionReceipt(
            transaction_id=req.transaction_id,
            status=TransactionStatus.COMPLETED,
            outcome={"result": "fulfilled", "terms": req.agreed_terms},
            initiator=req.initiator,
            counterparty=req.counterparty,
        )
        receipt.finalise()
        ledger.record_receipt(receipt)
        resp = make_transact_result(node.identity.mesh_id, msg, receipt)

    elif msg.type == MessageType.VERIFY:
        tx_id = msg.payload.get("transaction_id", "")
        receipt = ledger.get_receipt(tx_id)
        resp = make_verify_result(node.identity.mesh_id, msg, receipt)

    else:
        resp = make_error(
            node.identity.mesh_id,
            msg.sender,
            f"Unsupported message type: {msg.type}",
            reply_to=msg.id,
        )

    node.sign(resp)
    node.remember(resp)
    return resp


# ── FastAPI app ──────────────────────────────────────────────────────────────

mesh_app = FastAPI(title="AgentOS Mesh", version="0.3.1")


@mesh_app.post("/mesh/message")
async def receive_message(msg: MeshMessage) -> dict:
    """Receive any mesh protocol message and return the response."""
    resp = handle_message(msg)
    return resp.model_dump()


@mesh_app.get("/mesh/identity")
async def get_identity() -> dict:
    return get_node().identity.model_dump()


class RegisterBody(BaseModel):
    mesh_id: str
    display_name: str = ""
    public_key: str = ""
    capabilities: list[str] = []
    endpoint_url: str = ""
    organisation: str = ""


@mesh_app.post("/mesh/register")
async def register_agent(body: RegisterBody) -> dict:
    identity = MeshIdentity(**body.model_dump())
    get_registry().register(identity)
    return {"status": "registered", "mesh_id": identity.mesh_id}


@mesh_app.post("/mesh/deregister")
async def deregister_agent(body: dict) -> dict:
    mesh_id = body.get("mesh_id", "")
    ok = get_registry().deregister(mesh_id)
    if not ok:
        raise HTTPException(404, f"Agent {mesh_id} not found")
    return {"status": "deregistered", "mesh_id": mesh_id}


@mesh_app.get("/mesh/registry")
async def list_registry() -> list[dict]:
    return get_registry().to_list()


@mesh_app.get("/mesh/registry/search")
async def search_registry(
    q: str = "",
    capability: str = "",
    organisation: str = "",
) -> list[dict]:
    results = get_registry().search(
        query=q, capability=capability, organisation=organisation
    )
    return [a.model_dump() for a in results]


@mesh_app.get("/mesh/transactions")
async def list_transactions() -> list[dict]:
    return get_ledger().list_transactions()


@mesh_app.get("/mesh/verify/{tx_id}")
async def verify_transaction(tx_id: str) -> dict:
    return get_ledger().verify(tx_id)


@mesh_app.get("/mesh/stats")
async def mesh_stats() -> dict:
    return {
        "registry": get_registry().stats(),
        "ledger": get_ledger().stats(),
        "node": get_node().identity.model_dump(),
    }


# ── Standalone runner ────────────────────────────────────────────────────────


def run_mesh_server(
    mesh_id: str = "node@localhost",
    display_name: str = "AgentOS Node",
    organisation: str = "local",
    host: str = "0.0.0.0",
    port: int = 9100,
) -> None:
    """Start the mesh server (blocking)."""
    import uvicorn

    init_node(
        mesh_id=mesh_id,
        display_name=display_name,
        organisation=organisation,
        endpoint_url=f"http://{host}:{port}/mesh",
    )
    uvicorn.run(mesh_app, host=host, port=port)


if __name__ == "__main__":
    run_mesh_server()
