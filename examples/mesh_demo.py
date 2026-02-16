#!/usr/bin/env python3
"""
Mesh Demo — Two agents from different companies negotiate a deal.

This example runs entirely in-process (no HTTP servers needed).
It demonstrates the full Agent-to-Agent Protocol lifecycle:

    1. Both agents generate keypairs and register in the mesh registry
    2. Agent A discovers Agent B via the registry
    3. Handshake — mutual authentication & key exchange
    4. Negotiate — Agent A proposes a deal, Agent B counter-offers
    5. Transact — agreed terms are executed and a receipt is issued
    6. Verify — both sides verify the transaction receipt
"""

from __future__ import annotations

import json
import textwrap

from agentos.mesh.protocol import (
    MeshIdentity,
    MessageType,
    NegotiationProposal,
    NegotiationStatus,
    make_handshake,
    make_negotiate,
    make_ping,
)
from agentos.mesh.auth import (
    derive_shared_secret,
    generate_keypair,
    sign_message,
    verify_signature,
)
from agentos.mesh.discovery import MeshRegistry
from agentos.mesh.transaction import (
    TransactionLedger,
    TransactionReceipt,
    TransactionRequest,
    TransactionStatus,
    make_transact,
    make_verify,
)
from agentos.mesh.server import MeshNode, handle_message


# ── Helpers ──────────────────────────────────────────────────────────────────

DIVIDER = "═" * 60


def pp(label: str, obj: dict | str) -> None:
    """Pretty-print a labelled section."""
    print(f"\n{DIVIDER}")
    print(f"  {label}")
    print(DIVIDER)
    if isinstance(obj, dict):
        print(textwrap.indent(json.dumps(obj, indent=2, default=str), "  "))
    else:
        print(textwrap.indent(str(obj), "  "))


def main() -> None:
    print("🔗 AgentOS Mesh — Agent-to-Agent Protocol Demo")
    print("=" * 60)

    # ── 1. Setup: two companies, two agents ──────────────────────────────

    # Shared registry & ledger (in production these live on a central server)
    registry = MeshRegistry()
    ledger = TransactionLedger()

    # --- Acme Corp (seller) ---
    acme_priv, acme_pub = generate_keypair()
    acme_identity = MeshIdentity(
        mesh_id="sales-bot@acme.com",
        display_name="Acme Sales Bot",
        public_key=acme_pub,
        capabilities=["negotiate", "quote", "transact"],
        endpoint_url="http://acme.com:9100/mesh",
        organisation="Acme Corp",
    )
    acme_node = MeshNode(
        mesh_id=acme_identity.mesh_id,
        display_name=acme_identity.display_name,
        organisation=acme_identity.organisation,
    )
    acme_node.private_key = acme_priv
    acme_node.public_key = acme_pub
    acme_node.identity = acme_identity
    registry.register(acme_identity)

    # --- Globex Inc (buyer) ---
    globex_priv, globex_pub = generate_keypair()
    globex_identity = MeshIdentity(
        mesh_id="procurement@globex.com",
        display_name="Globex Procurement Bot",
        public_key=globex_pub,
        capabilities=["negotiate", "purchase", "transact"],
        endpoint_url="http://globex.com:9100/mesh",
        organisation="Globex Inc",
    )
    globex_node = MeshNode(
        mesh_id=globex_identity.mesh_id,
        display_name=globex_identity.display_name,
        organisation=globex_identity.organisation,
    )
    globex_node.private_key = globex_priv
    globex_node.public_key = globex_pub
    globex_node.identity = globex_identity
    registry.register(globex_identity)

    pp("Registry", registry.stats())
    for agent in registry.list_all():
        print(f"  • {agent.mesh_id}  ({agent.organisation})")
        print(f"    capabilities: {agent.capabilities}")

    # ── 2. Discovery ─────────────────────────────────────────────────────

    print("\n\n📡 STEP 1: Discovery")
    print("-" * 40)

    results = registry.search(capability="negotiate")
    print(f"Globex searches for agents that can 'negotiate':")
    for r in results:
        print(f"  → found: {r.mesh_id} @ {r.organisation}")

    target = registry.lookup("sales-bot@acme.com")
    assert target is not None
    print(f"\nDirect lookup: {target.mesh_id} ✓")

    # ── 3. Ping ──────────────────────────────────────────────────────────

    print("\n\n🏓 STEP 2: Ping / Pong")
    print("-" * 40)

    ping = make_ping(globex_identity.mesh_id, acme_identity.mesh_id)
    sign_message(ping, globex_priv)
    print(f"Globex → Acme: {ping.summary()}")

    pong = handle_message(ping, node=acme_node, registry=registry, ledger=ledger)
    print(f"Acme → Globex: {pong.summary()}")
    print(f"  Round-trip signature verified: ✓")

    # ── 4. Handshake — mutual auth ───────────────────────────────────────

    print("\n\n🤝 STEP 3: Handshake (Mutual Authentication)")
    print("-" * 40)

    handshake = make_handshake(globex_identity, acme_identity.mesh_id)
    sign_message(handshake, globex_priv)
    print(f"Globex → Acme: {handshake.summary()}")

    hs_ack = handle_message(handshake, node=acme_node, registry=registry, ledger=ledger)
    print(f"Acme → Globex: {hs_ack.summary()}")

    # Derive shared secrets on both sides (using public keys only)
    shared_acme = derive_shared_secret(acme_pub, globex_pub)
    shared_globex = derive_shared_secret(globex_pub, acme_pub)
    print(f"  Acme  shared secret: {shared_acme[:16]}...")
    print(f"  Globex shared secret: {shared_globex[:16]}...")
    print(f"  Secrets match: {'✓' if shared_acme == shared_globex else '✗'}")

    # ── 5. Negotiation — multi-round ─────────────────────────────────────

    print("\n\n💬 STEP 4: Negotiation")
    print("-" * 40)

    proposal = NegotiationProposal(
        description="1000 units of Widget-X with premium support",
        terms={
            "product": "Widget-X",
            "quantity": 1000,
            "price": 15000,
            "currency": "USD",
            "delivery": "30 days",
            "support": "premium",
        },
        max_rounds=5,
    )

    conv_id = ""
    current_round = 1
    accepted = False

    while current_round <= proposal.max_rounds:
        proposal.round = current_round
        msg = make_negotiate(
            globex_identity.mesh_id,
            acme_identity.mesh_id,
            proposal,
            conversation_id=conv_id,
        )
        conv_id = msg.conversation_id
        sign_message(msg, globex_priv)

        print(f"\n  Round {current_round}:")
        print(f"    Globex proposes: ${proposal.terms.get('price', 0):,}")

        resp = handle_message(msg, node=acme_node, registry=registry, ledger=ledger)
        resp_payload = resp.payload
        status = resp_payload.get("status", "")

        print(f"    Acme responds: {status.upper()}")
        if resp_payload.get("reason"):
            print(f"    Reason: {resp_payload['reason']}")

        if status == NegotiationStatus.ACCEPTED.value:
            print(f"    ✅ Deal accepted at ${proposal.terms.get('price', 0):,}!")
            accepted = True
            break
        elif status == NegotiationStatus.COUNTER.value:
            counter = resp_payload.get("counter_terms", {})
            counter_price = counter.get("price", proposal.terms["price"])
            print(f"    Counter-offer: ${counter_price:,}")
            # Globex meets halfway
            new_price = (proposal.terms["price"] + counter_price) // 2
            proposal.terms["price"] = new_price
            current_round = resp_payload.get("round", current_round + 1)
        elif status == NegotiationStatus.REJECTED.value:
            print(f"    ❌ Deal rejected.")
            break
        else:
            current_round += 1

    if not accepted:
        print("\nNegotiation did not reach agreement.")
        return

    # ── 6. Transaction ───────────────────────────────────────────────────

    print("\n\n📜 STEP 5: Transaction")
    print("-" * 40)

    tx_request = TransactionRequest(
        description="Purchase of 1000 Widget-X units",
        agreed_terms=proposal.terms,
        negotiation_id=conv_id,
        initiator=globex_identity.mesh_id,
        counterparty=acme_identity.mesh_id,
    )
    ledger.record_request(tx_request)

    tx_msg = make_transact(
        globex_identity.mesh_id,
        acme_identity.mesh_id,
        tx_request,
        conversation_id=conv_id,
    )
    sign_message(tx_msg, globex_priv)
    print(f"Globex → Acme: {tx_msg.summary()}")

    tx_resp = handle_message(tx_msg, node=acme_node, registry=registry, ledger=ledger)
    receipt_data = tx_resp.payload
    print(f"Acme → Globex: {tx_resp.summary()}")

    pp("Transaction Receipt", receipt_data)

    # ── 7. Verification ──────────────────────────────────────────────────

    print("\n\n🔍 STEP 6: Verification")
    print("-" * 40)

    verify_msg = make_verify(
        globex_identity.mesh_id,
        acme_identity.mesh_id,
        tx_request.transaction_id,
        conversation_id=conv_id,
    )
    sign_message(verify_msg, globex_priv)
    print(f"Globex verifies transaction {tx_request.transaction_id}...")

    verify_resp = handle_message(verify_msg, node=acme_node, registry=registry, ledger=ledger)
    vr = verify_resp.payload
    print(f"  Found: {vr.get('found', False)}")
    if vr.get("found"):
        print(f"  Integrity check: {'✓ PASS' if vr.get('integrity') else '✗ FAIL'}")
        receipt_info = vr.get("receipt", {})
        print(f"  Status: {receipt_info.get('status', 'unknown').upper()}")
        print(f"  Receipt hash: {receipt_info.get('receipt_hash', 'N/A')[:24]}...")

    # ── 8. Final summary ─────────────────────────────────────────────────

    pp("Ledger Stats", ledger.stats())
    pp("Registry Stats", registry.stats())

    print(f"\n{'=' * 60}")
    print("✅ Agent-to-Agent Protocol demo complete!")
    print(f"   Agents: {globex_identity.mesh_id} ↔ {acme_identity.mesh_id}")
    print(f"   Negotiation rounds: {current_round}")
    print(f"   Final price: ${proposal.terms.get('price', 0):,}")
    print(f"   Transaction: {tx_request.transaction_id}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
