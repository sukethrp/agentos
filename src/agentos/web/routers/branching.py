from __future__ import annotations
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agentos.core.branching import get_or_create_tree, get_tree, list_trees
from agentos.monitor.store import store

router = APIRouter(tags=["branching"])

class BranchRequest(BaseModel):
    tree_id: str
    at_message_index: int | None = None
    label: str = ""
    source_branch_id: str | None = None


class BranchMessageRequest(BaseModel):
    tree_id: str
    branch_id: str | None = None
    role: str = "user"
    content: str = ""


class CompareBranchesRequest(BaseModel):
    tree_id: str
    branch_a: str
    branch_b: str


class MergeBranchesRequest(BaseModel):
    tree_id: str
    branch_a: str
    branch_b: str
    label: str = "merged"

@router.post("/api/branch")
def create_branch(req: BranchRequest):
    """Create a new conversation branch."""
    tree = get_tree(req.tree_id)
    if not tree:
        return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
    try:
        new_id = tree.branch(
            at_message_index=req.at_message_index,
            label=req.label,
            source_branch_id=req.source_branch_id,
        )
        return {"status": "created", "branch_id": new_id, "tree": tree.to_dict()}
    except (KeyError, IndexError) as e:
        return JSONResponse({"status": "error", "message": str(e)}, 400)


@router.get("/api/branches")
def list_branches(tree_id: str | None = None):
    """List branches. If tree_id provided, show that tree; else list all trees."""
    if tree_id:
        tree = get_tree(tree_id)
        if not tree:
            return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
        return {"status": "ok", "tree": tree.to_dict()}
    return {"status": "ok", "trees": list_trees()}


@router.post("/api/branch/switch")
def switch_branch(tree_id: str, branch_id: str):
    """Switch the active branch for a tree."""
    tree = get_tree(tree_id)
    if not tree:
        return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
    try:
        tree.switch_branch(branch_id)
        return {"status": "ok", "active_branch_id": branch_id}
    except KeyError as e:
        return JSONResponse({"status": "error", "message": str(e)}, 404)


@router.post("/api/branch/message")
def add_branch_message(req: BranchMessageRequest):
    """Add a message to a branch (or the active branch)."""
    tree = get_tree(req.tree_id)
    if not tree:
        return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
    if req.branch_id:
        try:
            tree.switch_branch(req.branch_id)
        except KeyError as e:
            return JSONResponse({"status": "error", "message": str(e)}, 404)
    msg = tree.add_message(req.role, req.content)
    return {
        "status": "ok",
        "message": msg.to_dict(),
        "branch_id": tree.active_branch_id,
    }


@router.post("/api/branch/chat")
def chat_on_branch(req: BranchMessageRequest):
    """Send a user message on a branch and get an agent response.

    This adds the user message, runs the agent with the branch's full
    history, and adds the assistant response.
    """
    tree = get_tree(req.tree_id)
    if not tree:
        return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
    if req.branch_id:
        try:
            tree.switch_branch(req.branch_id)
        except KeyError as e:
            return JSONResponse({"status": "error", "message": str(e)}, 404)

    tree.add_message("user", req.content)

    from agentos.core.agent import Agent

    agent = Agent(
        name="branch-agent",
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant.",
    )
    agent.messages = tree.get_messages_openai()
    agent.events = []

    import io
    import sys

    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        msg, event = agent._run_sync_step()
    except AttributeError:
        # Fallback: run full agent
        from agentos.providers.router import call_model as call_llm_direct

        msg_result, event = call_llm_direct(
            model="gpt-4o-mini",
            messages=tree.get_messages_openai(),
            tools=[],
            agent_name="branch-agent",
        )
        response_text = msg_result.content or ""
        tree.add_message("assistant", response_text)
        for e in [event]:
            store.log_event(e)
        sys.stdout = old
        return {
            "status": "ok",
            "response": response_text,
            "cost": round(event.cost_usd, 6),
            "tokens": event.tokens_used,
            "branch_id": tree.active_branch_id,
            "messages": [m.to_dict() for m in tree.get_messages()],
        }
    finally:
        sys.stdout = old


@router.post("/api/branch/compare")
def compare_branches(req: CompareBranchesRequest):
    """Compare two branches side by side."""
    tree = get_tree(req.tree_id)
    if not tree:
        return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
    try:
        result = tree.compare_branches(req.branch_a, req.branch_b)
        return {"status": "ok", "comparison": result}
    except KeyError as e:
        return JSONResponse({"status": "error", "message": str(e)}, 404)


@router.post("/api/branch/merge")
def merge_branches(req: MergeBranchesRequest):
    """Merge two branches into a new combined branch."""
    tree = get_tree(req.tree_id)
    if not tree:
        return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
    try:
        merged_id = tree.merge_branches(req.branch_a, req.branch_b, label=req.label)
        return {"status": "ok", "merged_branch_id": merged_id, "tree": tree.to_dict()}
    except KeyError as e:
        return JSONResponse({"status": "error", "message": str(e)}, 404)


@router.delete("/api/branch/{tree_id}/{branch_id}")
def delete_branch(tree_id: str, branch_id: str):
    """Delete a branch (cannot delete the main branch)."""
    tree = get_tree(tree_id)
    if not tree:
        return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
    if tree.delete_branch(branch_id):
        return {"status": "deleted", "branch_id": branch_id}
    return JSONResponse(
        {"status": "error", "message": "Cannot delete main branch or branch not found"},
        400,
    )


@router.get("/api/branch/messages")
def get_branch_messages(tree_id: str, branch_id: str | None = None):
    """Get all messages for a specific branch (or the active branch)."""
    tree = get_tree(tree_id)
    if not tree:
        return JSONResponse({"status": "error", "message": "Tree not found"}, 404)
    try:
        bid = branch_id or tree.active_branch_id
        msgs = tree.get_messages(bid)
        return {
            "status": "ok",
            "branch_id": bid,
            "messages": [m.to_dict() for m in msgs],
        }
    except KeyError as e:
        return JSONResponse({"status": "error", "message": str(e)}, 404)


@router.post("/api/branch/new-tree")
def create_new_tree():
    """Create a new conversation tree."""
    tree = get_or_create_tree()
    return {"status": "created", "tree": tree.to_dict()}

