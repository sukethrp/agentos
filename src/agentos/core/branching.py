"""Conversation Branching — explore "what if" scenarios by forking conversations.

A ConversationTree manages a set of branches, each with its own message
history that diverges from a shared ancestor.  Users can:

- branch at any message index to create a parallel timeline
- switch between branches freely
- compare two branches side-by-side
- merge insights from multiple branches back together

Usage:
    from agentos.core.branching import ConversationTree

    tree = ConversationTree()
    tree.add_message("user", "What is the best investment strategy?")
    tree.add_message("assistant", "It depends on your risk tolerance...")
    tree.add_message("user", "I have $100k to invest")
    tree.add_message("assistant", "Here are some options...")

    # Fork at message index 2 to explore a different path
    new_id = tree.branch(at_message_index=2, label="aggressive")
    tree.switch_branch(new_id)
    tree.add_message("user", "I want high risk, high reward")
    tree.add_message("assistant", "Consider growth stocks and crypto...")

    # Compare both paths
    comparison = tree.compare_branches("main", new_id)
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    index: int = 0  # position within the branch
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "index": self.index,
            "metadata": self.metadata,
        }

    def to_openai(self) -> dict:
        """Convert to OpenAI message format."""
        return {"role": self.role, "content": self.content}


@dataclass
class Branch:
    """A single conversation branch (timeline)."""

    branch_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label: str = "main"
    parent_branch_id: str | None = None
    branch_point: int = 0  # message index where this branch diverged
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def last_message(self) -> ChatMessage | None:
        return self.messages[-1] if self.messages else None

    def to_dict(self) -> dict:
        return {
            "branch_id": self.branch_id,
            "label": self.label,
            "parent_branch_id": self.parent_branch_id,
            "branch_point": self.branch_point,
            "message_count": self.message_count,
            "created_at": self.created_at,
            "last_message": self.last_message.to_dict() if self.last_message else None,
            "metadata": self.metadata,
        }

    def get_messages_as_openai(self) -> list[dict]:
        """Get all messages in OpenAI format."""
        return [m.to_openai() for m in self.messages]


class ConversationTree:
    """Manages a tree of conversation branches.

    The tree starts with a single "main" branch.  Users can fork at any
    message index to create parallel timelines, switch between them,
    compare outcomes, and merge insights.
    """

    def __init__(self, tree_id: str | None = None):
        self.tree_id = tree_id or uuid.uuid4().hex[:12]
        self.created_at = time.time()

        # Bootstrap with a "main" branch
        main = Branch(label="main")
        self.branches: dict[str, Branch] = {main.branch_id: main}
        self._active_branch_id: str = main.branch_id
        self._main_branch_id: str = main.branch_id

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_branch(self) -> Branch:
        return self.branches[self._active_branch_id]

    @property
    def active_branch_id(self) -> str:
        return self._active_branch_id

    @property
    def main_branch_id(self) -> str:
        return self._main_branch_id

    # ------------------------------------------------------------------
    # Message management (acts on active branch)
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """Append a message to the currently active branch."""
        branch = self.active_branch
        msg = ChatMessage(
            role=role,
            content=content,
            index=len(branch.messages),
            metadata=metadata or {},
        )
        branch.messages.append(msg)
        return msg

    def get_messages(self, branch_id: str | None = None) -> list[ChatMessage]:
        """Get all messages for a branch (default: active)."""
        bid = branch_id or self._active_branch_id
        branch = self.branches.get(bid)
        if not branch:
            raise KeyError(f"Branch '{bid}' not found")
        return list(branch.messages)

    def get_messages_openai(self, branch_id: str | None = None) -> list[dict]:
        """Get messages in OpenAI format for a branch."""
        bid = branch_id or self._active_branch_id
        branch = self.branches.get(bid)
        if not branch:
            raise KeyError(f"Branch '{bid}' not found")
        return branch.get_messages_as_openai()

    # ------------------------------------------------------------------
    # Branching
    # ------------------------------------------------------------------

    def branch(
        self,
        at_message_index: int | None = None,
        label: str = "",
        source_branch_id: str | None = None,
    ) -> str:
        """Create a new branch that forks from *source_branch* at *at_message_index*.

        Messages up to and including *at_message_index* are deep-copied into
        the new branch.  If *at_message_index* is ``None``, the branch starts
        from the latest message.

        Returns the new branch ID.
        """
        source_id = source_branch_id or self._active_branch_id
        source = self.branches.get(source_id)
        if not source:
            raise KeyError(f"Source branch '{source_id}' not found")

        if at_message_index is None:
            at_message_index = len(source.messages) - 1

        if at_message_index < 0 or at_message_index >= len(source.messages):
            at_message_index = max(0, min(at_message_index, len(source.messages) - 1))

        # Deep-copy messages up to the branch point
        copied = copy.deepcopy(source.messages[: at_message_index + 1])

        new_branch = Branch(
            label=label or f"branch-{len(self.branches)}",
            parent_branch_id=source_id,
            branch_point=at_message_index,
            messages=copied,
        )
        self.branches[new_branch.branch_id] = new_branch
        return new_branch.branch_id

    def switch_branch(self, branch_id: str) -> Branch:
        """Switch the active branch."""
        if branch_id not in self.branches:
            raise KeyError(f"Branch '{branch_id}' not found")
        self._active_branch_id = branch_id
        return self.branches[branch_id]

    def delete_branch(self, branch_id: str) -> bool:
        """Delete a branch (cannot delete the main branch)."""
        if branch_id == self._main_branch_id:
            return False
        if branch_id not in self.branches:
            return False
        if branch_id == self._active_branch_id:
            self._active_branch_id = self._main_branch_id
        del self.branches[branch_id]
        return True

    # ------------------------------------------------------------------
    # Listing & inspection
    # ------------------------------------------------------------------

    def list_branches(self) -> list[dict]:
        """Return metadata for every branch, sorted by creation time."""
        result = []
        for b in sorted(self.branches.values(), key=lambda x: x.created_at):
            d = b.to_dict()
            d["is_active"] = b.branch_id == self._active_branch_id
            d["is_main"] = b.branch_id == self._main_branch_id
            result.append(d)
        return result

    def get_branch(self, branch_id: str) -> Branch:
        if branch_id not in self.branches:
            raise KeyError(f"Branch '{branch_id}' not found")
        return self.branches[branch_id]

    # ------------------------------------------------------------------
    # Compare & merge
    # ------------------------------------------------------------------

    def compare_branches(self, branch_a_id: str, branch_b_id: str) -> dict:
        """Side-by-side comparison of two branches.

        Returns a dict with:
        - shared_messages: messages that are identical (before branch point)
        - branch_a_unique: messages only in branch A
        - branch_b_unique: messages only in branch B
        - stats for both branches
        """
        a = self.branches.get(branch_a_id)
        b = self.branches.get(branch_b_id)
        if not a:
            raise KeyError(f"Branch '{branch_a_id}' not found")
        if not b:
            raise KeyError(f"Branch '{branch_b_id}' not found")

        # Find the longest common prefix
        shared_len = 0
        for i in range(min(len(a.messages), len(b.messages))):
            if (
                a.messages[i].role == b.messages[i].role
                and a.messages[i].content == b.messages[i].content
            ):
                shared_len = i + 1
            else:
                break

        shared = [m.to_dict() for m in a.messages[:shared_len]]
        a_unique = [m.to_dict() for m in a.messages[shared_len:]]
        b_unique = [m.to_dict() for m in b.messages[shared_len:]]

        return {
            "branch_a": {"id": branch_a_id, "label": a.label, "total_messages": len(a.messages)},
            "branch_b": {"id": branch_b_id, "label": b.label, "total_messages": len(b.messages)},
            "shared_messages": shared,
            "shared_count": shared_len,
            "branch_a_unique": a_unique,
            "branch_b_unique": b_unique,
        }

    def merge_branches(
        self,
        branch_a_id: str,
        branch_b_id: str,
        label: str = "merged",
    ) -> str:
        """Merge two branches by creating a new branch containing all shared
        context plus a summary of both divergent paths.

        This doesn't use an LLM — it combines the messages structurally.
        The merged branch contains:
        1. All shared messages (common prefix)
        2. A system-injected summary of both divergent paths
        3. Ready for the user to continue the conversation with full context
        """
        comparison = self.compare_branches(branch_a_id, branch_b_id)
        a = self.branches[branch_a_id]
        b = self.branches[branch_b_id]

        # Start the merged branch from the shared prefix
        shared_msgs = copy.deepcopy(a.messages[: comparison["shared_count"]])

        # Build a summary message of both paths
        a_summary_parts = []
        for m in comparison["branch_a_unique"]:
            a_summary_parts.append(f"[{m['role']}]: {m['content'][:200]}")
        b_summary_parts = []
        for m in comparison["branch_b_unique"]:
            b_summary_parts.append(f"[{m['role']}]: {m['content'][:200]}")

        merge_content = (
            f"--- MERGED CONTEXT ---\n"
            f"Two conversation paths explored from this point:\n\n"
            f"PATH A ({a.label}):\n"
            + "\n".join(a_summary_parts)
            + f"\n\nPATH B ({b.label}):\n"
            + "\n".join(b_summary_parts)
            + "\n\nBoth paths are provided as context. "
            "Continue the conversation incorporating insights from both explorations."
            "\n--- END MERGED CONTEXT ---"
        )

        merge_msg = ChatMessage(
            role="system",
            content=merge_content,
            index=len(shared_msgs),
            metadata={"type": "merge", "source_a": branch_a_id, "source_b": branch_b_id},
        )

        merged_messages = shared_msgs + [merge_msg]

        merged_branch = Branch(
            label=label,
            parent_branch_id=None,
            branch_point=comparison["shared_count"],
            messages=merged_messages,
            metadata={
                "merged_from": [branch_a_id, branch_b_id],
                "shared_count": comparison["shared_count"],
            },
        )
        self.branches[merged_branch.branch_id] = merged_branch
        return merged_branch.branch_id

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "tree_id": self.tree_id,
            "created_at": self.created_at,
            "active_branch_id": self._active_branch_id,
            "main_branch_id": self._main_branch_id,
            "branch_count": len(self.branches),
            "branches": self.list_branches(),
        }


# ---------------------------------------------------------------------------
# Global registry of conversation trees (in-memory)
# ---------------------------------------------------------------------------

_trees: dict[str, ConversationTree] = {}


def get_or_create_tree(tree_id: str | None = None) -> ConversationTree:
    """Get an existing tree or create a new one."""
    if tree_id and tree_id in _trees:
        return _trees[tree_id]
    tree = ConversationTree(tree_id=tree_id)
    _trees[tree.tree_id] = tree
    return tree


def get_tree(tree_id: str) -> ConversationTree | None:
    return _trees.get(tree_id)


def list_trees() -> list[dict]:
    return [t.to_dict() for t in _trees.values()]


def delete_tree(tree_id: str) -> bool:
    if tree_id in _trees:
        del _trees[tree_id]
        return True
    return False
