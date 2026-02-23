"""Agent Memory — conversation history + persistent knowledge store.

Two types:
1. ConversationMemory — remembers past messages in a session
2. KnowledgeMemory — stores facts that persist across runs

Usage:
    memory = Memory(max_conversations=50, enable_knowledge=True)

    # Conversation memory (automatic)
    memory.add_message({"role": "user", "content": "My name is Suketh"})
    memory.add_message({"role": "assistant", "content": "Nice to meet you!"})

    # Knowledge memory (extracted facts)
    memory.store_fact("user_name", "Suketh")
    memory.store_fact("user_company", "AgentOS")

    # Retrieve
    messages = memory.get_messages()
    name = memory.get_fact("user_name")  # "Suketh"

    # Get context for the agent
    context = memory.get_context()  # Injects known facts into system prompt
"""

from __future__ import annotations
import time
from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    key: str
    value: str
    category: str = "general"
    timestamp: float = Field(default_factory=time.time)
    source: str = "agent"


class Memory:
    """Combined conversation + knowledge memory for agents."""

    def __init__(
        self,
        max_messages: int = 100,
        max_facts: int = 500,
        enable_knowledge: bool = True,
        enable_summary: bool = True,
    ):
        self.max_messages = max_messages
        self.max_facts = max_facts
        self.enable_knowledge = enable_knowledge
        self.enable_summary = enable_summary

        # Conversation memory
        self.messages: list[dict] = []
        self.conversation_count = 0

        # Knowledge memory (persistent facts)
        self.facts: dict[str, MemoryEntry] = {}

        # Conversation summaries
        self.summaries: list[str] = []

    # ── Conversation Memory ──

    def add_message(self, message: dict):
        """Add a message to conversation history."""
        self.messages.append(message)
        self.conversation_count += 1

        # Trim if over limit (keep system prompt + recent messages)
        if len(self.messages) > self.max_messages:
            system_msgs = [m for m in self.messages if m.get("role") == "system"]
            recent = self.messages[-(self.max_messages - len(system_msgs)) :]
            self.messages = system_msgs + recent

    def add_exchange(self, user_msg: str, assistant_msg: str):
        """Add a user-assistant exchange."""
        self.add_message({"role": "user", "content": user_msg})
        self.add_message({"role": "assistant", "content": assistant_msg})

    def get_messages(self, limit: int | None = None) -> list[dict]:
        """Get conversation history."""
        if limit:
            return self.messages[-limit:]
        return self.messages

    def get_last_exchange(self) -> tuple[str, str] | None:
        """Get the last user-assistant exchange."""
        user_msg = None
        asst_msg = None
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant" and asst_msg is None:
                asst_msg = msg.get("content", "")
            elif msg.get("role") == "user" and user_msg is None:
                user_msg = msg.get("content", "")
            if user_msg and asst_msg:
                return (user_msg, asst_msg)
        return None

    def clear_conversation(self):
        """Clear conversation history but keep knowledge."""
        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        self.messages = system_msgs

    # ── Knowledge Memory ──

    def store_fact(
        self, key: str, value: str, category: str = "general", source: str = "agent"
    ):
        """Store a persistent fact."""
        if len(self.facts) >= self.max_facts and key not in self.facts:
            # Remove oldest fact
            oldest_key = min(self.facts, key=lambda k: self.facts[k].timestamp)
            del self.facts[oldest_key]

        self.facts[key] = MemoryEntry(
            key=key,
            value=value,
            category=category,
            source=source,
        )

    def get_fact(self, key: str) -> str | None:
        """Retrieve a stored fact."""
        entry = self.facts.get(key)
        return entry.value if entry else None

    def search_facts(self, query: str) -> list[MemoryEntry]:
        """Search facts by key or value containing query."""
        query_lower = query.lower()
        results = []
        for entry in self.facts.values():
            if query_lower in entry.key.lower() or query_lower in entry.value.lower():
                results.append(entry)
        return results

    def get_facts_by_category(self, category: str) -> list[MemoryEntry]:
        """Get all facts in a category."""
        return [e for e in self.facts.values() if e.category == category]

    def delete_fact(self, key: str) -> bool:
        """Delete a fact."""
        if key in self.facts:
            del self.facts[key]
            return True
        return False

    # ── Context Building ──

    def get_context(self) -> str:
        """Build a context string from stored knowledge for injection into system prompt."""
        if not self.facts:
            return ""

        lines = ["Here is what you know from previous conversations:"]

        # Group by category
        categories: dict[str, list[MemoryEntry]] = {}
        for entry in self.facts.values():
            cat = entry.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(entry)

        for cat, entries in categories.items():
            lines.append(f"\n[{cat.upper()}]")
            for e in entries:
                lines.append(f"- {e.key}: {e.value}")

        return "\n".join(lines)

    def build_messages(self, system_prompt: str, user_input: str) -> list[dict]:
        """Build the full message list with memory context injected."""
        # Inject knowledge into system prompt
        context = self.get_context()
        if context:
            full_system = f"{system_prompt}\n\n{context}"
        else:
            full_system = system_prompt

        # Start with system prompt
        messages = [{"role": "system", "content": full_system}]

        # Add conversation history (skip old system messages)
        for msg in self.messages:
            if msg.get("role") != "system":
                messages.append(msg)

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        return messages

    # ── Auto-Extract Facts from Conversation ──

    def extract_facts_from_response(self, user_msg: str, assistant_msg: str):
        """Simple rule-based fact extraction from conversation.
        For production, replace with LLM-based extraction.
        """
        user_lower = user_msg.lower()

        # Extract name
        name_triggers = ["my name is", "i'm called", "call me", "i am"]
        for trigger in name_triggers:
            if trigger in user_lower:
                idx = user_lower.index(trigger) + len(trigger)
                name = user_msg[idx:].strip().split()[0].strip(".,!?")
                if name and len(name) > 1:
                    self.store_fact(
                        "user_name", name, category="personal", source="extracted"
                    )

        # Extract location
        location_triggers = ["i live in", "i'm from", "i'm based in", "located in"]
        for trigger in location_triggers:
            if trigger in user_lower:
                idx = user_lower.index(trigger) + len(trigger)
                location = user_msg[idx:].strip().split(".")[0].strip()
                if location:
                    self.store_fact(
                        "user_location",
                        location,
                        category="personal",
                        source="extracted",
                    )

        # Extract company
        company_triggers = ["i work at", "i work for", "my company is", "i'm at"]
        for trigger in company_triggers:
            if trigger in user_lower:
                idx = user_lower.index(trigger) + len(trigger)
                company = user_msg[idx:].strip().split(".")[0].strip()
                if company:
                    self.store_fact(
                        "user_company", company, category="work", source="extracted"
                    )

        # Extract preferences
        pref_triggers = ["i like", "i love", "i prefer", "my favorite"]
        for trigger in pref_triggers:
            if trigger in user_lower:
                idx = user_lower.index(trigger) + len(trigger)
                pref = user_msg[idx:].strip().split(".")[0].strip()
                if pref:
                    key = f"preference_{len([k for k in self.facts if k.startswith('preference')])}"
                    self.store_fact(
                        key,
                        f"{trigger} {pref}",
                        category="preferences",
                        source="extracted",
                    )

    # ── Stats ──

    def get_stats(self) -> dict:
        return {
            "total_messages": len(self.messages),
            "conversation_count": self.conversation_count,
            "total_facts": len(self.facts),
            "categories": list(set(e.category for e in self.facts.values())),
            "summaries": len(self.summaries),
        }

    def print_memory(self):
        stats = self.get_stats()
        print(f"\n{'=' * 60}")
        print("🧠 Agent Memory")
        print(f"{'=' * 60}")
        print(f"   Messages:     {stats['total_messages']}")
        print(f"   Exchanges:    {stats['conversation_count']}")
        print(f"   Facts stored: {stats['total_facts']}")
        print(
            f"   Categories:   {', '.join(stats['categories']) if stats['categories'] else 'none'}"
        )

        if self.facts:
            print("\n   📚 Stored Knowledge:")
            for key, entry in self.facts.items():
                print(f"      [{entry.category}] {key}: {entry.value}")
        print(f"{'=' * 60}")
