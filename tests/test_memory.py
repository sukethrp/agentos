from agentos.core.memory import Memory


def test_add_message_and_get_messages_trimming():
    mem = Memory(max_messages=3)
    mem.add_message({"role": "system", "content": "sys"})
    mem.add_message({"role": "user", "content": "u1"})
    mem.add_message({"role": "assistant", "content": "a1"})
    mem.add_message({"role": "user", "content": "u2"})

    msgs = mem.get_messages()
    # Should keep system plus latest non-system messages
    assert len(msgs) <= 3
    assert any(m.get("role") == "system" for m in msgs)


def test_store_get_and_search_facts():
    mem = Memory()
    mem.store_fact("user_name", "Alice", category="personal")
    mem.store_fact("user_company", "AgentOS", category="work")

    assert mem.get_fact("user_name") == "Alice"

    results = mem.search_facts("agentos")
    assert any(r.key == "user_company" for r in results)


def test_extract_facts_from_response_rules():
    mem = Memory()
    user_msg = "Hi, my name is Bob. I live in Berlin. I work at AgentOS. I like pizza."
    mem.extract_facts_from_response(user_msg, "")

    assert mem.get_fact("user_name") == "Bob"
    assert mem.get_fact("user_location") is not None
    assert mem.get_fact("user_company") == "AgentOS"
    prefs = [k for k in mem.facts.keys() if k.startswith("preference_")]
    assert prefs


def test_build_messages_context_injection_and_history():
    mem = Memory()
    mem.store_fact("user_name", "Carol", category="personal")
    mem.add_message({"role": "user", "content": "Previous question"})

    messages = mem.build_messages("You are a bot.", "Current question")

    # First message is system with injected context
    assert messages[0]["role"] == "system"
    assert "You are a bot." in messages[0]["content"]
    assert "user_name" in messages[0]["content"]

    # History preserved
    roles = [m["role"] for m in messages]
    assert roles[-1] == "user"
    assert any(m.get("content") == "Previous question" for m in messages)
