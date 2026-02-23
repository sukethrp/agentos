from __future__ import annotations
from agentos.learning.feedback import FeedbackStore, FeedbackEntry, FeedbackType
from agentos.learning.few_shot import FewShotBuilder
from agentos.learning.prompt_optimizer import PromptOptimizer


def test_feedback_record(tmp_path):
    store = FeedbackStore(data_dir=str(tmp_path))
    entry = FeedbackEntry(
        agent_name="test-agent",
        feedback_type=FeedbackType.THUMBS_UP,
        query="What is 2+2?",
        response="4",
    )
    stored = store.add(entry)
    assert stored.id == entry.id
    assert stored.query == "What is 2+2?"
    assert stored.response == "4"
    all_entries = store.all()
    assert len(all_entries) == 1
    assert all_entries[0].query == "What is 2+2?"


def test_few_shot_retrieval(tmp_path):
    store = FeedbackStore(data_dir=str(tmp_path))
    store.add(
        FeedbackEntry(
            feedback_type=FeedbackType.THUMBS_UP,
            query="refund policy",
            response="Our refund policy allows 30 days.",
            topic="refund",
        )
    )
    store.add(
        FeedbackEntry(
            feedback_type=FeedbackType.THUMBS_UP,
            query="billing question",
            response="Your bill is $50.",
            topic="billing",
        )
    )
    store.add(
        FeedbackEntry(
            feedback_type=FeedbackType.THUMBS_UP,
            query="cancel subscription",
            response="You can cancel in settings.",
            topic="refund",
        )
    )
    builder = FewShotBuilder(store=store, max_examples=3, max_per_topic=2)
    examples = builder.build()
    assert len(examples) >= 1
    msgs = builder.to_messages()
    assert len(msgs) >= 2
    query = "refund"
    matching = [
        e for e in examples if query in e.query.lower() or query in e.topic.lower()
    ]
    assert len(matching) >= 1


def test_prompt_optimizer_returns_improved(tmp_path):
    store = FeedbackStore(data_dir=str(tmp_path))
    store.add(
        FeedbackEntry(
            feedback_type=FeedbackType.THUMBS_DOWN,
            query="refund",
            response="bad",
            topic="refund",
        )
    )
    store.add(
        FeedbackEntry(
            feedback_type=FeedbackType.CORRECTION,
            query="refund",
            response="wrong",
            correction="Our refund policy is 30 days.",
            topic="refund",
        )
    )
    optimizer = PromptOptimizer(store=store, use_llm=False)
    patches = optimizer.optimize()
    assert len(patches) >= 1
    applied = optimizer.apply_patches("You are helpful.")
    assert "You are helpful." in applied
    assert len(applied) > len("You are helpful.")
