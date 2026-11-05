from agentic_platform.memory import SQLiteConversationMemory


def test_memory_persists_and_is_isolated_by_session(tmp_path):
    memory = SQLiteConversationMemory(tmp_path / "memory.db")

    memory.add_turn("session-a", "I prefer Python.", "I will use Python examples.")
    memory.add_turn("session-b", "I prefer Rust.", "I will use Rust examples.")

    assert memory.get_history("session-a") == [
        {"role": "user", "content": "I prefer Python."},
        {"role": "assistant", "content": "I will use Python examples."},
    ]
    assert memory.get_history("session-b")[0]["content"] == "I prefer Rust."


def test_memory_prunes_old_messages_to_configured_limit(tmp_path):
    memory = SQLiteConversationMemory(tmp_path / "memory.db", max_messages=3)

    memory.add_turn("session-a", "first", "first answer")
    memory.add_turn("session-a", "second", "second answer")

    history = memory.get_history("session-a")
    assert [item["content"] for item in history] == ["first answer", "second", "second answer"]


def test_memory_can_be_cleared_without_affecting_other_sessions(tmp_path):
    memory = SQLiteConversationMemory(tmp_path / "memory.db")
    memory.add_turn("session-a", "hello", "hi")
    memory.add_turn("session-b", "keep this", "still here")

    assert memory.clear("session-a") == 2
    assert memory.get_history("session-a") == []
    assert memory.get_history("session-b")[0]["content"] == "keep this"
