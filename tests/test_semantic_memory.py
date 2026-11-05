from time import time

import pytest

from agentic_platform.semantic_memory import ChromaSemanticMemory


class FakeCollection:
    def __init__(self):
        self.items = {}

    def add(self, ids, embeddings, documents, metadatas):
        for item_id, embedding, document, metadata in zip(
            ids, embeddings, documents, metadatas, strict=True
        ):
            self.items[item_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            }

    def update(self, ids, metadatas):
        for item_id, metadata in zip(ids, metadatas, strict=True):
            self.items[item_id]["metadata"] = metadata

    def query(self, query_embeddings, n_results, where, include):
        matches = [
            (item_id, item)
            for item_id, item in self.items.items()
            if item["metadata"]["session_id"] == where["session_id"]
        ]
        query = query_embeddings[0]
        matches.sort(
            key=lambda pair: sum(
                (left - right) ** 2
                for left, right in zip(pair[1]["embedding"], query, strict=True)
            )
        )
        matches = matches[:n_results]
        return {
            "ids": [[item_id for item_id, _ in matches]],
            "documents": [[item["document"] for _, item in matches]],
            "metadatas": [[item["metadata"] for _, item in matches]],
            "distances": [[
                sum(
                    (left - right) ** 2
                    for left, right in zip(item["embedding"], query, strict=True)
                ) ** 0.5
                for _, item in matches
            ]],
        }

    def get(self, where=None, ids=None, include=None):
        matches = [
            (item_id, item)
            for item_id, item in self.items.items()
            if (ids is None or item_id in ids)
            and (
                where is None
                or item["metadata"]["session_id"] == where["session_id"]
            )
        ]
        return {
            "ids": [item_id for item_id, _ in matches],
            "documents": [item["document"] for _, item in matches],
            "metadatas": [item["metadata"] for _, item in matches],
        }

    def delete(self, ids=None, where=None):
        targets = [
            item_id
            for item_id, item in self.items.items()
            if (ids is None or item_id in ids)
            and (
                where is None
                or item["metadata"]["session_id"] == where["session_id"]
            )
        ]
        for item_id in targets:
            del self.items[item_id]


def fake_embed(text: str) -> list[float]:
    lowered = text.lower()
    return [
        float("python" in lowered),
        float("coffee" in lowered),
        float("machine learning" in lowered),
    ]


def test_semantic_memory_is_session_scoped_and_retrievable():
    collection = FakeCollection()
    memory = ChromaSemanticMemory(collection, fake_embed)
    memory.add("session-a", "The user studies machine learning.")
    memory.add("session-b", "The user likes coffee.")

    results = memory.search("session-a", "machine learning", limit=2)

    assert len(results) == 1
    assert results[0]["content"] == "The user studies machine learning."
    assert results[0]["metadata"]["session_id"] == "session-a"


def test_semantic_memory_deduplicates_and_keeps_highest_importance():
    collection = FakeCollection()
    memory = ChromaSemanticMemory(collection, fake_embed)
    first_id = memory.add("session-a", "Python project", importance=0.4)
    second_id = memory.add("session-a", "Python project", importance=0.9)

    assert first_id == second_id
    assert len(collection.items) == 1
    assert collection.items[first_id]["metadata"]["importance"] == 0.9


def test_semantic_memory_expires_old_memories():
    collection = FakeCollection()
    memory = ChromaSemanticMemory(collection, fake_embed, ttl_days=1)
    memory.add("session-a", "Python project")
    memory_id = next(iter(collection.items))
    collection.items[memory_id]["metadata"]["expires_at"] = int(time()) - 1

    assert memory.search("session-a", "Python") == []
    assert memory.prune("session-a")["expired"] == 1
    assert collection.items == {}


def test_semantic_memory_enforces_per_session_retention_cap():
    collection = FakeCollection()
    memory = ChromaSemanticMemory(
        collection, fake_embed, max_items_per_session=2
    )
    memory.add("session-a", "Python project", importance=0.1)
    memory.add("session-a", "Coffee preference", importance=0.9)
    memory.add("session-a", "Machine learning notes", importance=0.8)

    stored = memory.list_memories("session-a")

    assert len(stored) == 2
    assert all(item["metadata"]["importance"] >= 0.8 for item in stored)


def test_memory_cannot_be_deleted_from_another_session():
    collection = FakeCollection()
    memory = ChromaSemanticMemory(collection, fake_embed)
    memory_id = memory.add("session-a", "Python project")

    assert memory.delete("session-b", memory_id) is False
    assert memory.delete("session-a", memory_id) is True


def test_semantic_memory_clear_only_removes_target_session():
    collection = FakeCollection()
    memory = ChromaSemanticMemory(collection, fake_embed)
    memory.add("session-a", "Python project")
    memory.add("session-b", "Coffee preference")

    memory.clear("session-a")

    assert memory.search("session-a", "Python") == []
    assert len(memory.search("session-b", "coffee")) == 1


def test_semantic_memory_rejects_invalid_limits_and_policies():
    with pytest.raises(ValueError):
        ChromaSemanticMemory(FakeCollection(), fake_embed, max_items_per_session=0)
    with pytest.raises(ValueError):
        ChromaSemanticMemory(FakeCollection(), fake_embed, ttl_days=-1)
    memory = ChromaSemanticMemory(FakeCollection(), fake_embed)
    with pytest.raises(ValueError):
        memory.search("session-a", "query", limit=0)
