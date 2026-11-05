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

    def query(self, query_embeddings, n_results, where, include):
        matches = [
            item for item in self.items.values()
            if item["metadata"]["session_id"] == where["session_id"]
        ]
        query = query_embeddings[0]
        matches.sort(
            key=lambda item: sum(
                (left - right) ** 2
                for left, right in zip(item["embedding"], query, strict=True)
            )
        )
        matches = matches[:n_results]
        return {
            "documents": [[item["document"] for item in matches]],
            "metadatas": [[item["metadata"] for item in matches]],
            "distances": [[0.0 for _ in matches]],
        }

    def delete(self, where):
        self.items = {
            item_id: item
            for item_id, item in self.items.items()
            if item["metadata"]["session_id"] != where["session_id"]
        }


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


def test_semantic_memory_clear_only_removes_target_session():
    collection = FakeCollection()
    memory = ChromaSemanticMemory(collection, fake_embed)
    memory.add("session-a", "Python project")
    memory.add("session-b", "Coffee preference")

    memory.clear("session-a")

    assert memory.search("session-a", "Python") == []
    assert len(memory.search("session-b", "coffee")) == 1


def test_semantic_memory_rejects_non_positive_limit():
    memory = ChromaSemanticMemory(FakeCollection(), fake_embed)

    try:
        memory.search("session-a", "query", limit=0)
    except ValueError as exc:
        assert "greater than zero" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
