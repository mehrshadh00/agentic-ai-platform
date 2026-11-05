"""Session-isolated vector memory with retention and deduplication policies."""

from collections.abc import Callable
from pathlib import Path
from time import time
from uuid import uuid4


class ChromaSemanticMemory:
    """Manage semantic memories, including deduplication, expiry, and retention."""

    def __init__(
        self,
        collection,
        embed: Callable[[str], list[float]],
        *,
        duplicate_distance_threshold: float = 0.08,
        max_items_per_session: int = 100,
        ttl_days: int = 0,
    ) -> None:
        if not 0.0 <= duplicate_distance_threshold <= 2.0:
            raise ValueError("duplicate_distance_threshold must be between 0 and 2")
        if max_items_per_session < 1:
            raise ValueError("max_items_per_session must be positive")
        if ttl_days < 0:
            raise ValueError("ttl_days cannot be negative")
        self.collection = collection
        self.embed = embed
        self.duplicate_distance_threshold = duplicate_distance_threshold
        self.max_items_per_session = max_items_per_session
        self.ttl_seconds = ttl_days * 86400

    @classmethod
    def from_local_model(
        cls,
        persist_directory: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "agent_memories",
        **policy,
    ) -> "ChromaSemanticMemory":
        """Load optional ChromaDB and SentenceTransformers dependencies lazily."""
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic memory requires the optional 'semantic-memory' dependencies. "
                "Install with: pip install -e '.[semantic-memory]'"
            ) from exc

        client = chromadb.PersistentClient(path=str(persist_directory))
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        model = SentenceTransformer(model_name)

        def embed(text: str) -> list[float]:
            return model.encode(text, normalize_embeddings=True).tolist()

        return cls(collection=collection, embed=embed, **policy)

    def add(self, session_id: str, content: str, importance: float = 0.5) -> str:
        """Add a memory, refreshing a near-duplicate instead of storing it twice."""
        if not content.strip():
            raise ValueError("memory content cannot be empty")
        if not 0.0 <= importance <= 1.0:
            raise ValueError("importance must be between 0 and 1")

        now = int(time())
        existing = self.collection.query(
            query_embeddings=[self.embed(content)],
            n_results=1,
            where={"session_id": session_id},
            include=["metadatas", "distances"],
        )
        ids = existing.get("ids") or [[]]
        distances = existing.get("distances") or [[]]
        if (
            ids
            and ids[0]
            and distances
            and distances[0]
            and distances[0][0] <= self.duplicate_distance_threshold
        ):
            memory_id = ids[0][0]
            metadata = (existing.get("metadatas") or [[]])[0][0] or {}
            updated = {
                **metadata,
                "importance": max(float(metadata.get("importance", 0.5)), importance),
                "updated_at": now,
                "expires_at": now + self.ttl_seconds if self.ttl_seconds else 0,
            }
            self.collection.update(ids=[memory_id], metadatas=[updated])
            self.prune(session_id)
            return memory_id

        memory_id = str(uuid4())
        self.collection.add(
            ids=[memory_id],
            embeddings=[self.embed(content)],
            documents=[content.strip()],
            metadatas=[{
                "session_id": session_id,
                "importance": importance,
                "created_at": now,
                "updated_at": now,
                "expires_at": now + self.ttl_seconds if self.ttl_seconds else 0,
            }],
        )
        self.prune(session_id)
        return memory_id

    def search(
        self,
        session_id: str,
        query: str,
        limit: int = 3,
    ) -> list[dict[str, object]]:
        """Return relevant, non-expired memories for exactly one session."""
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        result = self.collection.query(
            query_embeddings=[self.embed(query)],
            n_results=max(limit * 5, limit),
            where={"session_id": session_id},
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        now = int(time())
        matches = []
        for index, (document, metadata) in enumerate(
            zip(documents[0], metadatas[0], strict=False)
        ):
            metadata = metadata or {}
            expires_at = int(metadata.get("expires_at", 0))
            if expires_at and expires_at <= now:
                continue
            distance = distances[0][index] if distances and distances[0] else None
            importance = float(metadata.get("importance", 0.5))
            relevance = 1.0 - float(distance or 0.0)
            score = 0.8 * relevance + 0.2 * importance
            matches.append({
                "id": (result.get("ids") or [[]])[0][index],
                "content": document,
                "metadata": metadata,
                "distance": distance,
                "score": score,
            })
        matches.sort(key=lambda item: float(item["score"]), reverse=True)
        return matches[:limit]

    def prune(self, session_id: str) -> dict[str, int]:
        """Remove expired items and enforce the per-session retention cap."""
        result = self.collection.get(
            where={"session_id": session_id},
            include=["metadatas"],
        )
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        now = int(time())
        expired = [
            memory_id
            for memory_id, metadata in zip(ids, metadatas, strict=False)
            if metadata and metadata.get("expires_at")
            and int(metadata["expires_at"]) <= now
        ]
        if expired:
            self.collection.delete(ids=expired)

        remaining = [
            (memory_id, metadata or {})
            for memory_id, metadata in zip(ids, metadatas, strict=False)
            if memory_id not in set(expired)
        ]
        remaining.sort(
            key=lambda item: (
                float(item[1].get("importance", 0.5)),
                int(item[1].get("updated_at", item[1].get("created_at", 0))),
            ),
            reverse=True,
        )
        overflow = [item[0] for item in remaining[self.max_items_per_session:]]
        if overflow:
            self.collection.delete(ids=overflow)
        return {"expired": len(expired), "evicted": len(overflow)}

    def list_memories(self, session_id: str) -> list[dict[str, object]]:
        """List active memory metadata and content within one session."""
        result = self.collection.get(
            where={"session_id": session_id},
            include=["documents", "metadatas"],
        )
        now = int(time())
        memories = []
        for memory_id, document, metadata in zip(
            result.get("ids") or [],
            result.get("documents") or [],
            result.get("metadatas") or [],
            strict=False,
        ):
            metadata = metadata or {}
            expires_at = int(metadata.get("expires_at", 0))
            if expires_at and expires_at <= now:
                continue
            memories.append({
                "id": memory_id,
                "content": document,
                "metadata": metadata,
            })
        memories.sort(
            key=lambda item: int(item["metadata"].get("updated_at", 0)),
            reverse=True,
        )
        return memories

    def delete(self, session_id: str, memory_id: str) -> bool:
        """Delete one memory only if it belongs to the requested session."""
        result = self.collection.get(
            ids=[memory_id],
            where={"session_id": session_id},
            include=["metadatas"],
        )
        if not result.get("ids"):
            return False
        self.collection.delete(ids=[memory_id])
        return True

    def clear(self, session_id: str) -> None:
        """Delete every vector memory associated with one session."""
        self.collection.delete(where={"session_id": session_id})
