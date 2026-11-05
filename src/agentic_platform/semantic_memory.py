"""Optional vector-backed semantic memory with session isolation."""

from collections.abc import Callable
from pathlib import Path
from uuid import uuid4


class ChromaSemanticMemory:
    """Store and retrieve semantically relevant conversation memories."""

    def __init__(self, collection, embed: Callable[[str], list[float]]) -> None:
        self.collection = collection
        self.embed = embed

    @classmethod
    def from_local_model(
        cls,
        persist_directory: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "agent_memories",
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

        return cls(collection=collection, embed=embed)

    def add(self, session_id: str, content: str) -> str:
        """Persist one memory item inside the caller's session namespace."""
        memory_id = str(uuid4())
        self.collection.add(
            ids=[memory_id],
            embeddings=[self.embed(content)],
            documents=[content],
            metadatas=[{"session_id": session_id}],
        )
        return memory_id

    def search(
        self,
        session_id: str,
        query: str,
        limit: int = 3,
    ) -> list[dict[str, object]]:
        """Return nearest memories belonging only to the requested session."""
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        result = self.collection.query(
            query_embeddings=[self.embed(query)],
            n_results=limit,
            where={"session_id": session_id},
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        return [
            {
                "content": document,
                "metadata": metadata or {},
                "distance": distances[0][index] if distances and distances[0] else None,
            }
            for index, (document, metadata) in enumerate(
                zip(documents[0], metadatas[0], strict=False)
            )
        ]

    def clear(self, session_id: str) -> None:
        """Delete every vector memory associated with one session."""
        self.collection.delete(where={"session_id": session_id})
