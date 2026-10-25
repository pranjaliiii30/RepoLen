"""Chroma persistence and embedding-backed repository indexing."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from core.models import Chunk

EmbeddingModel = Any


def create_collection(
    client: Any,
    collection_name: str,
    chunks: Sequence[Chunk],
    embedder: EmbeddingModel,
    batch_size: int = 64,
) -> Any:
    """Create and populate a cosine-similarity collection."""
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    try:
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            embeddings = embedder.encode(
                [chunk.content for chunk in batch],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            collection.add(
                ids=[f"chunk-{offset + index}" for index in range(len(batch))],
                documents=[chunk.content for chunk in batch],
                embeddings=np.asarray(embeddings, dtype=np.float32).tolist(),
                metadatas=[
                    {
                        "path": chunk.path,
                        "symbol": chunk.symbol,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "language": chunk.language,
                        "folder": chunk.path.rpartition("/")[0] or "(root)",
                    }
                    for chunk in batch
                ],
            )
        return collection
    except Exception:
        client.delete_collection(name=collection_name)
        raise


def load_collection(client: Any, collection_name: str) -> Any:
    """Get a persisted collection by its registered name."""
    return client.get_collection(name=collection_name)


def collection_records(collection: Any) -> tuple[list[str], list[dict[str, Any]], np.ndarray]:
    """Load all indexed documents, metadata, and embeddings for hybrid retrieval."""
    result = collection.get(include=["documents", "metadatas", "embeddings"])
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    embeddings = result.get("embeddings")
    if not documents or embeddings is None:
        return [], [], np.empty((0, 0), dtype=np.float32)
    return (
        documents,
        metadatas,
        np.asarray(embeddings, dtype=np.float32),
    )
