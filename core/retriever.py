"""Hybrid vector and BM25 retrieval with reciprocal-rank fusion."""

import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from core.indexer import collection_records
from core.models import Chunk, RetrievedChunk

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def hybrid_search(
    collection: Any,
    query: str,
    embedder: Any,
    top_k: int = 6,
    language: str | None = None,
    folder: str | None = None,
    query_rewriter: Callable[[str], list[str]] | None = None,
) -> list[RetrievedChunk]:
    """Retrieve matching chunks using vector and lexical rankings fused by RRF."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    documents, metadatas, vectors = collection_records(collection)
    if not documents:
        return []

    selected = [
        index
        for index, metadata in enumerate(metadatas)
        if (not language or metadata.get("language") == language)
        and (not folder or metadata.get("folder") == folder)
    ]
    if not selected:
        return []

    rewritten: list[str] = []
    if query_rewriter:
        rewritten = [candidate.strip() for candidate in query_rewriter(query) if candidate.strip()]
    queries = list(dict.fromkeys([query, *rewritten]))[:4]
    corpus = [_tokenize(documents[index]) for index in selected]
    bm25 = BM25Okapi(corpus)
    fused: dict[int, float] = defaultdict(float)
    similarities: dict[int, float] = {}
    rrf_constant = 60

    for search_query in queries:
        query_vector = np.asarray(
            embedder.encode(
                search_query,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        ).reshape(-1)
        cosine_scores = vectors[selected] @ query_vector
        lexical_scores = bm25.get_scores(_tokenize(search_query))
        vector_order = np.argsort(cosine_scores)[::-1]
        lexical_order = np.argsort(lexical_scores)[::-1]
        for ranking in (vector_order, lexical_order):
            for rank, local_index in enumerate(ranking[: max(top_k * 5, top_k)], start=1):
                global_index = selected[int(local_index)]
                fused[global_index] += 1 / (rrf_constant + rank)
        for local_index, score in enumerate(cosine_scores):
            global_index = selected[local_index]
            similarities[global_index] = max(
                similarities.get(global_index, -1.0), float(score)
            )

    ranked = sorted(fused, key=fused.get, reverse=True)[:top_k]
    results: list[RetrievedChunk] = []
    for index in ranked:
        metadata = metadatas[index]
        chunk = Chunk(
            path=str(metadata["path"]),
            symbol=str(metadata["symbol"]),
            start_line=int(metadata["start_line"]),
            end_line=int(metadata["end_line"]),
            content=documents[index],
            language=str(metadata["language"]),
        )
        results.append(
            RetrievedChunk(
                chunk=chunk,
                score=float(fused[index]),
                similarity=max(0.0, min(1.0, similarities.get(index, 0.0))),
            )
        )
    return results
