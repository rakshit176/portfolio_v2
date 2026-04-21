# agents/retriever_agent.py
from typing import Dict, Any, List
from graph.state import RAGState
from utils.providers import get_embeddings
from utils.logger import get_logger
from utils.cache import CacheManager
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import CrossEncoder

logger = get_logger("retriever")

# Global cross-encoder model (lazy loaded)
_cross_encoder = None


def get_cross_encoder():
    """Get or load cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


async def retriever_agent(
    state: RAGState,
    top_k: int = 5,
    use_mmr: bool = True,
    rerank: bool = True
) -> Dict[str, Any]:
    """
    Retrieve relevant chunks from Qdrant.

    Args:
        state: Current graph state
        top_k: Number of chunks to return
        use_mmr: Whether to use Maximal Marginal Relevance
        rerank: Whether to apply cross-encoder reranking

    Returns:
        Updated state with retrieved_chunks and retrieval_metadata
    """
    sub_queries = state["sub_queries"]

    logger.info("retrieval_start", extra={"num_queries": len(sub_queries), "top_k": top_k})

    # Get embeddings
    embeddings = get_embeddings()

    # Connect to Qdrant
    client = QdrantClient(url="http://localhost:6334")

    all_chunks = []

    for query in sub_queries:
        # Embed query
        query_embedding = await embeddings.aembed_query(query)

        # Search Qdrant
        search_results = client.search(
            collection_name="virallens_docs",
            query_vector=query_embedding,
            limit=top_k * 2 if rerank else top_k,
            with_payload=True
        )

        # Convert to chunk format
        chunks = [
            {
                "text": hit.payload.get("text", ""),
                "source": hit.payload.get("source", ""),
                "score": hit.score,
                "chunk_id": hit.payload.get("chunk_id", ""),
            }
            for hit in search_results
        ]

        all_chunks.extend(chunks)

    # Deduplicate by chunk_id
    seen_ids = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk["chunk_id"] not in seen_ids:
            seen_ids.add(chunk["chunk_id"])
            unique_chunks.append(chunk)

    # Apply cross-encoder reranking if enabled
    if rerank and unique_chunks:
        unique_chunks = await _rerank_chunks(sub_queries[0], unique_chunks, top_k)

    # Apply MMR to reduce redundancy
    if use_mmr and len(unique_chunks) > top_k:
        unique_chunks = _apply_mmr(unique_chunks, top_k)

    # Take top_k
    final_chunks = unique_chunks[:top_k]

    logger.info("retrieval_complete", extra={"num_chunks": len(final_chunks)})

    return {
        "retrieved_chunks": final_chunks,
        "retrieval_metadata": {
            "num_queries": len(sub_queries),
            "total_retrieved": len(final_chunks),
            "top_k": top_k
        }
    }


async def _rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int
) -> List[Dict[str, Any]]:
    """Rerank chunks using cross-encoder."""
    model = get_cross_encoder()

    # Prepare documents (just the text from chunks)
    documents = [chunk["text"] for chunk in chunks]

    # Rank using cross-encoder
    ranked_results = model.rank(query, documents, top_k=top_k)

    # Reorder chunks by rerank scores
    reranked = []
    for result in ranked_results:
        chunk = chunks[result["corpus_id"]].copy()
        chunk["rerank_score"] = result["score"]
        reranked.append(chunk)

    return reranked


def _apply_mmr(
    chunks: List[Dict[str, Any]],
    top_k: int,
    lambda_param: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Apply Maximal Marginal Relevance to reduce redundancy.

    Args:
        chunks: List of chunks with embeddings
        top_k: Number of chunks to return
        lambda_param: Balance between relevance and diversity (0-1)

    Returns:
        Filtered chunks
    """
    import numpy as np

    if len(chunks) <= top_k:
        return chunks

    # Simple implementation: select diverse chunks based on score threshold
    selected = [chunks[0]]  # Always take top result

    for chunk in chunks[1:]:
        # Check similarity with already selected chunks
        is_diverse = True
        for sel in selected:
            # Simple heuristic: skip if scores are too close
            if abs(chunk["score"] - sel["score"]) < 0.1:
                is_diverse = False
                break

        if is_diverse:
            selected.append(chunk)
            if len(selected) >= top_k:
                break

    return selected
