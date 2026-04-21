# tests/unit/test_retriever_agent.py
import pytest
from unittest.mock import Mock, patch, AsyncMock
from agents.retriever_agent import retriever_agent
from graph.state import RAGState


@pytest.mark.asyncio
async def test_retriever_fetches_chunks_from_qdrant():
    """Should retrieve chunks from Qdrant and apply MMR."""
    state = RAGState(
        query="test query",
        chat_history=[],
        query_type="factual",
        sub_queries=["test query"],
        route="retriever",
        retrieved_chunks=[],
        retrieval_metadata={},
        answer="",
        citations=[],
        confidence="",
        reasoning_trace="",
        verdict="",
        critique="",
        final_answer=None,
        retry_count=0
    )

    # Create mock Qdrant search results with proper structure
    from qdrant_client.models import Record, ScoredPoint

    mock_hit_1 = Mock(spec=ScoredPoint)
    mock_hit_1.score = 0.9
    mock_hit_1.payload = {"text": "chunk 1", "source": "doc1.pdf", "chunk_id": "chunk_001"}

    mock_hit_2 = Mock(spec=ScoredPoint)
    mock_hit_2.score = 0.8
    mock_hit_2.payload = {"text": "chunk 2", "source": "doc2.pdf", "chunk_id": "chunk_002"}

    with patch("agents.retriever_agent.get_embeddings") as mock_get_emb:
        # Mock embeddings
        mock_emb = Mock()
        mock_emb.aembed_query = AsyncMock(return_value=[0.1] * 768)
        mock_get_emb.return_value = mock_emb

        with patch("agents.retriever_agent.QdrantClient") as mock_qdrant:
            mock_client = Mock()
            mock_client.search.return_value = [mock_hit_1, mock_hit_2]
            mock_qdrant.return_value = mock_client

            result = await retriever_agent(state, rerank=False)

            assert "retrieved_chunks" in result
            assert len(result["retrieved_chunks"]) > 0
            assert "retrieval_metadata" in result


@pytest.mark.asyncio
async def test_retriever_applies_cross_encoder_reranking():
    """Should rerank chunks using cross-encoder."""
    state = RAGState(
        query="test",
        chat_history=[],
        query_type="factual",
        sub_queries=["test"],
        route="retriever",
        retrieved_chunks=[],
        retrieval_metadata={},
        answer="",
        citations=[],
        confidence="",
        reasoning_trace="",
        verdict="",
        critique="",
        final_answer=None,
        retry_count=0
    )

    # Create mock Qdrant search results
    from qdrant_client.models import ScoredPoint

    mock_hit_1 = Mock(spec=ScoredPoint)
    mock_hit_1.score = 0.9
    mock_hit_1.payload = {"text": "chunk 1", "source": "doc1.pdf", "chunk_id": "chunk_001"}

    mock_hit_2 = Mock(spec=ScoredPoint)
    mock_hit_2.score = 0.8
    mock_hit_2.payload = {"text": "chunk 2", "source": "doc2.pdf", "chunk_id": "chunk_002"}

    with patch("agents.retriever_agent.get_embeddings") as mock_get_emb:
        # Mock embeddings
        mock_emb = Mock()
        mock_emb.aembed_query = AsyncMock(return_value=[0.1] * 768)
        mock_get_emb.return_value = mock_emb

        with patch("agents.retriever_agent.QdrantClient") as mock_qdrant:
            mock_client = Mock()
            mock_client.search.return_value = [mock_hit_1, mock_hit_2]
            mock_qdrant.return_value = mock_client

            with patch("agents.retriever_agent.get_cross_encoder") as mock_get_ce:
                mock_model = Mock()
                mock_model.rank.return_value = [
                    {"corpus_id": 1, "score": 0.95},
                    {"corpus_id": 0, "score": 0.75}
                ]
                mock_get_ce.return_value = mock_model

                result = await retriever_agent(state, top_k=10, rerank=True)

                # Verify reranking was called
                mock_model.rank.assert_called_once()
