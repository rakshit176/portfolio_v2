# tests/unit/test_critic_agent.py
import pytest
from unittest.mock import Mock, patch, AsyncMock
from agents.critic_agent import critic_agent
from graph.state import RAGState


@pytest.mark.asyncio
async def test_critic_approves_good_answer():
    """Should approve faithful and complete answers."""
    state = RAGState(
        query="What are revenue drivers?",
        chat_history=[],
        query_type="factual",
        sub_queries=["revenue drivers"],
        route="retriever",
        retrieved_chunks=[
            {"text": "Product sales are the main driver", "chunk_id": "chunk_001"}
        ],
        retrieval_metadata={},
        answer="Product sales are the main revenue driver.",
        citations=["chunk_001"],
        confidence="high",
        reasoning_trace="found in chunk_001",
        verdict="",
        critique="",
        final_answer=None,
        retry_count=0
    )

    with patch("utils.prompts.CRITIC_PROMPT"):
        with patch("utils.providers.get_llm") as mock_get_llm:
            mock_llm = Mock()
            mock_llm.ainvoke = AsyncMock(return_value=Mock(
                content='{"faithfulness_score": 3, "completeness_score": 3, "coherence_score": 2, "total_score": 8, "verdict": "approve", "critique": "", "final_answer": "Product sales are the main revenue driver."}'
            ))
            mock_get_llm.return_value = mock_llm

            result = await critic_agent(state)

            assert result["verdict"] == "approve"
            assert result["final_answer"] is not None


@pytest.mark.asyncio
async def test_critic_requests_retry_for_hallucination():
    """Should request retry for ungrounded answers."""
    state = RAGState(
        query="What are revenue drivers?",
        chat_history=[],
        query_type="factual",
        sub_queries=["revenue drivers"],
        route="retriever",
        retrieved_chunks=[{"text": "Product sales", "chunk_id": "chunk_001"}],
        retrieval_metadata={},
        answer="Revenue increased by 500% due to new AI product.",  # Not in context!
        citations=["chunk_001"],
        confidence="high",
        reasoning_trace="hallucinated",
        verdict="",
        critique="",
        final_answer=None,
        retry_count=0
    )

    with patch("utils.prompts.CRITIC_PROMPT"):
        with patch("utils.providers.get_llm") as mock_get_llm:
            mock_llm = Mock()
            mock_llm.ainvoke = AsyncMock(return_value=Mock(
                content='{"faithfulness_score": 1, "completeness_score": 2, "coherence_score": 1, "total_score": 4, "verdict": "retry", "critique": "Answer contains claims not found in context", "final_answer": null}'
            ))
            mock_get_llm.return_value = mock_llm

            result = await critic_agent(state)

            assert result["verdict"] == "retry"
            assert result["final_answer"] is None
            assert len(result["critique"]) > 0


@pytest.mark.asyncio
async def test_critic_escalates_after_max_retries():
    """Should escalate after 2 failed retries."""
    state = RAGState(
        query="test",
        chat_history=[],
        query_type="factual",
        sub_queries=["test"],
        route="retriever",
        retrieved_chunks=[],
        retrieval_metadata={},
        answer="bad answer",
        citations=[],
        confidence="low",
        reasoning_trace="",
        verdict="",
        critique="",
        final_answer=None,
        retry_count=2  # Already retried twice
    )

    with patch("utils.prompts.CRITIC_PROMPT"):
        with patch("utils.providers.get_llm") as mock_get_llm:
            mock_llm = Mock()
            mock_llm.ainvoke = AsyncMock(return_value=Mock(
                content='{"faithfulness_score": 1, "completeness_score": 1, "coherence_score": 1, "total_score": 3, "verdict": "escalate", "critique": "Unable to generate faithful answer", "final_answer": null}'
            ))
            mock_get_llm.return_value = mock_llm

            result = await critic_agent(state)

            assert result["verdict"] == "escalate"
