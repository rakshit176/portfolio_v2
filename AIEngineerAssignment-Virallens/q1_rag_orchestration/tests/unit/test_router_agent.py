# tests/unit/test_router_agent.py
import pytest
from unittest.mock import Mock, patch, AsyncMock
from agents.router_agent import router_agent
from graph.state import RAGState


@pytest.mark.asyncio
async def test_router_classifies_factual_query():
    """Should classify factual queries correctly."""
    state = RAGState(
        query="What were the revenue drivers in Q3?",
        chat_history=[],
        query_type="",
        sub_queries=[],
        route="",
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

    with patch("agents.router_agent.get_llm") as mock_get_llm:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content='{"query_type": "factual", "sub_queries": ["revenue drivers"], "route": "retriever", "reasoning": "needs data"}'
        ))
        mock_get_llm.return_value = mock_llm

        result = await router_agent(state)

        assert result["query_type"] == "factual"
        assert result["sub_queries"] == ["revenue drivers"]
        assert result["route"] == "retriever"


@pytest.mark.asyncio
async def test_router_handles_conversational_query():
    """Should route conversational queries directly to reasoner."""
    state = RAGState(
        query="Hello, how are you?",
        chat_history=[],
        query_type="",
        sub_queries=[],
        route="",
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

    with patch("agents.router_agent.get_llm") as mock_get_llm:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content='{"query_type": "conversational", "sub_queries": [], "route": "reasoner", "reasoning": "greeting"}'
        ))
        mock_get_llm.return_value = mock_llm

        result = await router_agent(state)

        assert result["query_type"] == "conversational"
        assert result["route"] == "reasoner"
