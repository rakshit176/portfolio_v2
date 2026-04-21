# tests/unit/test_reasoning_agent.py
import pytest
from unittest.mock import Mock, patch, AsyncMock
from agents.reasoning_agent import reasoning_agent
from graph.state import RAGState

@pytest.mark.asyncio
async def test_reasoning_generates_answer_with_citations():
    """Should generate answer with source citations."""
    state = RAGState(
        query="What are the revenue drivers?",
        chat_history=[],
        query_type="factual",
        sub_queries=["revenue drivers"],
        route="retriever",
        retrieved_chunks=[
            {"text": "Product sales and services are key drivers", "chunk_id": "chunk_001"}
        ],
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

    with patch("agents.reasoning_agent.get_llm") as mock_get_llm:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content='{"answer": "Revenue drivers are product sales and services", "citations": ["chunk_001"], "confidence": "high", "reasoning_trace": "found in context"}'
        ))
        mock_get_llm.return_value = mock_llm

        result = await reasoning_agent(state)

        assert result["answer"] != ""
        assert "chunk_001" in result["citations"]
        assert result["confidence"] in ["low", "medium", "high"]

@pytest.mark.asyncio
async def test_reasoning_uses_fallback_on_provider_failure():
    """Should fallback to Gemini when Groq fails."""
    state = RAGState(
        query="test",
        chat_history=[],
        query_type="factual",
        sub_queries=["test"],
        route="retriever",
        retrieved_chunks=[{"text": "context", "chunk_id": "chunk_001"}],
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

    # Track which providers were called
    providers_called = []

    # Create a simple class to mock the LLM
    class MockLLM:
        def __init__(self, provider):
            self.provider = provider

        async def ainvoke(self, prompt, *args, **kwargs):
            providers_called.append(self.provider)
            if self.provider == "groq":
                raise Exception("Groq rate limited")
            elif self.provider == "gemini":
                return Mock(content='{"answer": "fallback answer", "citations": ["chunk_001"], "confidence": "medium", "reasoning_trace": "used fallback"}')
            else:
                raise Exception("Provider not available")

    def mock_get_llm(provider, **kwargs):
        return MockLLM(provider)

    with patch("agents.reasoning_agent.get_llm", side_effect=mock_get_llm):
        result = await reasoning_agent(state)

        # Should have tried groq first, then gemini
        assert "groq" in providers_called
        assert "gemini" in providers_called
        assert providers_called.index("groq") < providers_called.index("gemini")
        assert result["answer"] == "fallback answer"
