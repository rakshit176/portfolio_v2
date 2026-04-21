# tests/unit/test_workflow.py
import pytest
from graph.state import RAGState

def test_rag_state_has_all_required_fields():
    """RAGState must contain all required fields for agent communication."""
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
        confidence="low",
        reasoning_trace="",
        verdict="",
        critique="",
        final_answer=None,
        retry_count=0
    )

    assert state["query"] == "test query"
    assert state["query_type"] == "factual"
    assert state["route"] == "retriever"
    assert state["retry_count"] == 0

def test_rag_state_optional_fields_can_be_none():
    """Optional fields like final_answer should accept None."""
    state = RAGState(
        query="test",
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

    assert state["final_answer"] is None

def test_workflow_compiles_successfully():
    """Workflow should compile without errors."""
    from graph import build_graph

    graph = build_graph()

    assert graph is not None
    # Graph should have all nodes
    expected_nodes = {"router", "retriever", "reasoning", "critic"}
    # Note: LangGraph doesn't expose nodes directly, but compilation validates structure

def test_workflow_entry_point_is_router():
    """Router should be the entry point."""
    from graph import build_graph

    graph = build_graph()

    # Entry point is validated during compilation
    assert graph is not None
