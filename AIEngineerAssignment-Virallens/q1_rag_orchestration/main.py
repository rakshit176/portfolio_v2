#!/usr/bin/env python3
"""
Q1 RAG Orchestration - Main Entry Point

Multi-agent RAG system with Router, Retriever, Reasoning, and Critic agents.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from graph import build_graph, RAGState
from rag.ingestor import ingest_documents
from utils.logger import get_logger

load_dotenv()

logger = get_logger("main")


SAMPLE_QUERIES = [
    "What are the main revenue drivers?",
    "Explain the company's product strategy.",
    "Who are the key executives mentioned in the documents?",
    "What risks are outlined in the annual report?",
    "Describe the financial performance for Q4.",
]


async def run_query(graph, query: str, chat_history: list = None):
    """Run a single query through the RAG pipeline."""
    logger.info("query_start", extra={"query": query[:50]})

    initial_state = RAGState(
        query=query,
        chat_history=chat_history or [],
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

    try:
        result = await graph.ainvoke(initial_state)

        logger.info("query_complete", extra={
            "query": query[:50],
            "final_answer": result.get("final_answer", "")[:100] if result.get("final_answer") else ""
        })

        return result
    except Exception as e:
        logger.error("query_error", extra={"query": query[:50], "error": str(e)})
        return None


async def main():
    """Main entry point."""
    print("=" * 60)
    print("Q1 RAG Orchestration System")
    print("=" * 60)

    # Check for ingestion mode
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        print("\n[1] Running document ingestion...")
        data_dir = sys.argv[2] if len(sys.argv) > 2 else "./data"
        await ingest_documents(data_dir=data_dir, recreate=True)
        print("Ingestion complete!")
        return

    # Build the graph
    print("\n[1] Building workflow graph...")
    graph = build_graph()
    print("Graph built successfully!")

    # Run sample queries
    print("\n[2] Running sample queries...")
    print("-" * 60)

    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)

        result = await run_query(graph, query)

        if result and result.get("final_answer"):
            print(f"Answer: {result['final_answer'][:200]}...")
            if result.get("citations"):
                print(f"Citations: {', '.join(result['citations'][:3])}")
            if result.get("confidence"):
                print(f"Confidence: {result['confidence']}")
        elif result:
            print(f"Processing... (verdict: {result.get('verdict', 'unknown')})")
        else:
            print("Error processing query")

    print("\n" + "=" * 60)
    print("All queries complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
