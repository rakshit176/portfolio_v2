# agents/reasoning_agent.py
import json
from typing import Dict, Any
from graph.state import RAGState
from utils.prompts import RAG_PROMPT
from utils.providers import get_llm
from utils.logger import get_logger

logger = get_logger("reasoning")

async def reasoning_agent(state: RAGState) -> Dict[str, Any]:
    """
    Generate grounded answer using retrieved context.

    Args:
        state: Current graph state with retrieved_chunks

    Returns:
        Updated state with answer, citations, confidence, reasoning_trace
    """
    query = state["query"]
    sub_queries = state["sub_queries"]
    retrieved_chunks = state["retrieved_chunks"]
    chat_history = state["chat_history"]

    logger.info("reasoning_start", extra={
        "query": query[:50],
        "num_chunks": len(retrieved_chunks)
    })

    # Format context from chunks
    context = _format_context(retrieved_chunks)

    # Format prompt
    prompt = RAG_PROMPT.format(
        context=context,
        query=query,
        sub_queries=", ".join(sub_queries)
    )

    # Try providers in fallback chain
    providers = ["groq", "gemini", "openrouter", "ollama"]
    response = None

    for provider in providers:
        try:
            llm = get_llm(provider=provider)
            response = await llm.ainvoke(prompt)
            logger.info("llm_success", extra={"provider": provider})
            break
        except Exception as e:
            logger.warning("llm_failed", extra={"provider": provider, "error": str(e)})
            continue

    if response is None:
        logger.error("all_providers_failed")
        return {
            "answer": "I apologize, but I'm unable to generate a response at this time. Please try again later.",
            "citations": [],
            "confidence": "low",
            "reasoning_trace": "all providers failed"
        }

    # Parse JSON response
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        logger.error("json_parse_error", extra={"response": response.content[:100]})
        result = {
            "answer": response.content,
            "citations": [c["chunk_id"] for c in retrieved_chunks],
            "confidence": "low",
            "reasoning_trace": "parse error, using raw response"
        }

    logger.info("reasoning_complete", extra={
        "confidence": result["confidence"],
        "num_citations": len(result.get("citations", []))
    })

    return {
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "confidence": result.get("confidence", "low"),
        "reasoning_trace": result.get("reasoning_trace", "")
    }

def _format_context(chunks: list) -> str:
    """Format retrieved chunks as context string."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[{chunk.get('chunk_id', f'chunk_{i:03d}')}]\n"
            f"{chunk['text']}\n"
        )
    return "\n".join(context_parts)
