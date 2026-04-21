# agents/critic_agent.py
import json
from typing import Dict, Any
from graph.state import RAGState
from utils.prompts import CRITIC_PROMPT
from utils.providers import get_llm
from utils.logger import get_logger

logger = get_logger("critic")


async def critic_agent(state: RAGState) -> Dict[str, Any]:
    """
    Evaluate answer faithfulness and completeness.

    Args:
        state: Current graph state with answer and citations

    Returns:
        Updated state with verdict, critique, and final_answer
    """
    query = state["query"]
    retrieved_chunks = state["retrieved_chunks"]
    answer = state["answer"]
    citations = state["citations"]
    confidence = state["confidence"]
    reasoning_trace = state["reasoning_trace"]
    retry_count = state["retry_count"]

    logger.info("critic_evaluation_start", extra={
        "query": query[:50],
        "retry_count": retry_count
    })

    # Format context
    context = _format_context_for_critic(retrieved_chunks)

    # Format prompt
    prompt = CRITIC_PROMPT.format(
        query=query,
        context=context,
        answer=answer,
        citations=", ".join(citations),
        confidence=confidence,
        reasoning_trace=reasoning_trace
    )

    # Get evaluation
    llm = get_llm(provider="groq", model="llama3-8b-8192")
    response = await llm.ainvoke(prompt)

    # Parse JSON response
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        logger.error("json_parse_error", extra={"response": response.content[:100]})
        # Default to escalate on parse error
        result = {
            "faithfulness_score": 1,
            "completeness_score": 1,
            "coherence_score": 1,
            "total_score": 3,
            "verdict": "escalate",
            "critique": "parse error in critic response",
            "final_answer": None
        }

    verdict = result["verdict"]
    total_score = result.get("total_score", 0)

    logger.info("critic_evaluation_complete", extra={
        "verdict": verdict,
        "total_score": total_score
    })

    # Set final_answer only if approved
    final_answer = None
    if verdict == "approve":
        final_answer = result.get("final_answer", answer)

    return {
        "verdict": verdict,
        "critique": result.get("critique", ""),
        "final_answer": final_answer
    }


def _format_context_for_critic(chunks: list) -> str:
    """Format chunks for critic evaluation."""
    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"[{chunk.get('chunk_id', 'unknown')}]\n{chunk['text']}\n"
        )
    return "\n".join(context_parts)
