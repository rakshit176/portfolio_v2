# agents/router_agent.py
import json
from typing import Dict, Any
from graph.state import RAGState
from utils.prompts import ROUTER_PROMPT
from utils.providers import get_llm
from utils.logger import get_logger

logger = get_logger("router")

async def router_agent(state: RAGState) -> Dict[str, Any]:
    """
    Classify query and determine routing.

    Args:
        state: Current graph state

    Returns:
        Updated state with query_type, sub_queries, and route
    """
    query = state["query"]
    chat_history = state["chat_history"]

    logger.info("routing_query", extra={"query": query[:50]})

    # Format prompt
    prompt = ROUTER_PROMPT.format(
        query=query,
        chat_history=chat_history
    )

    # Get LLM response
    llm = get_llm(provider="groq", model="llama-3.1-8b-instant")
    response = await llm.ainvoke(prompt)

    # Parse JSON response
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        logger.error("json_parse_error", extra={"response": response.content[:100]})
        # Fallback to factual for safety
        result = {
            "query_type": "factual",
            "sub_queries": [query],
            "route": "retriever",
            "reasoning": "fallback due to parse error"
        }

    logger.info("query_classified", extra={
        "query_type": result["query_type"],
        "route": result["route"]
    })

    return {
        "query_type": result["query_type"],
        "sub_queries": result.get("sub_queries", [query]),
        "route": result["route"]
    }
