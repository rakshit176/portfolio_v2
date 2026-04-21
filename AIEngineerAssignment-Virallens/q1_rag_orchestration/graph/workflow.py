# graph/workflow.py
from langgraph.graph import StateGraph, END
from graph.state import RAGState
from agents.router_agent import router_agent
from agents.retriever_agent import retriever_agent
from agents.reasoning_agent import reasoning_agent
from agents.critic_agent import critic_agent

def build_graph():
    """
    Build and compile the LangGraph workflow.

    Returns:
        Compiled StateGraph ready for invocation
    """
    # Create workflow
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("router", router_agent)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("reasoning", reasoning_agent)
    workflow.add_node("critic", critic_agent)

    # Set entry point
    workflow.set_entry_point("router")

    # Conditional edges from router
    def route_from_router(state: RAGState) -> str:
        return state["route"]

    workflow.add_conditional_edges(
        "router",
        route_from_router,
        {
            "retriever": "retriever",
            "reasoner": "reasoning",
            "clarify": END
        }
    )

    # Standard flow edges
    workflow.add_edge("retriever", "reasoning")
    workflow.add_edge("reasoning", "critic")

    # Conditional edges from critic
    def route_from_critic(state: RAGState) -> str:
        verdict = state["verdict"]

        if verdict == "retry" and state["retry_count"] < 2:
            return "retry"
        return verdict

    workflow.add_conditional_edges(
        "critic",
        route_from_critic,
        {
            "approve": END,
            "retry": "reasoning",
            "escalate": END
        }
    )

    # Compile and return
    return workflow.compile()
