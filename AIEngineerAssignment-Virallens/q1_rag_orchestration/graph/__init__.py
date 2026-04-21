from .state import RAGState

def build_graph():
    """Lazy import to avoid circular imports."""
    from .workflow import _build_graph
    return _build_graph()

__all__ = ["RAGState", "build_graph"]
