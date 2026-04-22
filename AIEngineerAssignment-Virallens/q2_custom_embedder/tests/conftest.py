"""
Shared test configuration and fixtures for Q2 Custom Embedder tests.

Ensures the project root is on sys.path so `src.*` imports resolve
regardless of the working directory from which pytest is invoked.
"""

import sys
from pathlib import Path

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Path setup – add project root so `from src.*` works in every test file
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Config fixture – provide a fresh Config per test (no singleton leakage)
# ---------------------------------------------------------------------------
@pytest.fixture()
def fresh_config():
    """Return a brand-new Config instance, resetting the global singleton."""
    from src.utils.config import Config
    return Config()


# ---------------------------------------------------------------------------
# Synthetic embeddings fixture – deterministic random embeddings for tests
# ---------------------------------------------------------------------------
@pytest.fixture()
def sample_embeddings():
    """Return (n=20, dim=384) deterministic embeddings."""
    rng = np.random.RandomState(42)
    return rng.randn(20, 384).astype(np.float32)


@pytest.fixture()
def sample_embeddings_two_sources():
    """
    Return embeddings split across two conceptual "sources"
    for within/between similarity tests.

    Returns (embeddings, chunks) where chunks[0:10].source == "doc_a"
    and chunks[10:20].source == "doc_b".
    """
    rng = np.random.RandomState(42)
    # Make doc_a vectors cluster together and doc_b vectors cluster together
    center_a = rng.randn(1, 384).astype(np.float32)
    center_b = rng.randn(1, 384).astype(np.float32)
    # Ensure centers are different
    center_b += np.array([3.0] * 384, dtype=np.float32).reshape(1, -1)

    emb_a = center_a + rng.randn(10, 384).astype(np.float32) * 0.3
    emb_b = center_b + rng.randn(10, 384).astype(np.float32) * 0.3
    embeddings = np.vstack([emb_a, emb_b])

    chunks = [
        {"source": "doc_a", "text": f"chunk a-{i} " * 30, "chunk_id": f"a{i}"}
        for i in range(10)
    ] + [
        {"source": "doc_b", "text": f"chunk b-{i} " * 30, "chunk_id": f"b{i}"}
        for i in range(10)
    ]
    return embeddings, chunks


@pytest.fixture()
def two_embedding_spaces():
    """
    Return two embedding arrays (baseline, custom) for correlation /
    neighbor-agreement tests.  They share structure but are perturbed.
    """
    rng = np.random.RandomState(42)
    base = rng.randn(20, 384).astype(np.float32)
    custom = base + rng.randn(20, 384).astype(np.float32) * 0.5
    return base, custom


@pytest.fixture()
def sample_chunks():
    """Return a list of chunk dicts suitable for clustering/evaluation tests."""
    return [
        {"source": f"doc_{i % 3}", "text": f"This is chunk number {i}. " * 20,
         "chunk_id": f"chunk_{i}", "chunk_index": i, "char_count": 250}
        for i in range(15)
    ]
