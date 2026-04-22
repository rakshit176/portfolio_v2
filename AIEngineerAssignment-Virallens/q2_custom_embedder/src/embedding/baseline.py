"""
Baseline Embedder - Uses pre-trained all-MiniLM-L6-v2 for comparison.

This module wraps the sentence-transformers library to provide a baseline
embedding model for benchmarking against the custom domain embedder.
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import get_config, EmbeddingConfig

logger = logging.getLogger(__name__)


class BaselineEmbedder:
    """
    Baseline embedding model using all-MiniLM-L6-v2.

    This model is a general-purpose sentence embedding model trained on
    1B sentence pairs and serves as the comparison baseline for the
    custom legal domain embedder.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or get_config().embedding
        self.model = None
        self.model_name = self.config.baseline_model_name

    def load(self) -> "BaselineEmbedder":
        """Load the baseline model."""
        logger.info(f"Loading baseline model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        # Truncate to our max_seq_length for fair comparison
        self.model.max_seq_length = self.config.max_seq_length
        logger.info(
            f"Baseline model loaded. Embedding dim: {self.model.get_sentence_embedding_dimension()}"
        )
        return self

    def embed(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Override batch size.

        Returns:
            numpy array of shape (n_texts, embedding_dim).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        batch_size = batch_size or self.config.batch_size
        logger.info(f"Embedding {len(texts)} texts with baseline model (batch={batch_size})")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        logger.info(f"Generated embeddings: shape={embeddings.shape}")
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension."""
        if self.model is None:
            return self.config.embedding_dim
        return self.model.get_sentence_embedding_dimension()
