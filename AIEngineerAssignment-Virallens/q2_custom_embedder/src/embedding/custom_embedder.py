"""
Custom Domain Embedder - Fine-tuned sentence-transformers for legal documents.

Fine-tunes all-MiniLM-L6-v2 on the extracted legal corpus using
Multiple Negatives Ranking Loss (MNRL) with hard negative mining.
Produces a domain-specific embedding model optimized for legal text
similarity and retrieval tasks.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    models,
)
from torch.utils.data import DataLoader

from src.utils.config import get_config, EmbeddingConfig

logger = logging.getLogger(__name__)


class CustomEmbedder:
    """
    Domain-specific embedding model fine-tuned on legal documents.

    Training approach:
      1. Start from all-MiniLM-L6-v2 (general-purpose baseline)
      2. Generate synthetic training pairs from the legal corpus
         using (anchor, positive, negative) triples
      3. Fine-tune with Multiple Negatives Ranking Loss (MNRL)
      4. Evaluate on held-out legal similarity pairs
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or get_config().embedding
        self.model = None
        self.model_name = self.config.custom_model_name

    def load(self) -> "CustomEmbedder":
        """Load the fine-tuned model from disk, or the base model if not yet trained."""
        from src.utils.config import get_config
        full_config = get_config()
        model_path = full_config.models_dir / self.model_name

        if model_path.exists():
            logger.info(f"Loading fine-tuned model from: {model_path}")
            self.model = SentenceTransformer(str(model_path))
        else:
            logger.info("No fine-tuned model found. Loading base model.")
            self.model = SentenceTransformer(self.config.baseline_model_name)
            logger.warning(
                "Using BASE model (not fine-tuned). Run train() first for domain adaptation."
            )

        self.model.max_seq_length = self.config.max_seq_length
        logger.info(
            f"Model loaded. Embedding dim: {self.model.get_sentence_embedding_dimension()}"
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
        logger.info(f"Embedding {len(texts)} texts with custom model (batch={batch_size})")

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

    def save(self, path: Optional[Path] = None) -> Path:
        """Save the fine-tuned model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save. Train or load first.")

        save_path = path or (self.config.models_dir / self.model_name)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        logger.info(f"Model saved to: {save_path}")
        return save_path
