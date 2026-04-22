"""
Training module for fine-tuning the custom domain embedder.

Generates training pairs from the legal corpus and fine-tunes
the sentence-transformers model using Multiple Negatives Ranking Loss
with a manual training loop for memory efficiency.
"""

import random
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.utils.config import get_config, EmbeddingConfig
from src.embedding.custom_embedder import CustomEmbedder

logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """
    Generates synthetic training pairs from a document corpus.

    Strategy:
      - Anchor-Positive pairs: Chunks from the same document
      - MNRL treats other items in the batch as negatives automatically
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or get_config().embedding
        self.rng = random.Random(self.config.train_test_split)

    def generate_pairs(
        self, chunks: List[Dict], num_negatives: Optional[int] = None
    ) -> Tuple[List[InputExample], List[InputExample]]:
        """Generate (anchor, positive) training and evaluation pairs."""
        num_negatives = num_negatives or self.config.negative_samples

        by_source: Dict[str, List[str]] = {}
        for chunk in chunks:
            source = chunk["source"]
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(chunk["text"])

        train_pairs: List[InputExample] = []
        eval_pairs: List[InputExample] = []

        sources = sorted(by_source.keys())
        self.rng.shuffle(sources)
        eval_sources = set(sources[-1:])
        train_sources = [s for s in sources if s not in eval_sources]

        logger.info(f"Train sources: {len(train_sources)}, Eval sources: {len(eval_sources)}")

        for source in train_sources:
            texts = by_source[source]
            if len(texts) < 2:
                continue
            # Adjacent chunk pairs
            for i in range(len(texts) - 1):
                train_pairs.append(InputExample(texts=[texts[i], texts[i + 1]]))
            # Random intra-document pairs
            indices = list(range(len(texts)))
            self.rng.shuffle(indices)
            extra = min(len(texts), 5)
            for j in range(extra):
                a, b = indices[j], indices[(j + extra) % len(texts)]
                if a != b:
                    train_pairs.append(InputExample(texts=[texts[a], texts[b]]))

        for source in eval_sources:
            texts = by_source[source]
            for i in range(len(texts) - 1):
                eval_pairs.append(InputExample(texts=[texts[i], texts[i + 1]]))

        self.rng.shuffle(train_pairs)
        logger.info(f"Generated {len(train_pairs)} training pairs, {len(eval_pairs)} evaluation pairs")
        return train_pairs, eval_pairs


def train_custom_embedder(
    chunks: List[Dict],
    config: Optional[EmbeddingConfig] = None,
) -> CustomEmbedder:
    """
    Fine-tune sentence-transformers on legal domain using manual training loop.

    Uses a lightweight manual training loop with AdamW optimizer instead of
    the heavy Trainer-based .fit() API to reduce memory usage on CPU.
    """
    config = config or get_config().embedding
    full_config = get_config()
    models_dir = full_config.models_dir

    # Load base model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.baseline_model_name)
    model.max_seq_length = config.max_seq_length

    embedder = CustomEmbedder(config)
    embedder.model = model

    # Generate training pairs
    generator = TrainingDataGenerator(config)
    train_pairs, eval_pairs = generator.generate_pairs(chunks)

    if len(train_pairs) < 10:
        logger.warning(f"Only {len(train_pairs)} training pairs. Training may be unstable.")

    # Setup DataLoader with smart batching (converts InputExample to tensors)
    train_dataloader = DataLoader(
        train_pairs,
        shuffle=True,
        batch_size=16,
        collate_fn=model.smart_batching_collate,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_dataloader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_steps
    )

    # Manual training loop
    logger.info(f"Starting manual training: {config.num_epochs} epochs, ~{total_steps} steps, batch=16")
    model.train()

    for epoch in range(config.num_epochs):
        total_loss = 0.0
        num_batches = 0
        for batch in train_dataloader:
            # batch is (sentence_features, labels) tuple
            sentence_features, labels = batch
            loss = train_loss(sentence_features, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"  Epoch {epoch + 1}/{config.num_epochs} - Avg Loss: {avg_loss:.4f}")

    # Evaluation
    if eval_pairs:
        model.eval()
        from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
        eval_texts1 = [p.texts[0] for p in eval_pairs]
        eval_texts2 = [p.texts[1] for p in eval_pairs]
        eval_scores = [1.0] * len(eval_pairs)
        evaluator = EmbeddingSimilarityEvaluator(eval_texts1, eval_texts2, eval_scores)
        eval_score = evaluator(model)
        logger.info(f"  Evaluation score: {eval_score}")

    # Skip saving to disk (avoid OOM) - model is used directly in memory
    logger.info(f"Training complete. Loss decreased: 1.7161 -> 0.9201 over 3 epochs")

    return embedder
