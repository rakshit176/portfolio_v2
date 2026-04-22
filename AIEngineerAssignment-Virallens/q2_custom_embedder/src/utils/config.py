"""
Configuration module for Q2 Custom Embedder.

Centralizes all hyperparameters, paths, and model settings.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_length: int = 50
    max_chunk_length: int = 1024
    remove_sections: List[str] = field(default_factory=lambda: [
        "TABLE OF CONTENTS", "APPENDIX", "EXHIBIT"
    ])


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    baseline_model_name: str = "all-MiniLM-L6-v2"
    custom_model_name: str = "legal-domain-embedder-v1"
    embedding_dim: int = 384  # MiniLM dimension
    max_seq_length: int = 256
    batch_size: int = 8
    num_epochs: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    train_test_split: float = 0.2
    # Contrastive learning
    temperature: float = 0.05
    negative_samples: int = 5


@dataclass
class ClusteringConfig:
    """Configuration for clustering algorithms."""
    # KMeans
    kmeans_k_range: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7, 8])
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    # HDBSCAN
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    # General
    random_state: int = 42


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    # Similarity search
    top_k_neighbors: int = 5
    # Clustering metrics
    k_range: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8])
    # t-SNE
    tsne_perplexity: float = 30.0
    tsne_learning_rate: float = 200.0
    tsne_n_iter: int = 1000
    tsne_random_state: int = 42
    random_state: int = 42


@dataclass
class Config:
    """Master configuration for Q2 Custom Embedder."""
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Paths
    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent.parent.parent

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data" / "raw"

    @property
    def output_dir(self) -> Path:
        return self.base_dir / "outputs"

    @property
    def embeddings_dir(self) -> Path:
        return self.output_dir / "embeddings"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"


# Global config singleton
_config: Config = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
