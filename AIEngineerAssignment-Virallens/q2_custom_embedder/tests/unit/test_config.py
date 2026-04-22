"""
Unit tests for src.utils.config – Config dataclass, singleton, and path properties.
"""

from pathlib import Path
import pytest

from src.utils.config import (
    Config,
    PreprocessingConfig,
    EmbeddingConfig,
    ClusteringConfig,
    EvaluationConfig,
    get_config,
)


# ---------------------------------------------------------------------------
# Default config values
# ---------------------------------------------------------------------------


class TestDefaultConfigValues:

    def test_preprocessing_defaults(self):
        config = PreprocessingConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 64
        assert config.min_chunk_length == 50
        assert config.max_chunk_length == 1024
        assert isinstance(config.remove_sections, list)
        assert "TABLE OF CONTENTS" in config.remove_sections

    def test_embedding_defaults(self):
        config = EmbeddingConfig()
        assert config.baseline_model_name == "all-MiniLM-L6-v2"
        assert config.custom_model_name == "legal-domain-embedder-v1"
        assert config.embedding_dim == 384
        assert config.max_seq_length == 256
        assert config.batch_size == 8
        assert config.num_epochs == 1
        assert config.learning_rate == 2e-5
        assert config.temperature == 0.05
        assert config.negative_samples == 5

    def test_clustering_defaults(self):
        config = ClusteringConfig()
        assert config.kmeans_k_range == [3, 4, 5, 6, 7, 8]
        assert config.kmeans_n_init == 10
        assert config.kmeans_max_iter == 300
        assert config.hdbscan_min_cluster_size == 5
        assert config.hdbscan_min_samples == 3
        assert config.random_state == 42

    def test_evaluation_defaults(self):
        config = EvaluationConfig()
        assert config.top_k_neighbors == 5
        assert config.k_range == [2, 3, 4, 5, 6, 7, 8]
        assert config.tsne_perplexity == 30.0
        assert config.tsne_learning_rate == 200.0
        assert config.tsne_n_iter == 1000
        assert config.random_state == 42

    def test_master_config_composes_subconfigs(self):
        config = Config()
        assert isinstance(config.preprocessing, PreprocessingConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.clustering, ClusteringConfig)
        assert isinstance(config.evaluation, EvaluationConfig)


# ---------------------------------------------------------------------------
# Path properties
# ---------------------------------------------------------------------------


class TestPathProperties:

    def test_base_dir_is_project_root(self):
        config = Config()
        # base_dir is defined as Path(__file__).parent.parent.parent
        # which is q2_custom_embedder/
        expected = Path(__file__).resolve().parent.parent.parent
        assert config.base_dir == expected

    def test_data_dir(self):
        config = Config()
        assert config.data_dir == config.base_dir / "data" / "raw"

    def test_output_dir(self):
        config = Config()
        assert config.output_dir == config.base_dir / "outputs"

    def test_embeddings_dir(self):
        config = Config()
        assert config.embeddings_dir == config.output_dir / "embeddings"

    def test_plots_dir(self):
        config = Config()
        assert config.plots_dir == config.output_dir / "plots"

    def test_models_dir(self):
        config = Config()
        assert config.models_dir == config.output_dir / "models"

    def test_results_dir(self):
        config = Config()
        assert config.results_dir == config.output_dir / "results"

    def test_all_paths_are_path_objects(self):
        config = Config()
        for prop in ("base_dir", "data_dir", "output_dir", "embeddings_dir",
                      "plots_dir", "models_dir", "results_dir"):
            assert isinstance(getattr(config, prop), Path), f"{prop} is not a Path"


# ---------------------------------------------------------------------------
# Config singleton
# ---------------------------------------------------------------------------


class TestConfigSingleton:

    def test_get_config_returns_config_instance(self):
        """get_config() returns a Config object."""
        cfg = get_config()
        assert isinstance(cfg, Config)

    def test_get_config_returns_same_instance(self):
        """Repeated calls return the same object (singleton)."""
        # Reset singleton to ensure clean state for this test
        import src.utils.config as config_mod
        original = config_mod._config
        try:
            config_mod._config = None
            a = get_config()
            b = get_config()
            assert a is b
        finally:
            config_mod._config = original

    def test_get_config_lazy_initialization(self):
        """Config is only created on first call, not at import time."""
        import src.utils.config as config_mod
        original = config_mod._config
        try:
            config_mod._config = None
            cfg = get_config()
            assert cfg is not None
            assert isinstance(cfg, Config)
        finally:
            config_mod._config = original

    def test_custom_config_overrides(self):
        """Custom config values are properly stored."""
        config = Config(
            preprocessing=PreprocessingConfig(chunk_size=1024, chunk_overlap=128),
        )
        assert config.preprocessing.chunk_size == 1024
        assert config.preprocessing.chunk_overlap == 128
        # Other defaults are preserved
        assert config.preprocessing.min_chunk_length == 50
