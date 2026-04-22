"""
Unit tests for src.clustering.clusterer – DocumentClusterer.

KMeans tests use real sklearn (fast, deterministic with random_state).
HDBSCAN tests mock the hdbscan import so the optional dependency is not required.
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.clustering.clusterer import DocumentClusterer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clustered_embeddings():
    """
    60 embeddings in 3 well-separated clusters (20 points each).
    Guarantees KMeans will find meaningful structure.
    """
    rng = np.random.RandomState(42)
    centers = np.array([
        [+5.0] * 10,
        [-5.0] * 10,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, +8.0],
    ])
    embeddings = np.vstack([
        center + rng.randn(20, 10) * 0.3
        for center in centers
    ])
    return embeddings.astype(np.float32)


@pytest.fixture()
def sample_chunks_60():
    """60 chunk dicts (matching the clustered_embeddings fixture)."""
    return [
        {"source": f"doc_{i // 20}", "text": f"Chunk text {i}. " * 15,
         "chunk_id": f"c{i}", "chunk_index": i}
        for i in range(60)
    ]


# ---------------------------------------------------------------------------
# KMeans clustering
# ---------------------------------------------------------------------------


class TestKMeansClustering:

    def test_kmeans_clustering_produces_correct_labels(self, clustered_embeddings):
        """Labels should be a numpy array with shape (n_samples,)."""
        clusterer = DocumentClusterer(random_state=42)
        results = clusterer.kmeans_clustering(clustered_embeddings, k_range=[3])

        assert 3 in results
        labels = results[3]["labels"]
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (60,)
        # All labels should be in {0, 1, 2}
        assert set(labels).issubset({0, 1, 2})

    def test_kmeans_clustering_computes_metrics(self, clustered_embeddings):
        """Result dict must contain silhouette, davies_bouldin, calinski_harabasz."""
        clusterer = DocumentClusterer(random_state=42)
        results = clusterer.kmeans_clustering(clustered_embeddings, k_range=[3])

        result = results[3]
        assert "silhouette" in result
        assert "davies_bouldin" in result
        assert "calinski_harabasz" in result
        # Well-separated clusters → positive silhouette
        assert result["silhouette"] > 0.0
        assert result["silhouette"] <= 1.0

    def test_kmeans_clustering_tracks_best_k(self, clustered_embeddings):
        """best_kmeans_k and best_kmeans_score are updated."""
        clusterer = DocumentClusterer(random_state=42)
        results = clusterer.kmeans_clustering(clustered_embeddings, k_range=[2, 3, 4])

        assert clusterer.best_kmeans_k is not None
        assert clusterer.best_kmeans_score > 0
        assert clusterer.best_kmeans_k in results

    def test_kmeans_clustering_multiple_k(self, clustered_embeddings):
        """Running with multiple k values returns results for each."""
        clusterer = DocumentClusterer(random_state=42)
        k_range = [2, 3, 4, 5]
        results = clusterer.kmeans_clustering(clustered_embeddings, k_range=k_range)

        for k in k_range:
            assert k in results
            assert results[k]["n_clusters"] == k
            assert results[k]["algorithm"] == "kmeans"

    def test_kmeans_clustering_invalid_k(self, clustered_embeddings):
        """k >= n_samples should be silently skipped."""
        clusterer = DocumentClusterer(random_state=42)
        results = clusterer.kmeans_clustering(clustered_embeddings, k_range=[100])
        assert 100 not in results

    def test_kmeans_clustering_single_sample_raises(self):
        """A single sample cannot be clustered (no valid k)."""
        embeddings = np.random.randn(1, 10).astype(np.float32)
        clusterer = DocumentClusterer(random_state=42)
        results = clusterer.kmeans_clustering(embeddings, k_range=[2])
        assert len(results) == 0

    def test_kmeans_inertia_stored(self, clustered_embeddings):
        """Each result should store KMeans inertia."""
        clusterer = DocumentClusterer(random_state=42)
        results = clusterer.kmeans_clustering(clustered_embeddings, k_range=[3])
        assert "inertia" in results[3]
        assert isinstance(results[3]["inertia"], float)
        assert results[3]["inertia"] > 0


# ---------------------------------------------------------------------------
# HDBSCAN clustering (mocked import)
# ---------------------------------------------------------------------------


class TestHDBSCANClustering:

    def test_hdbscan_clustering_returns_result_on_success(self):
        """When hdbscan is available, returns a dict with labels."""
        mock_hdbscan_module = MagicMock()
        mock_clusterer_instance = MagicMock()
        mock_clusterer_instance.fit_predict.return_value = np.array([0, 0, 1, 1, -1, 1])
        mock_clusterer_instance.probabilities_ = np.array([0.9, 0.8, 0.85, 0.7, 0.3, 0.9])
        mock_clusterer_instance.cluster_persistence_ = np.array([0.8, 0.6])
        mock_hdbscan_module.HDBSCAN.return_value = mock_clusterer_instance

        rng = np.random.RandomState(0)
        embeddings = rng.randn(6, 10).astype(np.float32)

        clusterer = DocumentClusterer(random_state=42)
        with patch.dict("sys.modules", {"hdbscan": mock_hdbscan_module}):
            result = clusterer.hdbscan_clustering(embeddings, min_cluster_size=2)

        assert result is not None
        assert "labels" in result
        assert "n_clusters" in result
        assert result["n_clusters"] == 2  # clusters 0 and 1 (noise -1 excluded)
        assert result["n_noise"] == 1

    def test_hdbscan_clustering_handles_missing_package(self):
        """If hdbscan is not installed, returns None gracefully."""
        clusterer = DocumentClusterer(random_state=42)
        with patch.dict("sys.modules", {"hdbscan": None}, clear=False):
            # Force reimport behavior: the try/except should catch ImportError
            # We simulate this by making the import raise ImportError
            with patch("builtins.__import__", side_effect=ImportError("No hdbscan")):
                # Actually the module-level try/except handles this inside the method.
                # Let's patch at the method level instead.
                pass

        # More reliable approach: patch the import inside the method
        clusterer = DocumentClusterer(random_state=42)
        embeddings = np.random.randn(10, 10).astype(np.float32)

        with patch("builtins.__import__", side_effect=ImportError("No module hdbscan")):
            # The method does `import hdbscan` inside a try/except
            result = clusterer.hdbscan_clustering(embeddings)
            assert result is None

    def test_hdbscan_clustering_small_dataset(self):
        """Very small dataset with min_cluster_size > n_samples still works."""
        mock_hdbscan_module = MagicMock()
        mock_instance = MagicMock()
        # All points labeled as noise
        mock_instance.fit_predict.return_value = np.array([-1, -1, -1])
        mock_instance.probabilities_ = np.array([0.1, 0.1, 0.1])
        mock_instance.cluster_persistence_ = np.array([])
        mock_hdbscan_module.HDBSCAN.return_value = mock_instance

        embeddings = np.random.randn(3, 10).astype(np.float32)
        clusterer = DocumentClusterer(random_state=42)

        with patch.dict("sys.modules", {"hdbscan": mock_hdbscan_module}):
            result = clusterer.hdbscan_clustering(embeddings, min_cluster_size=5)

        assert result["n_clusters"] == 0
        assert result["n_noise"] == 3
        # No silhouette computed when all noise
        assert "silhouette" not in result

    def test_hdbscan_noise_ratio(self):
        """noise_ratio = n_noise / total_samples."""
        mock_hdbscan_module = MagicMock()
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = np.array([0, 0, -1, 1, 1, -1, -1])
        mock_instance.probabilities_ = np.ones(7) * 0.8
        mock_instance.cluster_persistence_ = np.array([0.7, 0.6])
        mock_hdbscan_module.HDBSCAN.return_value = mock_instance

        embeddings = np.random.randn(7, 10).astype(np.float32)
        clusterer = DocumentClusterer(random_state=42)

        with patch.dict("sys.modules", {"hdbscan": mock_hdbscan_module}):
            result = clusterer.hdbscan_clustering(embeddings, min_cluster_size=2)

        assert result["noise_ratio"] == pytest.approx(3 / 7)


# ---------------------------------------------------------------------------
# get_optimal_labels
# ---------------------------------------------------------------------------


class TestGetOptimalLabels:

    def test_returns_best_kmeans_labels(self, clustered_embeddings):
        """After kmeans_clustering, get_optimal_labels returns best-k labels."""
        clusterer = DocumentClusterer(random_state=42)
        clusterer.kmeans_clustering(clustered_embeddings, k_range=[3])

        labels = clusterer.get_optimal_labels(clustered_embeddings, method="kmeans")
        assert labels.shape == (60,)
        assert clusterer.best_kmeans_k == 3

    def test_returns_hdbscan_labels(self):
        """When hdbscan_result is set, method='hdbscan' returns its labels."""
        mock_hdbscan_module = MagicMock()
        mock_instance = MagicMock()
        expected = np.array([0, 0, 1, 1, 1, 0])
        mock_instance.fit_predict.return_value = expected
        mock_instance.probabilities_ = np.ones(6) * 0.9
        mock_instance.cluster_persistence_ = np.array([0.8, 0.7])
        mock_hdbscan_module.HDBSCAN.return_value = mock_instance

        embeddings = np.random.randn(6, 10).astype(np.float32)
        clusterer = DocumentClusterer(random_state=42)

        with patch.dict("sys.modules", {"hdbscan": mock_hdbscan_module}):
            clusterer.hdbscan_clustering(embeddings, min_cluster_size=2)

        labels = clusterer.get_optimal_labels(embeddings, method="hdbscan")
        np.testing.assert_array_equal(labels, expected)

    def test_raises_when_no_results(self):
        """Calling get_optimal_labels before any clustering raises RuntimeError."""
        clusterer = DocumentClusterer(random_state=42)
        with pytest.raises(RuntimeError, match="No clustering results available"):
            clusterer.get_optimal_labels(np.zeros((5, 10)))


# ---------------------------------------------------------------------------
# get_cluster_summary
# ---------------------------------------------------------------------------


class TestGetClusterSummary:

    def test_produces_correct_structure(self, clustered_embeddings, sample_chunks_60):
        """Summary dict has the expected keys for each cluster."""
        clusterer = DocumentClusterer(random_state=42)
        results = clusterer.kmeans_clustering(clustered_embeddings, k_range=[3])
        labels = results[3]["labels"]

        summary = clusterer.get_cluster_summary(sample_chunks_60, labels)

        assert isinstance(summary, dict)
        assert len(summary) == 3  # 3 clusters

        for cid, info in summary.items():
            assert "size" in info
            assert "sources" in info
            assert "sample_texts" in info
            assert isinstance(info["size"], int)
            assert isinstance(info["sources"], dict)
            assert isinstance(info["sample_texts"], list)
            assert len(info["sample_texts"]) <= 3

    def test_sizes_sum_to_total(self, clustered_embeddings, sample_chunks_60):
        """Sum of all cluster sizes equals total number of chunks."""
        clusterer = DocumentClusterer(random_state=42)
        results = clusterer.kmeans_clustering(clustered_embeddings, k_range=[3])
        labels = results[3]["labels"]
        summary = clusterer.get_cluster_summary(sample_chunks_60, labels)

        total = sum(info["size"] for info in summary.values())
        assert total == 60

    def test_sources_tracked_per_cluster(self, clustered_embeddings, sample_chunks_60):
        """Each cluster summary tracks which documents contributed chunks."""
        clusterer = DocumentClusterer(random_state=42)
        results = clusterer.kmeans_clustering(clustered_embeddings, k_range=[3])
        labels = results[3]["labels"]
        summary = clusterer.get_cluster_summary(sample_chunks_60, labels)

        # At least one cluster should reference a "doc_X" source
        all_sources = set()
        for info in summary.values():
            all_sources.update(info["sources"].keys())
        assert all_sources.issubset({"doc_0", "doc_1", "doc_2"})

    def test_sample_texts_truncated(self, clustered_embeddings, sample_chunks_60):
        """sample_texts entries are ≤ 153 chars (150 + '...')."""
        clusterer = DocumentClusterer(random_state=42)
        results = clusterer.kmeans_clustering(clustered_embeddings, k_range=[3])
        labels = results[3]["labels"]
        summary = clusterer.get_cluster_summary(sample_chunks_60, labels)

        for info in summary.values():
            for st in info["sample_texts"]:
                assert len(st) <= 153
