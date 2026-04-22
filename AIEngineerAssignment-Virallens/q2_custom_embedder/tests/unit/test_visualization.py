"""
Unit tests for src.visualization.plotter - EmbeddingVisualizer.

All heavy computations (t-SNE, KMeans, silhouette_score) and matplotlib
rendering are mocked so tests run fast without GPU or display dependencies.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock, call
from types import SimpleNamespace

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_embeddings_20():
    """20 deterministic 384-dim embeddings."""
    rng = np.random.RandomState(42)
    return rng.randn(20, 384).astype(np.float32)


@pytest.fixture()
def sample_chunks_20():
    """20 chunk dicts across 4 source documents."""
    return [
        {"source": f"doc_{i % 4}", "text": f"Chunk text {i}. " * 20,
         "chunk_id": f"c{i}", "chunk_index": i}
        for i in range(20)
    ]


@pytest.fixture()
def sample_kmeans_results():
    """Minimal KMeans results dict for k=4."""
    return {
        4: {
            "labels": np.array([0, 0, 1, 1, 2, 2, 3, 3,
                                0, 0, 1, 1, 2, 2, 3, 3,
                                0, 0, 1, 1]),
            "silhouette": 0.35,
            "n_clusters": 4,
        }
    }


@pytest.fixture()
def sample_eval_results():
    """Minimal evaluation results dict."""
    return {
        "baseline_similarity": {
            "within_mean": 0.36,
            "between_mean": 0.29,
            "separation_ratio": 1.24,
        },
        "custom_similarity": {
            "within_mean": 0.42,
            "between_mean": 0.36,
            "separation_ratio": 1.17,
        },
        "best_custom_k": 4,
    }


# ---------------------------------------------------------------------------
# _save_fig
# ---------------------------------------------------------------------------


class TestSaveFig:

    def test_save_fig_creates_file(self, tmp_path):
        """_save_fig writes a file to the output directory."""
        from src.visualization.plotter import EmbeddingVisualizer
        viz = EmbeddingVisualizer(str(tmp_path))

        mock_fig = MagicMock()
        with patch("src.visualization.plotter.plt") as mock_plt:
            viz._save_fig(mock_fig, "test_output.png")

        expected_path = str(tmp_path / "test_output.png")
        mock_fig.savefig.assert_called_once()
        assert mock_fig.savefig.call_args[0][0] == expected_path
        mock_plt.close.assert_called_once_with(mock_fig)

    def test_save_fig_tracks_generated_plots(self, tmp_path):
        """_save_fig appends the path to plots_generated."""
        from src.visualization.plotter import EmbeddingVisualizer
        viz = EmbeddingVisualizer(str(tmp_path))

        mock_fig = MagicMock()
        with patch("src.visualization.plotter.plt"):
            viz._save_fig(mock_fig, "plot1.png")
            viz._save_fig(mock_fig, "plot2.png")

        assert len(viz.plots_generated) == 2


# ---------------------------------------------------------------------------
# plot_tsne_comparison
# ---------------------------------------------------------------------------


class TestPlotTSNEComparison:

    @patch("src.visualization.plotter.TSNE")
    @patch("src.visualization.plotter.plt")
    def test_plot_tsne_generates_file(self, mock_plt, mock_tsne_cls, tmp_path,
                                       sample_embeddings_20, sample_chunks_20):
        """plot_tsne_comparison calls _save_fig to produce tsne_comparison.png."""
        mock_tsne = MagicMock()
        mock_tsne.fit_transform.return_value = np.random.randn(20, 2)
        mock_tsne_cls.return_value = mock_tsne

        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_plt.Figure.return_value = mock_fig

        from src.visualization.plotter import EmbeddingVisualizer
        viz = EmbeddingVisualizer(str(tmp_path))

        with patch.object(viz, "_save_fig") as mock_save:
            viz.plot_tsne_comparison(sample_embeddings_20, sample_embeddings_20, sample_chunks_20)
            mock_save.assert_called_once()

        # Should call TSNE twice (once for baseline, once for custom)
        assert mock_tsne_cls.call_count == 2


# ---------------------------------------------------------------------------
# plot_silhouette_comparison
# ---------------------------------------------------------------------------


class TestPlotSilhouetteComparison:

    @patch("src.visualization.plotter.silhouette_score", return_value=0.3)
    @patch("src.visualization.plotter.plt")
    def test_plot_silhouette_generates_file(self, mock_plt, mock_ss, tmp_path,
                                             sample_embeddings_20):
        """plot_silhouette_comparison produces silhouette_comparison.png."""
        mock_km = MagicMock()
        mock_km.fit_predict.return_value = np.random.randint(0, 3, 20)

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        from src.visualization.plotter import EmbeddingVisualizer
        viz = EmbeddingVisualizer(str(tmp_path))

        with patch("src.visualization.plotter.KMeans", return_value=mock_km), \
             patch.object(viz, "_save_fig") as mock_save:
            viz.plot_silhouette_comparison(sample_embeddings_20, sample_embeddings_20)
            mock_save.assert_called_once()
            assert "silhouette" in mock_save.call_args[0][1]


# ---------------------------------------------------------------------------
# plot_within_between_similarity
# ---------------------------------------------------------------------------


class TestPlotWithinBetweenSimilarity:

    @patch("src.visualization.plotter.plt")
    def test_plot_similarity_generates_file(self, mock_plt, tmp_path):
        """plot_within_between_similarity creates similarity_analysis.png."""
        baseline_sim = {"within_mean": 0.36, "between_mean": 0.29, "separation_ratio": 1.24}
        custom_sim = {"within_mean": 0.42, "between_mean": 0.36, "separation_ratio": 1.17}

        mock_fig = MagicMock()
        mock_axes = (MagicMock(), MagicMock())
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        from src.visualization.plotter import EmbeddingVisualizer
        viz = EmbeddingVisualizer(str(tmp_path))

        with patch.object(viz, "_save_fig") as mock_save:
            viz.plot_within_between_similarity(baseline_sim, custom_sim)
            mock_save.assert_called_once()
            assert "similarity" in mock_save.call_args[0][1]


# ---------------------------------------------------------------------------
# plot_cluster_distribution
# ---------------------------------------------------------------------------


class TestPlotClusterDistribution:

    @patch("src.visualization.plotter.plt")
    def test_plot_cluster_distribution_generates_file(self, mock_plt, tmp_path,
                                                       sample_chunks_20, sample_kmeans_results):
        """plot_cluster_distribution produces cluster_distribution.png."""
        mock_fig = MagicMock()
        mock_axes = (MagicMock(), MagicMock())
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        from src.visualization.plotter import EmbeddingVisualizer
        viz = EmbeddingVisualizer(str(tmp_path))

        with patch.object(viz, "_save_fig") as mock_save:
            viz.plot_cluster_distribution(sample_kmeans_results, sample_chunks_20, best_k=4)
            mock_save.assert_called_once()
            assert "cluster" in mock_save.call_args[0][1]

    def test_plot_cluster_distribution_skips_missing_k(self, tmp_path, sample_chunks_20):
        """If best_k is not in results, no plot is generated."""
        from src.visualization.plotter import EmbeddingVisualizer
        viz = EmbeddingVisualizer(str(tmp_path))

        with patch.object(viz, "_save_fig") as mock_save:
            viz.plot_cluster_distribution({}, sample_chunks_20, best_k=99)
            mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# plot_neighbor_overlap
# ---------------------------------------------------------------------------


class TestPlotNeighborOverlap:

    @patch("src.visualization.plotter.plt")
    def test_plot_neighbor_overlap_generates_file(self, mock_plt, tmp_path,
                                                  sample_embeddings_20, sample_chunks_20):
        """plot_neighbor_overlap creates neighbor_overlap.png."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        from src.visualization.plotter import EmbeddingVisualizer
        viz = EmbeddingVisualizer(str(tmp_path))

        with patch.object(viz, "_save_fig") as mock_save:
            viz.plot_neighbor_overlap(sample_embeddings_20, sample_embeddings_20, sample_chunks_20)
            mock_save.assert_called_once()
            assert "neighbor" in mock_save.call_args[0][1]


# ---------------------------------------------------------------------------
# generate_all_plots
# ---------------------------------------------------------------------------


class TestGenerateAllPlots:

    def test_generate_all_plots_calls_all_methods(self, tmp_path,
                                                   sample_embeddings_20, sample_chunks_20,
                                                   sample_kmeans_results, sample_eval_results):
        """generate_all_plots delegates to all five individual plot methods."""
        from src.visualization.plotter import EmbeddingVisualizer
        viz = EmbeddingVisualizer(str(tmp_path))

        method_names = [
            "plot_tsne_comparison",
            "plot_silhouette_comparison",
            "plot_within_between_similarity",
            "plot_cluster_distribution",
            "plot_neighbor_overlap",
        ]

        for name in method_names:
            with patch.object(viz, name) as mock_method:
                mock_method.return_value = None

        # Now call generate_all_plots (real methods, but we patch them)
        with patch.object(viz, "plot_tsne_comparison") as m1, \
             patch.object(viz, "plot_silhouette_comparison") as m2, \
             patch.object(viz, "plot_within_between_similarity") as m3, \
             patch.object(viz, "plot_cluster_distribution") as m4, \
             patch.object(viz, "plot_neighbor_overlap") as m5:

            result = viz.generate_all_plots(
                sample_embeddings_20, sample_embeddings_20,
                sample_chunks_20, sample_kmeans_results, sample_eval_results,
            )

            m1.assert_called_once_with(sample_embeddings_20, sample_embeddings_20, sample_chunks_20)
            m2.assert_called_once_with(sample_embeddings_20, sample_embeddings_20)
            m3.assert_called_once_with(
                sample_eval_results["baseline_similarity"],
                sample_eval_results["custom_similarity"],
            )
            m4.assert_called_once_with(sample_kmeans_results, sample_chunks_20, 4)
            m5.assert_called_once_with(sample_embeddings_20, sample_embeddings_20, sample_chunks_20)
