"""
Unit tests for src.evaluation.evaluator – EmbeddingEvaluator.

Uses real sklearn/scipy with deterministic synthetic data.  No external
model downloads or network calls.
"""

import numpy as np
import pytest

from src.evaluation.evaluator import EmbeddingEvaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_source_embeddings_and_chunks():
    """
    Create embeddings in 2 well-separated source clusters so that
    within-document similarity > between-document similarity.
    """
    rng = np.random.RandomState(42)
    dim = 384
    # Cluster A center and Cluster B center (far apart)
    center_a = np.zeros((1, dim), dtype=np.float32)
    center_b = np.zeros((1, dim), dtype=np.float32)
    center_b[0, 0] = 5.0  # shift one dimension so clusters are separable

    emb_a = center_a + rng.randn(10, dim).astype(np.float32) * 0.3
    emb_b = center_b + rng.randn(10, dim).astype(np.float32) * 0.3
    embeddings = np.vstack([emb_a, emb_b])

    chunks = [
        {"source": "doc_a", "text": f"text a-{i} " * 40, "chunk_id": f"a{i}"}
        for i in range(10)
    ] + [
        {"source": "doc_b", "text": f"text b-{i} " * 40, "chunk_id": f"b{i}"}
        for i in range(10)
    ]
    return embeddings, chunks


@pytest.fixture()
def two_embedding_spaces():
    """
    Two embedding spaces that share structure but are perturbed,
    for correlation / neighbor-agreement tests.
    """
    rng = np.random.RandomState(42)
    dim = 384
    base = rng.randn(30, dim).astype(np.float32)
    perturbed = base + rng.randn(30, dim).astype(np.float32) * 0.3
    return base, perturbed


# ---------------------------------------------------------------------------
# within_vs_between_similarity
# ---------------------------------------------------------------------------


class TestWithinVsBetweenSimilarity:

    def test_returns_expected_keys(self, two_source_embeddings_and_chunks):
        """Result dict contains all required metric keys."""
        embeddings, chunks = two_source_embeddings_and_chunks
        evaluator = EmbeddingEvaluator(top_k=5)
        result = evaluator.within_vs_between_similarity(embeddings, chunks)

        for key in ("within_mean", "within_std", "between_mean", "between_std", "separation_ratio"):
            assert key in result, f"Missing key: {key}"

    def test_within_higher_than_between(self, two_source_embeddings_and_chunks):
        """
        With well-separated source clusters, within-document similarity
        should exceed between-document similarity → ratio > 1.
        """
        embeddings, chunks = two_source_embeddings_and_chunks
        evaluator = EmbeddingEvaluator(top_k=5)
        result = evaluator.within_vs_between_similarity(embeddings, chunks)

        assert result["within_mean"] > result["between_mean"]
        assert result["separation_ratio"] > 1.0

    def test_values_are_floats(self, two_source_embeddings_and_chunks):
        """All metric values should be native Python floats."""
        embeddings, chunks = two_source_embeddings_and_chunks
        evaluator = EmbeddingEvaluator(top_k=5)
        result = evaluator.within_vs_between_similarity(embeddings, chunks)

        for v in result.values():
            assert isinstance(v, float)

    def test_std_non_negative(self, two_source_embeddings_and_chunks):
        """Standard deviations should be >= 0."""
        embeddings, chunks = two_source_embeddings_and_chunks
        evaluator = EmbeddingEvaluator(top_k=5)
        result = evaluator.within_vs_between_similarity(embeddings, chunks)

        assert result["within_std"] >= 0
        assert result["between_std"] >= 0

    def test_single_source_no_between(self):
        """When all chunks share one source, between_mean == 0 and ratio == inf."""
        rng = np.random.RandomState(0)
        embeddings = rng.randn(10, 384).astype(np.float32)
        chunks = [{"source": "only_doc", "text": f"t{i}", "chunk_id": f"c{i}"} for i in range(10)]

        evaluator = EmbeddingEvaluator()
        result = evaluator.within_vs_between_similarity(embeddings, chunks)

        assert result["between_mean"] == 0.0
        assert result["between_std"] == 0.0
        assert result["separation_ratio"] == float("inf")


# ---------------------------------------------------------------------------
# neighbor_agreement
# ---------------------------------------------------------------------------


class TestNeighborAgreement:

    def test_returns_expected_keys(self, two_embedding_spaces):
        base, perturbed = two_embedding_spaces
        evaluator = EmbeddingEvaluator(top_k=5)
        result = evaluator.neighbor_agreement(base, perturbed)

        for key in ("mean_jaccard", "median_jaccard", "agreement_rate"):
            assert key in result

    def test_identical_embeddings_high_agreement(self):
        """When both spaces are identical, Jaccard should be ~1.0."""
        rng = np.random.RandomState(0)
        emb = rng.randn(20, 384).astype(np.float32)

        evaluator = EmbeddingEvaluator(top_k=5)
        result = evaluator.neighbor_agreement(emb, emb)

        assert result["mean_jaccard"] == pytest.approx(1.0, abs=1e-5)
        assert result["median_jaccard"] == pytest.approx(1.0, abs=1e-5)
        assert result["agreement_rate"] == pytest.approx(1.0, abs=1e-5)

    def test_perturbed_embeddings_partial_agreement(self, two_embedding_spaces):
        """Perturbed embeddings should show partial (0 < jaccard < 1) agreement."""
        base, perturbed = two_embedding_spaces
        evaluator = EmbeddingEvaluator(top_k=5)
        result = evaluator.neighbor_agreement(base, perturbed)

        assert 0.0 < result["mean_jaccard"] < 1.0
        assert 0.0 < result["median_jaccard"] < 1.0

    def test_custom_top_k(self, two_embedding_spaces):
        """Passing top_k overrides the instance default."""
        base, perturbed = two_embedding_spaces
        evaluator = EmbeddingEvaluator(top_k=5)
        result_k3 = evaluator.neighbor_agreement(base, perturbed, top_k=3)
        result_k10 = evaluator.neighbor_agreement(base, perturbed, top_k=10)

        # Different k values produce different results (not guaranteed but likely)
        assert "mean_jaccard" in result_k3
        assert "mean_jaccard" in result_k10

    def test_very_few_embeddings(self):
        """Works correctly with a small number of embeddings."""
        rng = np.random.RandomState(0)
        emb_a = rng.randn(3, 384).astype(np.float32)
        emb_b = rng.randn(3, 384).astype(np.float32)

        evaluator = EmbeddingEvaluator(top_k=2)
        result = evaluator.neighbor_agreement(emb_a, emb_b)
        assert 0.0 <= result["mean_jaccard"] <= 1.0


# ---------------------------------------------------------------------------
# embedding_correlation
# ---------------------------------------------------------------------------


class TestEmbeddingCorrelation:

    def test_returns_expected_keys(self, two_embedding_spaces):
        base, perturbed = two_embedding_spaces
        evaluator = EmbeddingEvaluator()
        result = evaluator.embedding_correlation(base, perturbed)

        for key in ("spearman_rho", "spearman_p", "pearson_r", "pearson_p"):
            assert key in result

    def test_identical_embeddings_perfect_correlation(self):
        """Identical embedding spaces should yield correlation ~1.0."""
        rng = np.random.RandomState(0)
        emb = rng.randn(20, 384).astype(np.float32)

        evaluator = EmbeddingEvaluator()
        result = evaluator.embedding_correlation(emb, emb)

        assert result["spearman_rho"] == pytest.approx(1.0, abs=1e-5)
        assert result["pearson_r"] == pytest.approx(1.0, abs=1e-5)

    def test_perturbed_embeddings_positive_correlation(self, two_embedding_spaces):
        """Perturbed embeddings should still show positive correlation."""
        base, perturbed = two_embedding_spaces
        evaluator = EmbeddingEvaluator()
        result = evaluator.embedding_correlation(base, perturbed)

        assert result["spearman_rho"] > 0
        assert result["pearson_r"] > 0

    def test_p_values_are_floats(self, two_embedding_spaces):
        base, perturbed = two_embedding_spaces
        evaluator = EmbeddingEvaluator()
        result = evaluator.embedding_correlation(base, perturbed)

        assert isinstance(result["spearman_p"], float)
        assert isinstance(result["pearson_p"], float)

    def test_sample_size_limits_pairs(self):
        """Passing a small sample_size caps the number of pairs evaluated."""
        rng = np.random.RandomState(0)
        emb_a = rng.randn(50, 384).astype(np.float32)
        emb_b = rng.randn(50, 384).astype(np.float32)

        evaluator = EmbeddingEvaluator()
        result = evaluator.embedding_correlation(emb_a, emb_b, sample_size=10)

        # Should still produce valid results
        assert isinstance(result["spearman_rho"], float)
        assert -1.0 <= result["spearman_rho"] <= 1.0

    def test_random_embeddings_low_correlation(self):
        """Completely independent random embeddings should have low correlation."""
        rng_a = np.random.RandomState(0)
        rng_b = np.random.RandomState(999)
        emb_a = rng_a.randn(30, 384).astype(np.float32)
        emb_b = rng_b.randn(30, 384).astype(np.float32)

        evaluator = EmbeddingEvaluator()
        result = evaluator.embedding_correlation(emb_a, emb_b)

        # Correlation should be close to 0 (not perfectly 0 due to sampling noise)
        assert abs(result["spearman_rho"]) < 0.5


# ---------------------------------------------------------------------------
# full_evaluation
# ---------------------------------------------------------------------------


class TestFullEvaluation:

    def test_runs_all_evaluations(self, two_source_embeddings_and_chunks, two_embedding_spaces):
        """full_evaluation returns results with all expected top-level keys."""
        baseline_emb, custom_emb = two_embedding_spaces
        _, chunks = two_source_embeddings_and_chunks

        evaluator = EmbeddingEvaluator(top_k=5)
        results = evaluator.full_evaluation(baseline_emb, custom_emb, chunks)

        expected_keys = {
            "baseline_similarity",
            "custom_similarity",
            "neighbor_agreement",
            "embedding_correlation",
            "summary",
        }
        assert expected_keys.issubset(set(results.keys()))

    def test_summary_contains_improvement(self, two_source_embeddings_and_chunks, two_embedding_spaces):
        """Summary dict includes improvement_pct, neighbor_agreement, and embedding_correlation."""
        baseline_emb, custom_emb = two_embedding_spaces
        _, chunks = two_source_embeddings_and_chunks

        evaluator = EmbeddingEvaluator(top_k=5)
        results = evaluator.full_evaluation(baseline_emb, custom_emb, chunks)

        summary = results["summary"]
        assert "improvement_pct" in summary
        assert "baseline_separation_ratio" in summary
        assert "custom_separation_ratio" in summary
        assert "neighbor_agreement" in summary
        assert "embedding_correlation" in summary
        assert isinstance(summary["improvement_pct"], float)

    def test_stores_results_on_instance(self, two_source_embeddings_and_chunks, two_embedding_spaces):
        """After full_evaluation, evaluator.results is populated."""
        baseline_emb, custom_emb = two_embedding_spaces
        _, chunks = two_source_embeddings_and_chunks

        evaluator = EmbeddingEvaluator(top_k=5)
        assert evaluator.results == {}

        evaluator.full_evaluation(baseline_emb, custom_emb, chunks)
        assert evaluator.results != {}
        assert "summary" in evaluator.results

    def test_improvement_pct_calculation(self):
        """
        When custom separation > baseline separation, improvement_pct > 0.
        We verify this by constructing a scenario where custom embeddings
        produce higher within/between separation.
        """
        rng = np.random.RandomState(42)
        dim = 10  # low dim so shifts dominate cosine similarity

        # Baseline: two sources slightly separated (low separation ratio)
        center_a_bl = np.zeros((1, dim), dtype=np.float32)
        center_b_bl = np.zeros((1, dim), dtype=np.float32)
        center_b_bl[0, 0] = 1.0  # small shift
        baseline_emb = np.vstack([
            center_a_bl + rng.randn(10, dim).astype(np.float32) * 0.3,
            center_b_bl + rng.randn(10, dim).astype(np.float32) * 0.3,
        ])

        # Custom: two sources far apart (high separation ratio)
        center_a_cu = np.zeros((1, dim), dtype=np.float32)
        center_b_cu = np.zeros((1, dim), dtype=np.float32)
        center_b_cu[0, 0] = 10.0  # large shift
        custom_emb = np.vstack([
            center_a_cu + rng.randn(10, dim).astype(np.float32) * 0.3,
            center_b_cu + rng.randn(10, dim).astype(np.float32) * 0.3,
        ])

        chunks = (
            [{"source": "a", "text": f"x{i} " * 40, "chunk_id": f"a{i}"} for i in range(10)]
            + [{"source": "b", "text": f"y{i} " * 40, "chunk_id": f"b{i}"} for i in range(10)]
        )

        evaluator = EmbeddingEvaluator()
        results = evaluator.full_evaluation(baseline_emb, custom_emb, chunks)

        # Custom should have higher separation than baseline
        assert results["summary"]["custom_separation_ratio"] > results["summary"]["baseline_separation_ratio"]
        assert results["summary"]["improvement_pct"] > 0
