"""
Evaluation module for comparing baseline and custom embedding models.

Computes:
  - Cosine similarity distributions
  - Within-cluster vs between-cluster similarity
  - Neighbor overlap analysis
  - Pairwise similarity correlation between models
  - Clustering quality comparison (silhouette, Davies-Bouldin, Calinski-Harabasz)
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class EmbeddingEvaluator:
    """
    Compares baseline and custom embeddings on legal document chunks.

    Evaluation dimensions:
      1. Similarity quality: Do semantically similar chunks score higher?
      2. Clustering separation: Are document boundaries preserved in embedding space?
      3. Neighbor agreement: Do both models agree on nearest neighbors?
      4. Distribution shift: How different is the embedding distribution?
    """

    def __init__(self, top_k: int = 5, random_state: int = 42):
        self.top_k = top_k
        self.random_state = random_state
        self.results: Dict = {}

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.

        Args:
            embeddings: (n_samples, n_dim) array.

        Returns:
            (n_samples, n_samples) similarity matrix.
        """
        return cosine_similarity(embeddings)

    def within_vs_between_similarity(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict],
    ) -> Dict[str, float]:
        """
        Compute within-document vs between-document similarity.

        Measures how well the embeddings separate chunks from different
        source documents. A good domain embedder should have higher
        within-document similarity than between-document similarity.

        Args:
            embeddings: (n_samples, n_dim) array.
            chunks: List of chunk dicts with 'source' key.

        Returns:
            Dict with 'within_mean', 'between_mean', 'separation_ratio'.
        """
        sim_matrix = self.compute_similarity_matrix(embeddings)
        n = len(chunks)

        # Build source index
        source_indices: Dict[str, List[int]] = defaultdict(list)
        for i, chunk in enumerate(chunks):
            source_indices[chunk["source"]].append(i)

        within_similarities = []
        between_similarities = []

        for i in range(n):
            for j in range(i + 1, n):
                sim = sim_matrix[i, j]
                if chunks[i]["source"] == chunks[j]["source"]:
                    within_similarities.append(sim)
                else:
                    between_similarities.append(sim)

        within_mean = np.mean(within_similarities) if within_similarities else 0.0
        between_mean = np.mean(between_similarities) if between_similarities else 0.0
        separation_ratio = within_mean / between_mean if between_mean > 0 else float('inf')

        result = {
            "within_mean": float(within_mean),
            "within_std": float(np.std(within_similarities)) if within_similarities else 0.0,
            "between_mean": float(between_mean),
            "between_std": float(np.std(between_similarities)) if between_similarities else 0.0,
            "separation_ratio": float(separation_ratio),
        }

        logger.info(
            f"Similarity: within={within_mean:.4f} +/- {result['within_std']:.4f}, "
            f"between={between_mean:.4f} +/- {result['between_std']:.4f}, "
            f"ratio={separation_ratio:.4f}"
        )

        return result

    def neighbor_agreement(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Measure neighbor overlap between two embedding spaces.

        Computes the Jaccard similarity between top-k neighbor sets
        for each point in both embedding spaces.

        Args:
            emb_a: First embedding space.
            emb_b: Second embedding space.
            top_k: Number of neighbors to consider.

        Returns:
            Dict with 'mean_jaccard', 'median_jaccard', 'agreement_rate'.
        """
        top_k = top_k or self.top_k
        sim_a = self.compute_similarity_matrix(emb_a)
        sim_b = self.compute_similarity_matrix(emb_b)

        # For each point, get top-k neighbors (excluding self)
        jaccard_scores = []
        for i in range(len(emb_a)):
            # Get top-k neighbor indices (excluding self)
            neighbors_a = set(np.argsort(sim_a[i])[-(top_k + 1):-1])
            neighbors_b = set(np.argsort(sim_b[i])[-(top_k + 1):-1])

            # Jaccard similarity
            intersection = len(neighbors_a & neighbors_b)
            union = len(neighbors_a | neighbors_b)
            jaccard = intersection / union if union > 0 else 0.0
            jaccard_scores.append(jaccard)

        result = {
            "mean_jaccard": float(np.mean(jaccard_scores)),
            "median_jaccard": float(np.median(jaccard_scores)),
            "agreement_rate": float(np.mean([j > 0.3 for j in jaccard_scores])),
        }

        logger.info(
            f"Neighbor agreement (top-{top_k}): "
            f"mean_jaccard={result['mean_jaccard']:.4f}, "
            f"agreement_rate={result['agreement_rate']:.2%}"
        )

        return result

    def embedding_correlation(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
        sample_size: int = 5000,
    ) -> Dict[str, float]:
        """
        Compute correlation between pairwise similarity distributions.

        Measures how similarly two embedding models rank document pairs.

        Args:
            emb_a: First embedding space.
            emb_b: Second embedding space.
            sample_size: Number of random pairs to sample.

        Returns:
            Dict with 'spearman_rho', 'pearson_r', and p-values.
        """
        rng = np.random.RandomState(self.random_state)
        n = len(emb_a)
        num_pairs = min(sample_size, n * (n - 1) // 2)

        # Sample random pairs
        pairs = []
        for _ in range(num_pairs):
            i, j = rng.randint(0, n, size=2)
            if i != j:
                pairs.append((i, j))

        sim_a_pairs = []
        sim_b_pairs = []
        sim_matrix_a = self.compute_similarity_matrix(emb_a)
        sim_matrix_b = self.compute_similarity_matrix(emb_b)

        for i, j in pairs:
            sim_a_pairs.append(sim_matrix_a[i, j])
            sim_b_pairs.append(sim_matrix_b[i, j])

        spearman_rho, spearman_p = spearmanr(sim_a_pairs, sim_b_pairs)
        pearson_r, pearson_p = pearsonr(sim_a_pairs, sim_b_pairs)

        result = {
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
        }

        logger.info(
            f"Embedding correlation: Spearman={spearman_rho:.4f}, Pearson={pearson_r:.4f}"
        )

        return result

    def full_evaluation(
        self,
        baseline_embeddings: np.ndarray,
        custom_embeddings: np.ndarray,
        chunks: List[Dict],
    ) -> Dict[str, Any]:
        """
        Run the complete evaluation suite comparing two embedding models.

        Args:
            baseline_embeddings: Embeddings from baseline model.
            custom_embeddings: Embeddings from custom model.
            chunks: List of chunk dicts.

        Returns:
            Dict with all evaluation results.
        """
        logger.info("Running full evaluation suite...")

        results = {}

        # 1. Within vs between document similarity
        logger.info("1. Within/Between document similarity...")
        results["baseline_similarity"] = self.within_vs_between_similarity(
            baseline_embeddings, chunks
        )
        results["custom_similarity"] = self.within_vs_between_similarity(
            custom_embeddings, chunks
        )

        # 2. Neighbor agreement between models
        logger.info("2. Neighbor agreement analysis...")
        results["neighbor_agreement"] = self.neighbor_agreement(
            baseline_embeddings, custom_embeddings
        )

        # 3. Embedding correlation
        logger.info("3. Embedding correlation analysis...")
        results["embedding_correlation"] = self.embedding_correlation(
            baseline_embeddings, custom_embeddings
        )

        # 4. Summary comparison
        baseline_sep = results["baseline_similarity"]["separation_ratio"]
        custom_sep = results["custom_similarity"]["separation_ratio"]
        improvement = (custom_sep - baseline_sep) / baseline_sep * 100 if baseline_sep > 0 else 0

        results["summary"] = {
            "baseline_separation_ratio": baseline_sep,
            "custom_separation_ratio": custom_sep,
            "improvement_pct": improvement,
            "neighbor_agreement": results["neighbor_agreement"]["mean_jaccard"],
            "embedding_correlation": results["embedding_correlation"]["spearman_rho"],
        }

        self.results = results
        logger.info(
            f"Evaluation complete. Separation improvement: {improvement:+.1f}%"
        )

        return results
