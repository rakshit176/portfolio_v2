"""
Clustering module for document embedding analysis.

Provides KMeans and HDBSCAN clustering with comprehensive evaluation metrics
including Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

logger = logging.getLogger(__name__)


class DocumentClusterer:
    """
    Clusters document embeddings using multiple algorithms.

    Supports:
      - KMeans: Fast, scalable, good for evenly-sized clusters
      - HDBSCAN: Density-based, handles variable cluster sizes and noise
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.kmeans_results: Dict[int, Dict] = {}
        self.hdbscan_result: Optional[Dict] = None
        self.best_kmeans_k: Optional[int] = None
        self.best_kmeans_score: float = -1.0

    def kmeans_clustering(
        self,
        embeddings: np.ndarray,
        k_range: Optional[List[int]] = None,
        n_init: int = 10,
    ) -> Dict[int, Dict]:
        """
        Run KMeans clustering for multiple k values.

        Args:
            embeddings: numpy array of shape (n_samples, n_features).
            k_range: List of k values to try.
            n_init: Number of KMeans initializations per k.

        Returns:
            Dict mapping k -> {labels, inertia, silhouette, davies_bouldin, calinski_harabasz}.
        """
        k_range = k_range or [3, 4, 5, 6, 7, 8]
        n_samples = embeddings.shape[0]

        # Silhouette score requires at least 2 clusters and n_samples > n_clusters
        valid_k_range = [k for k in k_range if 2 <= k < n_samples]

        for k in valid_k_range:
            logger.info(f"Running KMeans with k={k}")
            kmeans = KMeans(
                n_clusters=k,
                n_init=n_init,
                max_iter=300,
                random_state=self.random_state,
            )
            labels = kmeans.fit_predict(embeddings)

            result = {
                "labels": labels,
                "inertia": float(kmeans.inertia_),
                "n_clusters": k,
                "algorithm": "kmeans",
            }

            # Calculate metrics (only if we have enough samples)
            if len(set(labels)) > 1 and n_samples > k:
                result["silhouette"] = silhouette_score(embeddings, labels)
                result["davies_bouldin"] = davies_bouldin_score(embeddings, labels)
                result["calinski_harabasz"] = calinski_harabasz_score(embeddings, labels)

                logger.info(
                    f"  k={k}: silhouette={result['silhouette']:.4f}, "
                    f"DB={result['davies_bouldin']:.4f}, "
                    f"CH={result['calinski_harabasz']:.1f}"
                )

                # Track best k by silhouette score
                if result["silhouette"] > self.best_kmeans_score:
                    self.best_kmeans_score = result["silhouette"]
                    self.best_kmeans_k = k

            self.kmeans_results[k] = result

        logger.info(f"Best KMeans k={self.best_kmeans_k} (silhouette={self.best_kmeans_score:.4f})")
        return self.kmeans_results

    def hdbscan_clustering(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        min_samples: int = 3,
    ) -> Optional[Dict]:
        """
        Run HDBSCAN density-based clustering.

        Args:
            embeddings: numpy array of shape (n_samples, n_features).
            min_cluster_size: Minimum cluster size for HDBSCAN.
            min_samples: Minimum samples in neighborhood.

        Returns:
            Dict with labels, cluster info, and metrics.
        """
        try:
            import hdbscan
        except ImportError:
            logger.warning(
                "HDBSCAN not installed. Skipping HDBSCAN clustering. "
                "Install with: pip install hdbscan"
            )
            return None

        logger.info(f"Running HDBSCAN (min_cluster_size={min_cluster_size})")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
        )

        labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        result = {
            "labels": labels,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": n_noise / len(labels),
            "algorithm": "hdbscan",
            "probabilities": clusterer.probabilities_.tolist(),
            "cluster_persistence": (
                clusterer.cluster_persistence_.tolist()
                if hasattr(clusterer, 'cluster_persistence_')
                else None
            ),
        }

        # Calculate metrics (exclude noise points)
        if n_clusters > 1 and n_clusters < len(labels) - n_noise:
            non_noise_mask = labels != -1
            if non_noise_mask.sum() > n_clusters:
                result["silhouette"] = silhouette_score(
                    embeddings[non_noise_mask], labels[non_noise_mask]
                )
                logger.info(f"  HDBSCAN: {n_clusters} clusters, silhouette={result['silhouette']:.4f}")
            else:
                logger.info(f"  HDBSCAN: {n_clusters} clusters, not enough non-noise for metrics")
        else:
            logger.info(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

        self.hdbscan_result = result
        return result

    def get_optimal_labels(
        self, embeddings: np.ndarray, method: str = "kmeans"
    ) -> np.ndarray:
        """
        Get cluster labels from the best clustering result.

        Args:
            embeddings: Input embeddings.
            method: 'kmeans' or 'hdbscan'.

        Returns:
            numpy array of cluster labels.
        """
        if method == "kmeans" and self.best_kmeans_k:
            return self.kmeans_results[self.best_kmeans_k]["labels"]
        elif method == "hdbscan" and self.hdbscan_result:
            return self.hdbscan_result["labels"]
        elif self.kmeans_results:
            # Fallback to best available KMeans
            return self.kmeans_results[self.best_kmeans_k]["labels"]
        else:
            raise RuntimeError("No clustering results available. Run clustering first.")

    def get_cluster_summary(
        self, chunks: List[Dict], labels: np.ndarray
    ) -> Dict[int, Dict]:
        """
        Generate per-cluster summary statistics.

        Args:
            chunks: List of chunk dicts with 'text', 'source', etc.
            labels: numpy array of cluster assignments.

        Returns:
            Dict mapping cluster_id -> {size, sources, sample_texts}.
        """
        summary = {}
        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id
            cluster_chunks = [c for c, m in zip(chunks, mask) if m]

            sources = {}
            for c in cluster_chunks:
                src = c.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1

            summary[int(cluster_id)] = {
                "size": int(mask.sum()),
                "sources": dict(sorted(sources.items(), key=lambda x: -x[1])),
                "sample_texts": [c["text"][:150] + "..." for c in cluster_chunks[:3]],
            }

        return summary
