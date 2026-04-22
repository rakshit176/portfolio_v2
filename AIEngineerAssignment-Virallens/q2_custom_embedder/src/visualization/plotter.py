"""
Visualization module for embedding and clustering analysis.

Generates publication-quality plots:
  1. t-SNE scatter plots (baseline vs custom)
  2. Silhouette score comparison (bar chart)
  3. Cluster size distribution
  4. Within/Between similarity box plots
  5. Neighbor overlap heatmap
"""

import logging
from typing import List, Dict, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score

# Font setup
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'FreeSans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

# Color palette
COLORS = {
    'baseline': '#4C72B0',
    'custom': '#DD8452',
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#2CA02C',
    'danger': '#D62728',
}

CLUSTER_COLORS = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52',
    '#8172B2', '#937860', '#DA8BC3', '#8C8C8C',
    '#CCB974', '#64B5CD',
]


class EmbeddingVisualizer:
    """Generates all evaluation and analysis plots."""

    def __init__(self, output_dir: str, figsize: tuple = (12, 8)):
        self.output_dir = output_dir
        self.figsize = figsize
        self.plots_generated: List[str] = []

    def _save_fig(self, fig, name: str, dpi: int = 150):
        """Save figure and close."""
        path = f"{self.output_dir}/{name}"
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        self.plots_generated.append(path)
        logger.info(f"Saved plot: {path}")

    def plot_tsne_comparison(
        self,
        baseline_emb: np.ndarray,
        custom_emb: np.ndarray,
        chunks: List[Dict],
        perplexity: float = 30.0,
        n_iter: int = 1000,
        random_state: int = 42,
    ):
        """
        Generate side-by-side t-SNE visualizations.

        Colors points by source document to show how well each model
        separates documents in embedding space.
        """
        # Fit t-SNE on combined embeddings for fair comparison
        logger.info("Computing t-SNE projections...")
        tsne = TSNE(
            perplexity=perplexity,
            learning_rate=200.0,
            max_iter=n_iter,
            random_state=random_state,
            init='pca',
        )

        tsne_baseline = tsne.fit_transform(baseline_emb)
        tsne_custom = TSNE(
            perplexity=perplexity,
            learning_rate=200.0,
            max_iter=n_iter,
            random_state=random_state + 1,
            init='pca',
        ).fit_transform(custom_emb)

        # Get unique sources and assign colors
        sources = list(set(c["source"] for c in chunks))
        source_to_color = {s: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, s in enumerate(sources)}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Baseline t-SNE
        for source in sources:
            mask = np.array([c["source"] == source for c in chunks])
            ax1.scatter(
                tsne_baseline[mask, 0], tsne_baseline[mask, 1],
                c=source_to_color[source], label=source[:30],
                alpha=0.6, s=20, edgecolors='white', linewidths=0.3,
            )

        ax1.set_title('Baseline: all-MiniLM-L6-v2', fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        ax1.legend(loc='best', fontsize=7, framealpha=0.8)

        # Custom t-SNE
        for source in sources:
            mask = np.array([c["source"] == source for c in chunks])
            ax2.scatter(
                tsne_custom[mask, 0], tsne_custom[mask, 1],
                c=source_to_color[source], label=source[:30],
                alpha=0.6, s=20, edgecolors='white', linewidths=0.3,
            )

        ax2.set_title('Custom: Legal Domain Embedder (Fine-tuned)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')
        ax2.legend(loc='best', fontsize=7, framealpha=0.8)

        plt.suptitle('t-SNE Embedding Space: Baseline vs Custom', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_fig(fig, 'tsne_comparison.png')

    def plot_silhouette_comparison(
        self,
        baseline_emb: np.ndarray,
        custom_emb: np.ndarray,
        k_range: List[int] = None,
    ):
        """
        Bar chart comparing silhouette scores across cluster counts.
        """
        k_range = k_range or list(range(2, 9))
        baseline_scores = []
        custom_scores = []

        for k in k_range:
            from sklearn.cluster import KMeans

            if k >= len(baseline_emb):
                baseline_scores.append(0)
                custom_scores.append(0)
                continue

            km_b = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels_b = km_b.fit_predict(baseline_emb)
            if len(set(labels_b)) > 1:
                baseline_scores.append(silhouette_score(baseline_emb, labels_b))
            else:
                baseline_scores.append(0)

            km_c = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels_c = km_c.fit_predict(custom_emb)
            if len(set(labels_c)) > 1:
                custom_scores.append(silhouette_score(custom_emb, labels_c))
            else:
                custom_scores.append(0)

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(k_range))
        width = 0.35

        bars1 = ax.bar(x - width/2, baseline_scores, width,
                        label='Baseline (all-MiniLM-L6-v2)', color=COLORS['baseline'], alpha=0.85)
        bars2 = ax.bar(x + width/2, custom_scores, width,
                        label='Custom (Fine-tuned Legal)', color=COLORS['custom'], alpha=0.85)

        ax.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title('Silhouette Score Comparison: Baseline vs Custom Embedder',
                      fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_range])
        ax.legend(loc='best', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        self._save_fig(fig, 'silhouette_comparison.png')

    def plot_within_between_similarity(
        self,
        baseline_sim: Dict[str, float],
        custom_sim: Dict[str, float],
    ):
        """
        Box-style comparison of within-doc vs between-doc similarity.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        categories = ['Within-Doc', 'Between-Doc']
        baseline_vals = [baseline_sim['within_mean'], baseline_sim['between_mean']]
        custom_vals = [custom_sim['within_mean'], custom_sim['between_mean']]

        x = np.arange(len(categories))
        width = 0.35

        axes[0].bar(x - width/2, baseline_vals, width,
                     label='Baseline', color=COLORS['baseline'], alpha=0.85)
        axes[0].bar(x + width/2, custom_vals, width,
                     label='Custom', color=COLORS['custom'], alpha=0.85)
        axes[0].set_ylabel('Mean Cosine Similarity')
        axes[0].set_title('Within vs Between Document Similarity')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories)
        axes[0].legend(loc='best')
        axes[0].grid(axis='y', alpha=0.3)

        # Separation ratio comparison
        models = ['Baseline', 'Custom']
        ratios = [baseline_sim['separation_ratio'], custom_sim['separation_ratio']]

        bars = axes[1].bar(models, ratios, color=[COLORS['baseline'], COLORS['custom']], alpha=0.85)
        axes[1].set_ylabel('Separation Ratio')
        axes[1].set_title('Document Separation Ratio (Within / Between)')
        axes[1].grid(axis='y', alpha=0.3)

        for bar, ratio in zip(bars, ratios):
            axes[1].annotate(
                f'{ratio:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11,
                fontweight='bold'
            )

        plt.suptitle('Similarity Quality Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'similarity_analysis.png')

    def plot_cluster_distribution(
        self,
        kmeans_results: Dict[int, Dict],
        chunks: List[Dict],
        best_k: int,
    ):
        """
        Visualize cluster size distribution and source composition.
        """
        if best_k not in kmeans_results:
            return

        labels = kmeans_results[best_k]["labels"]
        unique_labels = sorted(set(labels))

        # Cluster size distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart of cluster sizes
        sizes = [int(np.sum(labels == k)) for k in unique_labels]
        colors = [CLUSTER_COLORS[k % len(CLUSTER_COLORS)] for k in unique_labels]

        ax1.bar(unique_labels, sizes, color=colors, alpha=0.85, edgecolor='white')
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Number of Chunks')
        ax1.set_title(f'Cluster Size Distribution (k={best_k})', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        for i, (label, size) in enumerate(zip(unique_labels, sizes)):
            ax1.annotate(str(size), xy=(label, size), ha='center', va='bottom', fontsize=10)

        # Stacked bar: source composition per cluster
        sources = sorted(set(c["source"] for c in chunks))
        source_colors = {s: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, s in enumerate(sources)}

        bottom = np.zeros(len(unique_labels))
        for source in sources:
            counts = []
            for k in unique_labels:
                mask = (labels == k)
                source_count = sum(1 for c, m in zip(chunks, mask) if m and c["source"] == source)
                counts.append(source_count)
            ax2.bar(unique_labels, counts, bottom=bottom, label=source[:25],
                    color=source_colors[source], alpha=0.8)
            bottom += np.array(counts)

        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Chunks')
        ax2.set_title(f'Source Composition per Cluster (k={best_k})', fontweight='bold')
        ax2.legend(loc='best', fontsize=7)
        ax2.grid(axis='y', alpha=0.3)

        plt.suptitle('Clustering Analysis (Custom Embedder)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'cluster_distribution.png')

    def plot_neighbor_overlap(
        self,
        baseline_emb: np.ndarray,
        custom_emb: np.ndarray,
        chunks: List[Dict],
        top_k: int = 5,
    ):
        """
        Visualize neighbor agreement as a per-document bar chart.

        Shows the average Jaccard overlap of top-k neighbors
        between baseline and custom embeddings, grouped by source document.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        sim_b = cosine_similarity(baseline_emb)
        sim_c = cosine_similarity(custom_emb)

        sources = sorted(set(c["source"] for c in chunks))
        source_jaccards = {}

        for source in sources:
            indices = [i for i, c in enumerate(chunks) if c["source"] == source]
            jaccards = []
            for idx in indices:
                neighbors_b = set(np.argsort(sim_b[idx])[-(top_k + 1):-1])
                neighbors_c = set(np.argsort(sim_c[idx])[-(top_k + 1):-1])
                intersection = len(neighbors_b & neighbors_c)
                union = len(neighbors_b | neighbors_c)
                jaccards.append(intersection / union if union > 0 else 0)
            source_jaccards[source] = np.mean(jaccards)

        fig, ax = plt.subplots(figsize=(12, 6))
        source_names = [s[:35] for s in sources]
        jaccard_vals = [source_jaccards[s] for s in sources]
        colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(sources))]

        bars = ax.barh(range(len(source_names)), jaccard_vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(source_names)))
        ax.set_yticklabels(source_names, fontsize=9)
        ax.set_xlabel(f'Average Jaccard Overlap (top-{top_k} neighbors)', fontsize=11)
        ax.set_title(f'Neighbor Agreement: Baseline vs Custom by Source Document',
                      fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)

        for bar, val in zip(bars, jaccard_vals):
            ax.annotate(f'{val:.3f}', xy=(val, bar.get_y() + bar.get_height()/2),
                       xytext=(5, 0), textcoords="offset points",
                       va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        self._save_fig(fig, 'neighbor_overlap.png')

    def generate_all_plots(
        self,
        baseline_emb: np.ndarray,
        custom_emb: np.ndarray,
        chunks: List[Dict],
        kmeans_results: Dict[int, Dict],
        evaluation_results: Dict,
    ):
        """
        Generate all visualization plots in one call.

        Args:
            baseline_emb: Baseline embeddings.
            custom_emb: Custom embeddings.
            chunks: List of chunk dicts.
            kmeans_results: KMeans clustering results dict.
            evaluation_results: Full evaluation results dict.
        """
        logger.info("Generating all visualization plots...")

        # 1. t-SNE comparison
        self.plot_tsne_comparison(baseline_emb, custom_emb, chunks)

        # 2. Silhouette comparison
        self.plot_silhouette_comparison(baseline_emb, custom_emb)

        # 3. Within/Between similarity
        self.plot_within_between_similarity(
            evaluation_results["baseline_similarity"],
            evaluation_results["custom_similarity"],
        )

        # 4. Cluster distribution (use best k from custom embeddings)
        best_k = evaluation_results.get("best_custom_k", 5)
        self.plot_cluster_distribution(kmeans_results, chunks, best_k)

        # 5. Neighbor overlap
        self.plot_neighbor_overlap(baseline_emb, custom_emb, chunks)

        logger.info(f"Generated {len(self.plots_generated)} plots total")
        return self.plots_generated
