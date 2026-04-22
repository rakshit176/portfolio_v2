"""
Q2 Custom Embedder for Domain Documents - Main Entry Point

Runs the complete pipeline:
  1. Preprocess PDFs -> clean text -> chunks
  2. Generate baseline embeddings (all-MiniLM-L6-v2)
  3. Fine-tune custom embedder on legal domain
  4. Generate custom embeddings
  5. Cluster both embedding spaces
  6. Evaluate and compare (similarity, silhouette, neighbor overlap)
  7. Generate visualizations
  8. Save all results
"""

import logging
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure outputs dir exists before logging setup
_BASE_DIR = Path(__file__).parent
(_BASE_DIR / "outputs").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_BASE_DIR / 'outputs' / 'pipeline.log'), mode='w'),
    ],
)
logger = logging.getLogger("q2_main")


def run_full_pipeline():
    """Execute the complete Q2 pipeline."""
    from src.utils.config import get_config
    from src.preprocessing.text_processor import preprocess_pipeline
    from src.embedding.baseline import BaselineEmbedder
    from src.embedding.training import train_custom_embedder
    from src.embedding.custom_embedder import CustomEmbedder
    from src.clustering.clusterer import DocumentClusterer
    from src.evaluation.evaluator import EmbeddingEvaluator
    from src.visualization.plotter import EmbeddingVisualizer

    config = get_config()
    total_start = time.time()

    # Ensure output directories exist
    for subdir in [config.embeddings_dir, config.plots_dir, config.models_dir, config.results_dir]:
        subdir.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # STEP 1: Preprocessing
    # ======================================================================
    logger.info("=" * 70)
    logger.info("STEP 1: PREPROCESSING - Extract, clean, and chunk PDFs")
    logger.info("=" * 70)

    step_start = time.time()
    chunks, documents = preprocess_pipeline(config.data_dir)
    logger.info(f"Preprocessing complete: {len(chunks)} chunks from {len(documents)} documents "
                f"in {time.time() - step_start:.1f}s")

    if len(chunks) < 10:
        logger.error("Not enough chunks for meaningful analysis. Check PDF contents.")
        return

    # Save chunks metadata
    chunks_meta = [{
        "chunk_id": c["chunk_id"],
        "source": c["source"],
        "chunk_index": c["chunk_index"],
        "char_count": c["char_count"],
        "text_preview": c["text"][:100],
    } for c in chunks]
    with open(config.results_dir / "chunks_metadata.json", "w") as f:
        json.dump(chunks_meta, f, indent=2)

    # ======================================================================
    # STEP 2: Baseline Embeddings
    # ======================================================================
    logger.info("=" * 70)
    logger.info("STEP 2: BASELINE EMBEDDINGS - all-MiniLM-L6-v2")
    logger.info("=" * 70)

    step_start = time.time()
    baseline_embedder = BaselineEmbedder(config.embedding).load()
    texts = [c["text"] for c in chunks]
    baseline_embeddings = baseline_embedder.embed(texts)

    np.save(config.embeddings_dir / "baseline_embeddings.npy", baseline_embeddings)
    logger.info(f"Baseline embeddings saved: {baseline_embeddings.shape} "
                f"in {time.time() - step_start:.1f}s")

    # ======================================================================
    # STEP 3: Fine-tune Custom Embedder
    # ======================================================================
    logger.info("=" * 70)
    logger.info("STEP 3: TRAINING - Fine-tune custom domain embedder")
    logger.info("=" * 70)

    step_start = time.time()

    # Always train (skip disk load to avoid OOM)
    logger.info("Starting fine-tuning on legal corpus...")
    custom_embedder = train_custom_embedder(chunks, config.embedding)

    logger.info(f"Training/loading complete in {time.time() - step_start:.1f}s")

    # ======================================================================
    # STEP 4: Custom Embeddings
    # ======================================================================
    logger.info("=" * 70)
    logger.info("STEP 4: CUSTOM EMBEDDINGS - Legal Domain Embedder")
    logger.info("=" * 70)

    step_start = time.time()
    custom_embeddings = custom_embedder.embed(texts)

    np.save(config.embeddings_dir / "custom_embeddings.npy", custom_embeddings)
    logger.info(f"Custom embeddings saved: {custom_embeddings.shape} "
                f"in {time.time() - step_start:.1f}s")

    # ======================================================================
    # STEP 5: Clustering
    # ======================================================================
    logger.info("=" * 70)
    logger.info("STEP 5: CLUSTERING - KMeans & HDBSCAN")
    logger.info("=" * 70)

    step_start = time.time()

    # Cluster baseline embeddings
    logger.info("Clustering baseline embeddings...")
    baseline_clusterer = DocumentClusterer(random_state=42)
    baseline_clusterer.kmeans_clustering(baseline_embeddings, config.clustering.kmeans_k_range)
    baseline_clusterer.hdbscan_clustering(
        baseline_embeddings,
        config.clustering.hdbscan_min_cluster_size,
        config.clustering.hdbscan_min_samples,
    )

    # Cluster custom embeddings
    logger.info("Clustering custom embeddings...")
    custom_clusterer = DocumentClusterer(random_state=42)
    custom_clusterer.kmeans_clustering(custom_embeddings, config.clustering.kmeans_k_range)
    custom_clusterer.hdbscan_clustering(
        custom_embeddings,
        config.clustering.hdbscan_min_cluster_size,
        config.clustering.hdbscan_min_samples,
    )

    logger.info(f"Clustering complete in {time.time() - step_start:.1f}s")
    logger.info(f"  Best KMeans k (baseline): {baseline_clusterer.best_kmeans_k} "
                f"(silhouette={baseline_clusterer.best_kmeans_score:.4f})")
    logger.info(f"  Best KMeans k (custom): {custom_clusterer.best_kmeans_k} "
                f"(silhouette={custom_clusterer.best_kmeans_score:.4f})")

    # ======================================================================
    # STEP 6: Evaluation
    # ======================================================================
    logger.info("=" * 70)
    logger.info("STEP 6: EVALUATION - Baseline vs Custom comparison")
    logger.info("=" * 70)

    step_start = time.time()
    evaluator = EmbeddingEvaluator(top_k=config.evaluation.top_k_neighbors)
    eval_results = evaluator.full_evaluation(baseline_embeddings, custom_embeddings, chunks)

    # Add clustering results to evaluation
    eval_results["baseline_best_k"] = baseline_clusterer.best_kmeans_k
    eval_results["baseline_best_silhouette"] = baseline_clusterer.best_kmeans_score
    eval_results["custom_best_k"] = custom_clusterer.best_kmeans_k
    eval_results["custom_best_silhouette"] = custom_clusterer.best_kmeans_score
    eval_results["best_custom_k"] = custom_clusterer.best_kmeans_k

    # Save evaluation results
    with open(config.results_dir / "evaluation_results.json", "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(eval_results, f, indent=2, default=convert)

    logger.info(f"Evaluation complete in {time.time() - step_start:.1f}s")

    # ======================================================================
    # STEP 7: Visualization
    # ======================================================================
    logger.info("=" * 70)
    logger.info("STEP 7: VISUALIZATION - Generate plots")
    logger.info("=" * 70)

    step_start = time.time()
    visualizer = EmbeddingVisualizer(str(config.plots_dir))
    plot_files = visualizer.generate_all_plots(
        baseline_embeddings,
        custom_embeddings,
        chunks,
        custom_clusterer.kmeans_results,
        eval_results,
    )
    logger.info(f"Generated {len(plot_files)} plots in {time.time() - step_start:.1f}s")

    # ======================================================================
    # STEP 8: Summary
    # ======================================================================
    total_time = time.time() - total_start
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Documents processed: {len(documents)}")
    logger.info(f"Chunks created: {len(chunks)}")
    logger.info(f"Baseline embedding dim: {baseline_embeddings.shape[1]}")
    logger.info(f"Custom embedding dim: {custom_embeddings.shape[1]}")
    logger.info(f"Best baseline silhouette: {baseline_clusterer.best_kmeans_score:.4f} (k={baseline_clusterer.best_kmeans_k})")
    logger.info(f"Best custom silhouette: {custom_clusterer.best_kmeans_score:.4f} (k={custom_clusterer.best_kmeans_k})")
    logger.info(f"Separation improvement: {eval_results['summary']['improvement_pct']:+.1f}%")
    logger.info(f"Plots saved to: {config.plots_dir}")
    logger.info(f"Results saved to: {config.results_dir}")

    # Save summary
    summary = {
        "total_time_seconds": round(total_time, 1),
        "documents_processed": len(documents),
        "chunks_created": len(chunks),
        "baseline_model": config.embedding.baseline_model_name,
        "custom_model": config.embedding.custom_model_name,
        "embedding_dimension": int(baseline_embeddings.shape[1]),
        "baseline_best_k": baseline_clusterer.best_kmeans_k,
        "baseline_best_silhouette": round(baseline_clusterer.best_kmeans_score, 4),
        "custom_best_k": custom_clusterer.best_kmeans_k,
        "custom_best_silhouette": round(custom_clusterer.best_kmeans_score, 4),
        "separation_improvement_pct": round(eval_results['summary']['improvement_pct'], 2),
        "neighbor_agreement": round(eval_results['summary']['neighbor_agreement'], 4),
        "embedding_correlation": round(eval_results['summary']['embedding_correlation'], 4),
        "plots_generated": plot_files,
    }

    with open(config.results_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    summary = run_full_pipeline()
    if summary:
        print("\n" + "=" * 70)
        print("Q2 PIPELINE COMPLETE - Summary:")
        print("=" * 70)
        for key, value in summary.items():
            print(f"  {key}: {value}")
