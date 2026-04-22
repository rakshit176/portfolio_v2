"""Streamlined Q2 pipeline that runs with minimal memory."""
import os, gc, json, logging, time
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger('q2')

from src.utils.config import get_config
from src.preprocessing.text_processor import preprocess_pipeline
from src.embedding.baseline import BaselineEmbedder
from src.embedding.training import train_custom_embedder
from src.clustering.clusterer import DocumentClusterer
from src.evaluation.evaluator import EmbeddingEvaluator
from src.visualization.plotter import EmbeddingVisualizer

config = get_config()
for d in [config.embeddings_dir, config.plots_dir, config.results_dir]:
    d.mkdir(parents=True, exist_ok=True)

# --- STEP 1: Preprocessing ---
t0 = time.time()
chunks, documents = preprocess_pipeline(config.data_dir)
texts = [c['text'] for c in chunks]
logger.info(f"Step 1 done: {len(chunks)} chunks in {time.time()-t0:.1f}s")

# --- STEP 2: Baseline Embeddings ---
t0 = time.time()
bp = config.embeddings_dir / 'baseline_embeddings.npy'
if bp.exists():
    baseline_embeddings = np.load(bp)
    logger.info("Step 2: Loaded cached baseline")
else:
    be = BaselineEmbedder(config.embedding).load()
    baseline_embeddings = be.embed(texts, batch_size=8)
    del be
    gc.collect()
    np.save(bp, baseline_embeddings)
    logger.info(f"Step 2: Baseline {baseline_embeddings.shape} in {time.time()-t0:.1f}s")

# --- STEP 3-4: Train + Custom Embeddings ---
t0 = time.time()
logger.info("Step 3-4: Training custom embedder + generating embeddings...")
custom_embedder = train_custom_embedder(chunks, config.embedding)
custom_embeddings = custom_embedder.embed(texts, batch_size=8)
np.save(config.embeddings_dir / 'custom_embeddings.npy', custom_embeddings)
del custom_embedder
gc.collect()
logger.info(f"Step 3-4: Custom {custom_embeddings.shape} in {time.time()-t0:.1f}s")

# --- STEP 5: Clustering ---
t0 = time.time()
bc = DocumentClusterer(42)
bc.kmeans_clustering(baseline_embeddings, [3,4,5,6,7,8])
cc = DocumentClusterer(42)
cc.kmeans_clustering(custom_embeddings, [3,4,5,6,7,8])
logger.info(f"Step 5: Baseline k={bc.best_kmeans_k} sil={bc.best_kmeans_score:.4f}, "
            f"Custom k={cc.best_kmeans_k} sil={cc.best_kmeans_score:.4f} in {time.time()-t0:.1f}s")

# --- STEP 6: Evaluation ---
t0 = time.time()
ev = EmbeddingEvaluator(top_k=5)
er = ev.full_evaluation(baseline_embeddings, custom_embeddings, chunks)
er['best_custom_k'] = cc.best_kmeans_k

def c(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

with open(config.results_dir / 'evaluation_results.json', 'w') as f:
    json.dump(er, f, indent=2, default=c)
logger.info(f"Step 6: Eval done in {time.time()-t0:.1f}s, improvement={er['summary']['improvement_pct']:+.1f}%")

# --- STEP 7: Visualization ---
t0 = time.time()
viz = EmbeddingVisualizer(str(config.plots_dir))
plots = viz.generate_all_plots(baseline_embeddings, custom_embeddings, chunks, cc.kmeans_results, er)
logger.info(f"Step 7: {len(plots)} plots in {time.time()-t0:.1f}s")

# --- Summary ---
summary = {
    "documents": len(documents),
    "chunks": len(chunks),
    "baseline_best_silhouette": round(bc.best_kmeans_score, 4),
    "custom_best_silhouette": round(cc.best_kmeans_score, 4),
    "improvement_pct": round(er['summary']['improvement_pct'], 2),
    "plots": plots,
}
with open(config.results_dir / 'pipeline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("Q2 PIPELINE COMPLETE")
print("="*60)
for k, v in summary.items():
    print(f"  {k}: {v}")
