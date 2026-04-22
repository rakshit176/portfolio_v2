# Q2 - Custom Embedder for Domain Documents

A full pipeline that fine-tunes a sentence-transformer model on legal domain PDFs and evaluates whether domain-specific embeddings produce better clustering than a general-purpose baseline.

The pipeline covers five stages:

1. **Preprocessing** — Extract text from legal PDFs, clean it, and split into semantically meaningful chunks.
2. **Embedding** — Generate vectors using `all-MiniLM-L6-v2` (baseline) and a fine-tuned legal-domain model.
3. **Clustering** — Apply KMeans and HDBSCAN to both embedding spaces.
4. **Evaluation** — Compare baseline vs. custom embeddings via similarity metrics, silhouette scores, neighbor overlap, and embedding correlation.
5. **Visualization** — t-SNE scatter plots, silhouette plots, and cluster-assignment maps.

---

## Quick Start

```bash
# Local run (Python 3.11+)
cd q2_custom_embedder
pip install -r requirements.txt
python main.py

# Docker run
docker build -t q2-custom-embedder .
docker run --rm -v ./outputs:/app/outputs q2-custom-embedder
```

Outputs are written to `outputs/` (embeddings, plots, evaluation JSON).

---

## Project Structure

```
q2_custom_embedder/
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── text_processor.py      # PDF extraction, cleaning, chunking
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── baseline.py            # all-MiniLM-L6-v2 baseline
│   │   ├── custom_embedder.py     # Fine-tuned legal embedder
│   │   └── training.py            # MNRL fine-tuning
│   ├── clustering/
│   │   ├── __init__.py
│   │   └── clusterer.py           # KMeans + HDBSCAN
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py           # Similarity, silhouette, correlation
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plotter.py             # t-SNE, silhouette, cluster maps
│   └── utils/
│       ├── __init__.py
│       └── config.py              # Centralized configuration
├── tests/
│   ├── unit/
│   └── integration/
├── data/raw/                      # Source PDFs (5 legal docs)
├── outputs/                       # Generated embeddings, plots, models
├── main.py                        # Entry point
├── requirements.txt
└── Dockerfile
```

---

## Architecture

```
┌─────────────────┐
│  data/raw/*.pdf │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│   Preprocessing     │  PDF extraction → text cleaning → chunking
│  text_processor.py  │
└────────┬────────────┘
         │  chunks[]
         ▼
┌─────────────────────┐      ┌─────────────────────┐
│  Baseline Embedder  │      │   Training (MNRL)   │
│  all-MiniLM-L6-v2   │      │  training.py        │
└────────┬────────────┘      └────────┬────────────┘
         │                            │
         │                    ┌───────▼────────────┐
         │                    │ Custom Embedder     │
         │                    │ custom_embedder.py  │
         │                    └────────┬────────────┘
         │                             │
         ▼                             ▼
┌──────────────────────────────────────────────┐
│  Clustering                                  │
│  clusterer.py                                │
│  • KMeans  (sweep k, pick best silhouette)   │
│  • HDBSCAN (density-based, noise detection)  │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  Evaluation                                 │
│  evaluator.py                               │
│  • Intra-cluster similarity                 │
│  • Inter-cluster separation                 │
│  • Silhouette score                         │
│  • K-neighbor agreement                     │
│  • Embedding-space correlation (CKA)        │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  Visualization                              │
│  plotter.py                                 │
│  • t-SNE scatter (baseline vs custom)       │
│  • Silhouette bar chart                     │
│  • Cluster assignment map by source doc     │
└──────────────────────────────────────────────┘
```

---

## Approach Followed

### Preprocessing
- **PDF extraction** via PyMuPDF (`fitz`); OCR fallback with Tesseract for scanned pages.
- **Cleaning**: strip boilerplate headers/footers, normalise whitespace, remove non-alpha noise.
- **Chunking**: recursive character splitting with overlap to preserve semantic context at boundaries.

### Embedding
- **Baseline**: `sentence-transformers/all-MiniLM-L6-v2` — a compact 384-dim general-purpose model.
- **Custom**: The same base model fine-tuned on the legal corpus using **Multiple Negatives Ranking Loss (MNRL)**, which treats each chunk within the same document as a positive pair and all other chunks in the batch as negatives. This pulls together semantically related legal text while pushing apart unrelated content.

### Clustering
- **KMeans**: Sweep a range of *k* values, select the *k* that maximises the mean silhouette coefficient.
- **HDBSCAN**: Density-based clustering with configurable `min_cluster_size` and `min_samples`; automatically labels noise points.

### Evaluation Metrics
| Metric | What it measures |
|---|---|
| Silhouette Score | Cohesion vs. separation of clusters |
| Intra-cluster Similarity | Average cosine similarity among points in the same cluster |
| Inter-cluster Separation | Average cosine distance between cluster centroids |
| K-Neighbor Agreement | Fraction of shared top-*k* neighbors between baseline and custom spaces |
| Embedding Correlation (CKA) | Linear Kernel CKA between the two embedding matrices |

### Visualization
- **t-SNE** 2-D projections coloured by cluster label and source document.
- **Silhouette** bar charts for direct comparison.
- **Cluster maps** showing which source documents dominate each cluster — useful for assessing topical coherence.

---

## Key Findings

The pipeline was executed on 5 legal domain PDFs producing 688 text chunks.

| Metric | Baseline (MiniLM) | Custom (Fine-tuned) | Improvement |
|---|---|---|---|
| Best Silhouette Score (k=5) | 0.0760 | 0.0696 | -8.4% |
| Within-Doc Similarity | 0.3631 | 0.4170 | +14.8% |
| Between-Doc Similarity | 0.2923 | 0.3589 | +22.8% |
| Separation Ratio | 1.2423 | 1.1617 | -6.5% |
| K-Neighbor Agreement (Jaccard) | 0.4638 | — | — |
| Embedding Correlation (Spearman) | 0.8642 | — | — |

The custom embedder achieves higher absolute within-document similarity (+14.8%), indicating it learns to place chunks from the same document closer together. However, the separation ratio (within/between) shows a slight decrease (-6.5%) because the fine-tuning also increases between-document similarity. This suggests that with only 1 training epoch and 5 source documents, the model begins to learn domain structure but overfits to general legal language patterns. The high Spearman correlation (0.86) confirms the custom embeddings preserve the global rank-order structure of the baseline while shifting the similarity distribution upward.

See `outputs/results/evaluation_results.json` and `outputs/results/pipeline_summary.json` for the full numbers.

---

## Requirements

| Category | Libraries |
|---|---|
| ML / Embeddings | `torch >= 2.0.0`, `transformers >= 4.36.0`, `sentence-transformers >= 2.2.2` |
| NLP / Text | `langchain >= 0.1.0`, `nltk >= 3.8.0` |
| PDF Processing | `PyMuPDF >= 1.23.0` |
| Clustering / Eval | `scikit-learn >= 1.3.0`, `hdbscan >= 0.8.33`, `scipy >= 1.11.0` |
| Visualization | `matplotlib >= 3.7.0`, `numpy >= 1.24.0` |
| Testing | `pytest >= 7.4.0`, `pytest-cov >= 4.1.0`, `pytest-mock >= 3.11.0` |

Install everything at once:

```bash
pip install -r requirements.txt
```

> **System dependencies** (needed for OCR): `tesseract-ocr`, `poppler-utils`. These are installed automatically in the Docker image.

---

## How to Run Tests

```bash
# Run all tests with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Run a specific test file
pytest tests/unit/test_text_processor.py -v
```
