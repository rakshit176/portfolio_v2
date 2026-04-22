"""
Integration tests for the Q2 Custom Embedder preprocessing pipeline.

Tests the end-to-end flow: PDF extraction → cleaning → chunking, with all
external I/O mocked (no real PDFs read, no models downloaded, no network).
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.utils.config import PreprocessingConfig


# ---------------------------------------------------------------------------
# preprocess_pipeline integration test
# ---------------------------------------------------------------------------


class TestPreprocessingPipeline:

    def test_preprocessing_pipeline_with_mocked_pdfs(self, tmp_path):
        """
        Full preprocessing pipeline: extract_from_directory → clean → chunk.

        Verifies that chunks are produced with correct structure and metadata,
        without touching any real PDF file or external service.
        """
        # ---- Set up fake PDF files in a temp directory ----
        fake_dir = tmp_path / "raw"
        fake_dir.mkdir()

        # Create empty .pdf files so glob finds them
        (fake_dir / "doc_alpha.pdf").touch()
        (fake_dir / "doc_beta.pdf").touch()
        (fake_dir / "readme.txt").touch()  # non-PDF, should be ignored

        # ---- Mock fitz (PyMuPDF) to return controlled text ----
        def mock_fitz_open(path):
            doc = MagicMock()
            filename = Path(path).name

            if "alpha" in filename:
                pages_text = [
                    "Page 1 of 5\n\nThe defendant filed a motion to dismiss.",
                    "Case 1:23-cv-00456\n\nThe plaintiff argues that the "
                    "contract was breached under Section 7 of the agreement.",
                ]
            elif "beta" in filename:
                pages_text = [
                    "Page 1 of 3\n\nThis court has jurisdiction over the matter.",
                    "TABLE OF CONTENTS\n\nThe appellant seeks review of the "
                    "lower court decision under 28 U.S.C. § 1291.",
                ]
            else:
                pages_text = ["Unexpected file content."]

            mock_pages = []
            for i, text in enumerate(pages_text):
                page = MagicMock()
                page.get_text.return_value = text
                mock_pages.append(page)

            doc.__enter__ = MagicMock(return_value=doc)
            doc.__exit__ = MagicMock(return_value=False)
            doc.__iter__ = MagicMock(return_value=iter(mock_pages))
            return doc

        # ---- Mock langchain text splitter ----
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = [
            "This is a sufficiently long chunk of legal text that meets "
            "the minimum chunk length threshold for processing. "
            "It contains detailed legal arguments.",
            "This is another reasonably long chunk with enough characters "
            "to pass the filter. The court held that the statute applies.",
        ]

        config = PreprocessingConfig(
            chunk_size=512,
            chunk_overlap=64,
            min_chunk_length=50,
            max_chunk_length=1024,
        )

        # ---- Mock get_config to use our temp paths and config ----
        with (
            patch("src.preprocessing.text_processor.fitz") as mock_fitz_mod,
            patch("src.preprocessing.text_processor.RecursiveCharacterTextSplitter") as mock_splitter_cls,
            patch("src.preprocessing.text_processor.get_config") as mock_get_config,
        ):
            mock_get_config.return_value = MagicMock(preprocessing=config)
            mock_fitz_mod.open.side_effect = mock_fitz_open
            mock_splitter_cls.return_value = mock_splitter

            # Patch data_dir to point to our fake directory
            with patch.object(
                type(mock_get_config.return_value),
                "__getattr__",
                create=True,
                side_effect=lambda name: {"preprocessing": config, "data_dir": fake_dir}.get(name),
            ):
                # Import after patches applied
                from src.preprocessing.text_processor import (
                    PDFTextExtractor,
                    LegalTextChunker,
                )

                extractor = PDFTextExtractor(config=config)
                documents = extractor.extract_from_directory(fake_dir)

                # Should extract both PDFs (not readme.txt)
                assert len(documents) == 2
                assert "doc_alpha.pdf" in documents
                assert "doc_beta.pdf" in documents

                # Raw extracted text still contains headers/footers (cleaning
                # happens later in chunk_documents). Verify raw text is present.
                assert "defendant" in documents["doc_alpha.pdf"]
                assert "plaintiff" in documents["doc_alpha.pdf"]
                assert "jurisdiction" in documents["doc_beta.pdf"]

                chunker = LegalTextChunker(config=config)
                chunks = chunker.chunk_documents(documents)

                # Each document → 2 chunks from mock splitter → 4 total
                assert len(chunks) == 4

                # Verify chunk structure
                for chunk in chunks:
                    assert "text" in chunk
                    assert "chunk_id" in chunk
                    assert "source" in chunk
                    assert "chunk_index" in chunk
                    assert "char_count" in chunk
                    assert chunk["char_count"] >= 50

                # Sources are preserved
                sources = {c["source"] for c in chunks}
                assert "doc_alpha.pdf" in sources
                assert "doc_beta.pdf" in sources

    def test_preprocessing_pipeline_empty_directory(self, tmp_path):
        """Pipeline handles directory with no PDFs gracefully."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch("src.preprocessing.text_processor.get_config") as mock_get_config:
            config = PreprocessingConfig()
            mock_get_config.return_value = MagicMock(preprocessing=config)

            from src.preprocessing.text_processor import PDFTextExtractor
            extractor = PDFTextExtractor(config=config)
            result = extractor.extract_from_directory(empty_dir)

            assert result == {}

    def test_preprocessing_pipeline_handles_extraction_errors(self, tmp_path):
        """If one PDF fails to extract, others still succeed."""
        fake_dir = tmp_path / "raw"
        fake_dir.mkdir()
        (fake_dir / "good.pdf").touch()
        (fake_dir / "bad.pdf").touch()

        call_count = {"n": 0}

        def mock_fitz_open(path):
            call_count["n"] += 1
            filename = Path(path).name

            if "bad" in filename:
                raise RuntimeError("Corrupt PDF")
            else:
                doc = MagicMock()
                page = MagicMock()
                page.get_text.return_value = "Valid page content. " * 50
                doc.__enter__ = MagicMock(return_value=doc)
                doc.__exit__ = MagicMock(return_value=False)
                doc.__iter__ = MagicMock(return_value=iter([page]))
                return doc

        config = PreprocessingConfig()

        with patch("src.preprocessing.text_processor.fitz") as mock_fitz_mod:
            mock_fitz_mod.open.side_effect = mock_fitz_open

            from src.preprocessing.text_processor import PDFTextExtractor
            extractor = PDFTextExtractor(config=config)
            result = extractor.extract_from_directory(fake_dir)

            # Only the good PDF should be extracted
            assert len(result) == 1
            assert "good.pdf" in result

    def test_preprocessing_pipeline_chunking_preserves_order(self, tmp_path):
        """Chunks from the same document maintain their relative order."""
        fake_dir = tmp_path / "raw"
        fake_dir.mkdir()
        (fake_dir / "single.pdf").touch()

        doc = MagicMock()
        page = MagicMock()
        page.get_text.return_value = "Content " * 500  # long enough for multi-chunk
        doc.__enter__ = MagicMock(return_value=doc)
        doc.__exit__ = MagicMock(return_value=False)
        doc.__iter__ = MagicMock(return_value=iter([page]))

        mock_splitter = MagicMock()
        # Return 3 chunks to verify ordering
        mock_splitter.split_text.return_value = [
            "First legal argument section with enough text to pass the filter. " * 3,
            "Second legal argument section with enough text to pass the filter. " * 3,
            "Third legal argument section with enough text to pass the filter. " * 3,
        ]

        config = PreprocessingConfig(min_chunk_length=50)

        with (
            patch("src.preprocessing.text_processor.fitz") as mock_fitz_mod,
            patch("src.preprocessing.text_processor.RecursiveCharacterTextSplitter") as mock_cls,
        ):
            mock_fitz_mod.open.return_value = doc
            mock_cls.return_value = mock_splitter

            from src.preprocessing.text_processor import PDFTextExtractor, LegalTextChunker
            extractor = PDFTextExtractor(config=config)
            text = extractor.extract_from_pdf(fake_dir / "single.pdf")

            chunker = LegalTextChunker(config=config)
            chunks = chunker.chunk_text(text, source_name="single.pdf")

            assert len(chunks) == 3
            # chunk_index should be monotonically increasing
            indices = [c["chunk_index"] for c in chunks]
            assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# Full pipeline integration: preprocess -> embed -> cluster -> evaluate
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:

    """
    Integration test covering the complete pipeline flow:

        PDF extraction -> text cleaning -> chunking -> baseline embedding
        -> KMeans clustering -> full evaluation

    All heavy external dependencies (fitz, SentenceTransformer) are fully
    mocked so that no real PDFs are read, no models are downloaded, and no
    network activity occurs.
    """

    @staticmethod
    def _build_fake_documents():
        """Return synthetic documents dict (filename -> raw text)."""
        return {
            "contract.pdf": (
                "This agreement is entered into by and between the parties. "
                "The plaintiff alleges breach of contract under Section 7. "
                "Defendant filed a motion to dismiss on procedural grounds. "
                "The court shall determine jurisdiction pursuant to 28 U.S.C. "
            ) * 3,
            "opinion.pdf": (
                "The appellate court reviewed the lower court decision. "
                "We hold that the statute of limitations has expired. "
                "The appellant seeks review of the summary judgment. "
                "Judgment is reversed and the case is remanded. "
            ) * 3,
            "motion.pdf": (
                "Plaintiff moves for partial summary judgment. "
                "The undisputed facts establish liability. "
                "No genuine dispute of material fact exists. "
                "Defendant opposes the motion on evidentiary grounds. "
            ) * 3,
        }

    @staticmethod
    def _mock_chunks_from_documents(documents):
        """
        Produce a list of chunk dicts from the documents dict.
        Each document yields 2 chunks.  Chunks are long enough to pass
        min_chunk_length filters.
        """
        chunks = []
        for source_name, text in documents.items():
            sentences = text.split(". ")
            # Create 2 chunks per document by splitting sentences
            mid = len(sentences) // 2
            for part_idx, part in enumerate([sentences[:mid], sentences[mid:]]):
                chunk_text = ". ".join(part).strip()
                if len(chunk_text) < 50:
                    chunk_text = chunk_text + " " + "Additional legal text. " * 5
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": f"{source_name}_{part_idx}",
                    "source": source_name,
                    "chunk_index": part_idx,
                    "char_count": len(chunk_text),
                })
        return chunks

    def test_full_pipeline_preprocess_embed_cluster_evaluate(self, tmp_path):
        """
        End-to-end: extract -> clean -> chunk -> embed -> cluster -> evaluate.

        Verifies that evaluation_results has all expected top-level keys and
        that each component correctly hands off its output to the next.
        """
        import numpy as np
        from src.utils.config import PreprocessingConfig, EmbeddingConfig

        documents = self._build_fake_documents()
        chunks = self._mock_chunks_from_documents(documents)
        assert len(chunks) == 6  # 3 documents x 2 chunks each

        rng = np.random.RandomState(42)
        n_chunks = len(chunks)
        dim = 384

        # ----- Step 1: BaselineEmbedder (mocked SentenceTransformer) -----
        baseline_embeddings = rng.randn(n_chunks, dim).astype(np.float32)
        # Normalize so cosine similarity is well-defined
        baseline_embeddings = baseline_embeddings / np.linalg.norm(
            baseline_embeddings, axis=1, keepdims=True
        )

        # ----- Step 2: CustomEmbedder (mocked SentenceTransformer) -----
        # Make custom embeddings slightly better separated by source
        custom_embeddings = baseline_embeddings.copy()
        for i, chunk in enumerate(chunks):
            if "contract" in chunk["source"]:
                custom_embeddings[i, :10] += 1.0
            elif "opinion" in chunk["source"]:
                custom_embeddings[i, 10:20] += 1.0
            else:
                custom_embeddings[i, 20:30] += 1.0
        # Re-normalize
        custom_embeddings = custom_embeddings / np.linalg.norm(
            custom_embeddings, axis=1, keepdims=True
        )

        mock_st = MagicMock()
        mock_st.get_sentence_embedding_dimension.return_value = dim
        embed_config = EmbeddingConfig(
            baseline_model_name="all-MiniLM-L6-v2",
            custom_model_name="legal-domain-embedder-v1",
            embedding_dim=dim,
            max_seq_length=256,
            batch_size=8,
        )

        # Mock the model's save method for CustomEmbedder
        mock_st.save = MagicMock()

        call_count = {"n": 0}

        def mock_st_encode(texts, **kwargs):
            """Return pre-computed embeddings based on call order."""
            call_count["n"] += 1
            if call_count["n"] == 1:
                return baseline_embeddings
            else:
                return custom_embeddings

        mock_st.encode.side_effect = mock_st_encode

        with (
            patch("src.embedding.baseline.SentenceTransformer", return_value=mock_st) as baseline_st_cls,
            patch("src.embedding.custom_embedder.SentenceTransformer", return_value=mock_st) as custom_st_cls,
            patch("src.embedding.baseline.get_config") as baseline_gc,
            patch("src.embedding.custom_embedder.get_config") as custom_gc,
        ):
            baseline_gc.return_value = MagicMock(
                embedding=embed_config,
                models_dir=tmp_path / "models",
            )
            custom_gc.return_value = MagicMock(
                embedding=embed_config,
                models_dir=tmp_path / "models",
            )

            # -- Baseline embedding --
            from src.embedding.baseline import BaselineEmbedder
            baseline_embedder = BaselineEmbedder(config=embed_config)
            baseline_embedder.load()
            baseline_emb_result = baseline_embedder.embed(
                [c["text"] for c in chunks]
            )
            np.testing.assert_array_equal(baseline_emb_result, baseline_embeddings)

            # -- Custom embedding --
            from src.embedding.custom_embedder import CustomEmbedder
            custom_embedder = CustomEmbedder(config=embed_config)
            custom_embedder.load()
            custom_emb_result = custom_embedder.embed(
                [c["text"] for c in chunks]
            )
            np.testing.assert_array_equal(custom_emb_result, custom_embeddings)

        # Verify SentenceTransformer was instantiated for both embedders
        assert baseline_st_cls.call_count >= 1
        assert custom_st_cls.call_count >= 1

        # ----- Step 3: KMeans clustering (real sklearn, fast) -----
        from src.clustering.clusterer import DocumentClusterer
        clusterer = DocumentClusterer(random_state=42)
        kmeans_results = clusterer.kmeans_clustering(
            custom_emb_result, k_range=[2, 3]
        )

        # Should have results for k=2 and k=3
        assert 2 in kmeans_results
        assert 3 in kmeans_results
        assert kmeans_results[2]["n_clusters"] == 2
        assert kmeans_results[2]["algorithm"] == "kmeans"

        # ----- Step 4: Full evaluation -----
        from src.evaluation.evaluator import EmbeddingEvaluator
        evaluator = EmbeddingEvaluator(top_k=3)
        evaluation_results = evaluator.full_evaluation(
            baseline_emb_result, custom_emb_result, chunks
        )

        # Verify evaluation_results has all expected top-level keys
        expected_keys = {
            "baseline_similarity",
            "custom_similarity",
            "neighbor_agreement",
            "embedding_correlation",
            "summary",
        }
        assert expected_keys.issubset(set(evaluation_results.keys()))

        # Verify nested structure
        for sim_key in ("baseline_similarity", "custom_similarity"):
            sim = evaluation_results[sim_key]
            assert "within_mean" in sim
            assert "between_mean" in sim
            assert "separation_ratio" in sim
            assert isinstance(sim["within_mean"], float)
            assert isinstance(sim["between_mean"], float)

        agreement = evaluation_results["neighbor_agreement"]
        assert "mean_jaccard" in agreement
        assert "median_jaccard" in agreement
        assert "agreement_rate" in agreement

        correlation = evaluation_results["embedding_correlation"]
        assert "spearman_rho" in correlation
        assert "pearson_r" in correlation

        summary = evaluation_results["summary"]
        assert "baseline_separation_ratio" in summary
        assert "custom_separation_ratio" in summary
        assert "improvement_pct" in summary
        assert isinstance(summary["improvement_pct"], float)

    def test_pipeline_with_single_document(self):
        """
        Pipeline works end-to-end even with a single document source.
        All chunks come from one file; between-document similarity is 0.
        """
        import numpy as np
        from src.evaluation.evaluator import EmbeddingEvaluator

        chunks = [
            {"source": "only_doc.pdf", "text": f"Chunk {i} of the single document. " * 10,
             "chunk_id": f"c{i}", "chunk_index": i}
            for i in range(8)
        ]

        rng = np.random.RandomState(99)
        dim = 384
        baseline_emb = rng.randn(8, dim).astype(np.float32)
        custom_emb = rng.randn(8, dim).astype(np.float32)

        evaluator = EmbeddingEvaluator(top_k=3)
        results = evaluator.full_evaluation(baseline_emb, custom_emb, chunks)

        assert "summary" in results
        # Single source: between_mean == 0, separation_ratio == inf
        assert results["baseline_similarity"]["between_mean"] == 0.0
        assert results["custom_similarity"]["between_mean"] == 0.0
        assert results["baseline_similarity"]["separation_ratio"] == float("inf")

    def test_pipeline_chunk_count_propagates_through_all_stages(self):
        """
        Number of embeddings matches the number of chunks at every stage.
        """
        import numpy as np
        from src.evaluation.evaluator import EmbeddingEvaluator

        n_chunks = 12
        chunks = [
            {"source": f"doc_{i % 3}", "text": f"Text for chunk {i}. " * 15,
             "chunk_id": f"c{i}", "chunk_index": i}
            for i in range(n_chunks)
        ]

        rng = np.random.RandomState(7)
        dim = 384
        baseline_emb = rng.randn(n_chunks, dim).astype(np.float32)
        custom_emb = rng.randn(n_chunks, dim).astype(np.float32)

        # Cluster
        from src.clustering.clusterer import DocumentClusterer
        clusterer = DocumentClusterer(random_state=42)
        kmeans_results = clusterer.kmeans_clustering(custom_emb, k_range=[3])

        labels = kmeans_results[3]["labels"]
        assert labels.shape == (n_chunks,)

        # Evaluate
        evaluator = EmbeddingEvaluator(top_k=5)
        results = evaluator.full_evaluation(baseline_emb, custom_emb, chunks)

        assert results["summary"]["baseline_separation_ratio"] > 0
        assert results["summary"]["custom_separation_ratio"] > 0
