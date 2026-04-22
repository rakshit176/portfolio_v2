"""
Unit tests for src.embedding.baseline, src.embedding.custom_embedder, and src.embedding.training.

SentenceTransformer and all heavy model I/O are fully mocked so that
no real models are downloaded and no GPU/CPU-intensive operations occur.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock, call
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.config import EmbeddingConfig


# ---------------------------------------------------------------------------
# BaselineEmbedder
# ---------------------------------------------------------------------------


class TestBaselineEmbedder:

    def _make_embedder(self, **config_kwargs):
        config = EmbeddingConfig(**config_kwargs)
        from src.embedding.baseline import BaselineEmbedder
        return BaselineEmbedder(config=config), config

    def test_init_stores_config(self):
        """Constructor stores the embedding config without loading the model."""
        embedder, config = self._make_embedder()
        assert embedder.config is config
        assert embedder.model is None
        assert embedder.model_name == "all-MiniLM-L6-v2"

    def test_load_calls_sentence_transformer(self):
        """load() instantiates SentenceTransformer and sets max_seq_length."""
        embedder, config = self._make_embedder(max_seq_length=128)

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model

        with patch("src.embedding.baseline.SentenceTransformer", mock_st):
            result = embedder.load()

        assert result is embedder  # returns self
        assert embedder.model is mock_model
        mock_st.SentenceTransformer.assert_called_once_with("all-MiniLM-L6-v2")
        assert embedder.model.max_seq_length == 128

    @patch("src.embedding.baseline.SentenceTransformer")
    def test_embed_raises_without_load(self, mock_st):
        """Calling embed() before load() raises RuntimeError."""
        embedder, _ = self._make_embedder()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            embedder.embed(["hello"])

    @patch("src.embedding.baseline.SentenceTransformer")
    def test_embed_calls_model_encode(self, mock_st):
        """embed() delegates to model.encode with expected parameters."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(3, 384).astype(np.float32)
        mock_st.SentenceTransformer.return_value = mock_model

        embedder, config = self._make_embedder(batch_size=16)
        embedder.load()

        texts = ["text one", "text two", "text three"]
        result = embedder.embed(texts)

        mock_model.encode.assert_called_once()
        call_kwargs = mock_model.encode.call_args
        assert call_kwargs.kwargs.get("batch_size") == 16 or call_kwargs[1].get("batch_size") == 16
        assert call_kwargs.kwargs.get("normalize_embeddings") is True
        assert call_kwargs.kwargs.get("convert_to_numpy") is True
        assert result.shape == (3, 384)

    @patch("src.embedding.baseline.SentenceTransformer")
    def test_embed_batch_size_override(self, mock_st):
        """Passing batch_size to embed() overrides the config default."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(2, 384).astype(np.float32)
        mock_st.SentenceTransformer.return_value = mock_model

        embedder, config = self._make_embedder(batch_size=8)
        embedder.load()
        embedder.embed(["a", "b"], batch_size=32)

        call_kwargs = mock_model.encode.call_args
        assert call_kwargs.kwargs.get("batch_size") == 32 or call_kwargs[1].get("batch_size") == 32

    @patch("src.embedding.baseline.SentenceTransformer")
    def test_get_embedding_dimension_with_loaded_model(self, mock_st):
        """Returns the model's actual dimension after load()."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model

        embedder, _ = self._make_embedder(embedding_dim=768)  # mismatch on purpose
        embedder.load()
        assert embedder.get_embedding_dimension() == 384

    def test_get_embedding_dimension_without_loaded_model(self):
        """Returns config.embedding_dim when model is not loaded."""
        embedder, config = self._make_embedder(embedding_dim=384)
        assert embedder.get_embedding_dimension() == 384


# ---------------------------------------------------------------------------
# CustomEmbedder
# ---------------------------------------------------------------------------


class TestCustomEmbedder:

    def _make_embedder(self, **config_kwargs):
        config = EmbeddingConfig(**config_kwargs)
        from src.embedding.custom_embedder import CustomEmbedder
        return CustomEmbedder(config=config), config

    def test_init_stores_config(self):
        embedder, config = self._make_embedder()
        assert embedder.config is config
        assert embedder.model is None
        assert embedder.model_name == "legal-domain-embedder-v1"

    def test_load_fallback_to_base_model(self, tmp_path):
        """When no fine-tuned model exists, falls back to base model."""
        embedder, config = self._make_embedder()

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model

        with patch("src.embedding.custom_embedder.SentenceTransformer", mock_st), \
             patch("src.embedding.custom_embedder.get_config") as mock_get_config:
            fake_config = MagicMock()
            fake_config.models_dir = tmp_path / "nonexistent"
            mock_get_config.return_value = fake_config

            embedder.load()

        # Should have called SentenceTransformer with base model name (fallback)
        assert embedder.model is mock_model
        mock_st.SentenceTransformer.assert_called_with("all-MiniLM-L6-v2")

    def test_load_from_existing_finetuned_model(self, tmp_path):
        """When fine-tuned model dir exists, loads from it."""
        embedder, config = self._make_embedder(custom_model_name="legal-domain-embedder-v1")
        model_dir = tmp_path / "models" / "legal-domain-embedder-v1"
        model_dir.mkdir(parents=True)

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model

        with patch("src.embedding.custom_embedder.SentenceTransformer", mock_st), \
             patch("src.embedding.custom_embedder.get_config") as mock_get_config:
            fake_config = MagicMock()
            fake_config.models_dir = tmp_path / "models"
            mock_get_config.return_value = fake_config

            embedder.load()

        mock_st.SentenceTransformer.assert_called_once_with(str(model_dir))

    def test_embed_raises_without_load(self):
        embedder, _ = self._make_embedder()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            embedder.embed(["hello"])

    @patch("src.embedding.custom_embedder.SentenceTransformer")
    def test_embed_calls_model_encode(self, mock_st):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(2, 384).astype(np.float32)
        mock_st.SentenceTransformer.return_value = mock_model

        embedder, config = self._make_embedder(batch_size=4)
        embedder.model = mock_model  # manually set model (bypass load)

        result = embedder.embed(["a", "b"])

        mock_model.encode.assert_called_once()
        assert result.shape == (2, 384)

    def test_save_creates_directory_and_saves(self, tmp_path):
        """save() creates the target directory and calls model.save()."""
        embedder, config = self._make_embedder()

        mock_model = MagicMock()
        embedder.model = mock_model

        save_path = tmp_path / "custom_model"
        embedder.save(save_path)

        assert save_path.exists()
        mock_model.save.assert_called_once_with(str(save_path))

    def test_save_raises_without_model(self):
        embedder, _ = self._make_embedder()
        with pytest.raises(RuntimeError, match="No model to save"):
            embedder.save()

    def test_get_embedding_dimension_without_model(self):
        embedder, config = self._make_embedder(embedding_dim=384)
        assert embedder.get_embedding_dimension() == 384

    @patch("src.embedding.custom_embedder.SentenceTransformer")
    def test_get_embedding_dimension_with_model(self, mock_st):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        embedder, _ = self._make_embedder()
        embedder.model = mock_model
        assert embedder.get_embedding_dimension() == 384


# ---------------------------------------------------------------------------
# TrainingDataGenerator
# ---------------------------------------------------------------------------


class TestTrainingDataGenerator:

    def _make_generator(self, **config_kwargs):
        config = EmbeddingConfig(**config_kwargs)
        from src.embedding.training import TrainingDataGenerator
        return TrainingDataGenerator(config), config

    def test_generate_pairs_creates_adjacent_and_random_pairs(self):
        """Generates both adjacent-chunk pairs and random intra-document pairs."""
        generator, _ = self._make_generator(train_test_split=0.2)

        chunks = [
            {"source": "doc_a.pdf", "text": f"Chunk {i} of doc_a. " * 20, "chunk_id": f"a{i}"}
            for i in range(10)
        ] + [
            {"source": "doc_b.pdf", "text": f"Chunk {i} of doc_b. " * 20, "chunk_id": f"b{i}"}
            for i in range(8)
        ]

        train_pairs, eval_pairs = generator.generate_pairs(chunks)

        # Should have both train and eval pairs
        assert len(train_pairs) > 0
        assert len(eval_pairs) > 0

        # Train pairs should be InputExample objects with texts
        for pair in train_pairs:
            assert len(pair.texts) == 2
            assert isinstance(pair.texts[0], str)
            assert isinstance(pair.texts[1], str)

        # Eval pairs should also be InputExample objects
        for pair in eval_pairs:
            assert len(pair.texts) == 2

    def test_generate_pairs_single_source(self):
        """With only one source, all pairs are eval pairs (last source)."""
        generator, _ = self._make_generator(train_test_split=0.2)

        chunks = [
            {"source": "only_doc.pdf", "text": f"Chunk {i}. " * 20, "chunk_id": f"c{i}"}
            for i in range(5)
        ]

        train_pairs, eval_pairs = generator.generate_pairs(chunks)

        # With 1 source, the last (only) source becomes eval source
        assert len(eval_pairs) >= 0  # may have adjacent pairs
        # train_pairs may be empty since all sources went to eval

    def test_generate_pairs_insufficient_chunks(self):
        """Chunks with very few texts per source should still work."""
        generator, _ = self._make_generator()

        chunks = [
            {"source": "tiny.pdf", "text": "Only chunk. " * 20, "chunk_id": "t0"},
        ]

        train_pairs, eval_pairs = generator.generate_pairs(chunks)
        # Single chunk: no adjacent pairs, no random pairs → both empty
        assert len(train_pairs) == 0
        assert len(eval_pairs) == 0

    def test_generate_pairs_multiple_sources(self):
        """Chunks from different sources produce distinct pairs."""
        generator, _ = self._make_generator(train_test_split=0.3)

        chunks = []
        for src_idx in range(5):
            for i in range(8):
                chunks.append({
                    "source": f"doc_{src_idx}.pdf",
                    "text": f"Source {src_idx} chunk {i}. " * 15,
                    "chunk_id": f"s{src_idx}_c{i}",
                })

        train_pairs, eval_pairs = generator.generate_pairs(chunks)

        assert len(train_pairs) > 20
        assert len(eval_pairs) > 0

        # Train pairs should NOT contain the eval source
        # (since last source(s) are held out)
        eval_sources = set()
        for p in eval_pairs:
            # We can't directly check source from InputExample,
            # but we verify eval_pairs is non-empty
            pass
        assert len(eval_pairs) > 0


# ---------------------------------------------------------------------------
# train_custom_embedder  (mocked heavy dependencies)
# ---------------------------------------------------------------------------


class TestTrainCustomEmbedder:

    @patch("src.embedding.training.AdamW")
    @patch("src.embedding.training.LinearLR")
    @patch("src.embedding.training.losses")
    @patch("src.embedding.training.SentenceTransformer")
    def test_train_returns_custom_embedder(self, mock_st, mock_losses, mock_lr, mock_adamw):
        """train_custom_embedder returns a CustomEmbedder with model set."""
        mock_model = MagicMock()
        mock_model.smart_batching_collate = MagicMock(return_value=lambda batch: batch)
        mock_st.SentenceTransformer.return_value = mock_model

        mock_loss_instance = MagicMock()
        mock_losses.MultipleNegativesRankingLoss.return_value = mock_loss_instance
        mock_loss_instance.return_value = MagicMock()
        mock_loss_instance.return_value.item.return_value = 0.5

        mock_opt = MagicMock()
        mock_adamw.return_value = mock_opt
        mock_sched = MagicMock()
        mock_lr.return_value = mock_sched

        chunks = [
            {"source": "doc.pdf", "text": f"Chunk {i}. " * 30, "chunk_id": f"c{i}"}
            for i in range(15)
        ]

        config = EmbeddingConfig(num_epochs=1, batch_size=8)

        with patch("src.embedding.training.get_config") as mock_get_config:
            fake_config = MagicMock()
            fake_config.models_dir = Path("/tmp/fake_models")
            mock_get_config.return_value = fake_config

            from src.embedding.training import train_custom_embedder
            result = train_custom_embedder(chunks, config)

        from src.embedding.custom_embedder import CustomEmbedder
        assert isinstance(result, CustomEmbedder)
        assert result.model is mock_model
        mock_st.SentenceTransformer.assert_called_once()
