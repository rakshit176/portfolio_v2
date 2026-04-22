"""
Unit tests for src.preprocessing.text_processor - PDFTextExtractor & LegalTextChunker.

External dependencies (PyMuPDF/fitz, langchain_text_splitters) are fully mocked
so that no real PDFs are read and no internet activity occurs.
"""

import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from src.utils.config import PreprocessingConfig

from src.preprocessing.text_processor import PDFTextExtractor as _PDFTextExtractor


# ---------------------------------------------------------------------------
# PDFTextExtractor.clean_text  (static - no mocking required)
# ---------------------------------------------------------------------------


class TestCleanTextStatic:
    """Tests for PDFTextExtractor.clean_text (static method)."""

    @pytest.mark.parametrize("header", [
        "Page 1 of 42",
        "PAGE 99 OF 100",
        "Page 3 of 3",
    ])
    def test_removes_page_header_footer(self, header):
        text = f"Lorem ipsum dolor sit amet.\n{header}\nConsectetur adipiscing elit."
        result = _PDFTextExtractor.clean_text(text)
        assert header not in result
        assert "Lorem ipsum dolor sit amet." in result
        assert "Consectetur adipiscing elit." in result

    def test_removes_case_numbers(self):
        text = "Introduction\nCase 1:23-cv-00456\nBody text here."
        result = _PDFTextExtractor.clean_text(text)
        assert "Case 1:23-cv-00456" not in result
        assert "Body text here." in result

    def test_normalizes_whitespace(self):
        text = "Hello     world\n\n\n\n\nNew paragraph.\n   \nTrailing spaces."
        result = _PDFTextExtractor.clean_text(text)
        assert "     " not in result
        assert "\n\n\n" not in result

    def test_removes_non_printable(self):
        text = "Hello\x00\x01\x02 World\x07\nNew\tline"
        result = _PDFTextExtractor.clean_text(text)
        for c in ("\x00", "\x01", "\x02", "\x07"):
            assert c not in result
        assert "Hello World" in result

    def test_removes_short_uppercase_orphan_lines(self):
        text = "Important paragraph here.\nAB\nAnother paragraph here."
        result = _PDFTextExtractor.clean_text(text)
        assert "AB" not in result

    def test_removes_section_markers_like_A_dot(self):
        text = "First section content.\nA.\nSecond section content."
        result = _PDFTextExtractor.clean_text(text)
        assert "A." not in result

    def test_strips_leading_trailing_whitespace(self):
        text = "   \n\n  Hello world  \n\n   "
        result = _PDFTextExtractor.clean_text(text)
        assert result == result.strip()

    def test_preserves_longer_uppercase_lines(self):
        """Lines that are uppercase but >= 3 chars should be kept."""
        text = "THIS IS FINE\nAB\nEND"
        result = _PDFTextExtractor.clean_text(text)
        assert "THIS IS FINE" in result


# ---------------------------------------------------------------------------
# PDFTextExtractor.extract_from_pdf  (mocks fitz)
# ---------------------------------------------------------------------------


class TestPDFTextExtractorExtract:

    def test_extract_from_pdf_success(self, tmp_path):
        """Correctly extracts text from a mocked multi-page PDF."""
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.touch()  # must exist on disk for Path.exists() check

        # Build mock page objects
        mock_page_1 = MagicMock()
        mock_page_1.get_text.return_value = "Page one content."
        mock_page_2 = MagicMock()
        mock_page_2.get_text.return_value = ""

        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page_1, mock_page_2]))

        config = PreprocessingConfig()
        extractor = _PDFTextExtractor(config=config)

        with patch("src.preprocessing.text_processor.fitz") as mock_fitz:
            mock_fitz.open.return_value = mock_doc
            result = extractor.extract_from_pdf(str(fake_pdf))

        # Only page 1 had non-empty text
        assert "[PAGE 1]" in result
        assert "Page one content." in result
        assert "Page two" not in result  # page 2 was empty
        mock_fitz.open.assert_called_once_with(str(fake_pdf))

    def test_extract_from_pdf_missing_file(self):
        """Raises FileNotFoundError when the file does not exist."""
        config = PreprocessingConfig()
        extractor = _PDFTextExtractor(config=config)

        with pytest.raises(FileNotFoundError, match="PDF not found"):
            extractor.extract_from_pdf("/nonexistent/path/file.pdf")

    def test_extract_from_pdf_returns_all_pages(self, tmp_path):
        """All non-empty pages are concatenated with [PAGE N] prefix."""
        fake_pdf = tmp_path / "multi.pdf"
        fake_pdf.touch()

        mock_pages = []
        for i in range(4):
            p = MagicMock()
            p.get_text.return_value = f"Content of page {i + 1}."
            mock_pages.append(p)

        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_doc.__iter__ = MagicMock(return_value=iter(mock_pages))

        config = PreprocessingConfig()
        extractor = _PDFTextExtractor(config=config)

        with patch("src.preprocessing.text_processor.fitz") as mock_fitz:
            mock_fitz.open.return_value = mock_doc
            result = extractor.extract_from_pdf(str(fake_pdf))

        for i in range(1, 5):
            assert f"[PAGE {i}]" in result
            assert f"Content of page {i}." in result


# ---------------------------------------------------------------------------
# LegalTextChunker  (mocks langchain_text_splitters)
# ---------------------------------------------------------------------------


class TestLegalTextChunker:

    def _make_chunker(self, **config_kwargs):
        config = PreprocessingConfig(**config_kwargs)
        with patch("src.preprocessing.text_processor.RecursiveCharacterTextSplitter") as mock_splitter_cls:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = [
                "This is a reasonably long chunk of text that exceeds fifty characters in length.",
                "Second chunk of text that is also long enough to pass the minimum filter threshold.",
                "Short",  # This should be filtered out (below min_chunk_length)
            ]
            mock_splitter_cls.return_value = mock_splitter
            from src.preprocessing.text_processor import LegalTextChunker
            chunker = LegalTextChunker(config=config)
        return chunker, mock_splitter

    def test_chunk_text_creates_correct_chunks_with_metadata(self):
        """Chunks have the expected keys: text, chunk_id, source, chunk_index, char_count."""
        chunker, _ = self._make_chunker()
        chunks = chunker.chunk_text("some long input text", source_name="test.pdf")

        assert len(chunks) == 2  # "Short" filtered out
        for c in chunks:
            assert "text" in c
            assert "chunk_id" in c
            assert "source" in c
            assert "chunk_index" in c
            assert "char_count" in c
        assert chunks[0]["source"] == "test.pdf"

    def test_chunk_text_filters_short_chunks(self):
        """Chunks shorter than min_chunk_length are excluded."""
        chunker, _ = self._make_chunker(min_chunk_length=50)
        chunks = chunker.chunk_text("input", source_name="short_test.pdf")

        for c in chunks:
            assert c["char_count"] >= 50

    def test_chunk_text_truncates_very_long_chunks(self):
        """Chunks exceeding max_chunk_length are truncated."""
        long_text = "A" * 2000
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = [long_text]

        config = PreprocessingConfig(max_chunk_length=500)
        with patch("src.preprocessing.text_processor.RecursiveCharacterTextSplitter") as mock_cls:
            mock_cls.return_value = mock_splitter
            from src.preprocessing.text_processor import LegalTextChunker
            chunker = LegalTextChunker(config=config)

        chunks = chunker.chunk_text(long_text, source_name="long.pdf")
        assert len(chunks) == 1
        assert chunks[0]["char_count"] <= 500

    def test_chunk_text_generates_stable_chunk_ids(self):
        """chunk_id is deterministic (SHA-256 based)."""
        chunker, _ = self._make_chunker()
        chunks_a = chunker.chunk_text("input", source_name="deterministic.pdf")
        chunks_b = chunker.chunk_text("input", source_name="deterministic.pdf")
        assert [c["chunk_id"] for c in chunks_a] == [c["chunk_id"] for c in chunks_b]

    def test_chunk_text_different_sources_different_ids(self):
        """Different source names produce different chunk_ids."""
        chunker, _ = self._make_chunker()
        chunks_a = chunker.chunk_text("input", source_name="doc_a.pdf")
        chunks_b = chunker.chunk_text("input", source_name="doc_b.pdf")
        assert chunks_a[0]["chunk_id"] != chunks_b[0]["chunk_id"]

    def test_chunk_documents_processes_multiple_documents(self):
        """chunk_documents cleans and chunks every entry in the dict."""
        mock_splitter = MagicMock()
        # Each document produces one chunk
        mock_splitter.split_text.return_value = [
            "This is a sufficiently long piece of text for a legal document chunk."
        ]

        config = PreprocessingConfig(min_chunk_length=30)
        with patch("src.preprocessing.text_processor.RecursiveCharacterTextSplitter") as mock_cls:
            mock_cls.return_value = mock_splitter
            from src.preprocessing.text_processor import LegalTextChunker
            chunker = LegalTextChunker(config=config)

        documents = {
            "file_a.pdf": "Content from file A with some extra words to make it longer.",
            "file_b.pdf": "Content from file B with some extra words to make it longer.",
        }
        chunks = chunker.chunk_documents(documents)

        assert len(chunks) == 2
        sources = {c["source"] for c in chunks}
        assert sources == {"file_a.pdf", "file_b.pdf"}
        assert mock_splitter.split_text.call_count == 2

    def test_chunk_documents_empty_input(self):
        """Empty documents dict returns empty list."""
        mock_splitter = MagicMock()
        config = PreprocessingConfig()
        with patch("src.preprocessing.text_processor.RecursiveCharacterTextSplitter") as mock_cls:
            mock_cls.return_value = mock_splitter
            from src.preprocessing.text_processor import LegalTextChunker
            chunker = LegalTextChunker(config=config)

        assert chunker.chunk_documents({}) == []

    def test_splitter_initialized_with_config_values(self):
        """RecursiveCharacterTextSplitter receives correct chunk_size/overlap."""
        config = PreprocessingConfig(chunk_size=256, chunk_overlap=32)
        with patch("src.preprocessing.text_processor.RecursiveCharacterTextSplitter") as mock_cls:
            from src.preprocessing.text_processor import LegalTextChunker
            LegalTextChunker(config=config)
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args
            assert call_kwargs.kwargs.get("chunk_size") == 256 or call_kwargs[1].get("chunk_size") == 256
            assert call_kwargs.kwargs.get("chunk_overlap") == 32 or call_kwargs[1].get("chunk_overlap") == 32
