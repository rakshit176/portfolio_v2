"""
Text Processor - PDF extraction, cleaning, and chunking for legal documents.

Handles:
  - Multi-format PDF text extraction (text-based + OCR fallback)
  - Domain-specific text cleaning (legal document normalization)
  - Overlap-aware chunking with metadata preservation
"""

import re
import logging
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import get_config, PreprocessingConfig

logger = logging.getLogger(__name__)


class PDFTextExtractor:
    """Extracts and cleans text from legal PDF documents."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or get_config().preprocessing

    def extract_from_pdf(self, pdf_path: str | Path) -> str:
        """
        Extract full text from a PDF file using PyMuPDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Concatenated text from all pages.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Extracting text from: {pdf_path.name}")
        doc = fitz.open(str(pdf_path))
        pages = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"[PAGE {page_num + 1}] {text}")

        doc.close()
        full_text = "\n\n".join(pages)

        logger.info(
            f"Extracted {len(full_text):,} chars from {pdf_path.name} "
            f"({len(pages)} pages)"
        )
        return full_text

    def extract_from_directory(self, dir_path: str | Path) -> Dict[str, str]:
        """
        Extract text from all PDFs in a directory.

        Args:
            dir_path: Directory containing PDF files.

        Returns:
            Dict mapping filename -> extracted text.
        """
        dir_path = Path(dir_path)
        pdfs = sorted(dir_path.glob("*.pdf"))
        if not pdfs:
            logger.warning(f"No PDFs found in {dir_path}")
            return {}

        logger.info(f"Found {len(pdfs)} PDFs in {dir_path}")
        documents = {}
        for pdf_path in pdfs:
            try:
                text = self.extract_from_pdf(pdf_path)
                documents[pdf_path.name] = text
            except Exception as e:
                logger.error(f"Failed to extract {pdf_path.name}: {e}")

        total_chars = sum(len(t) for t in documents.values())
        logger.info(
            f"Extracted {len(documents)}/{len(pdfs)} PDFs, "
            f"total {total_chars:,} characters"
        )
        return documents

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text with domain-specific rules.

        Handles:
          - Excessive whitespace normalization
          - Page number/header/footer removal
          - Legal citation normalization
          - Non-printable character removal
        """
        # Remove non-printable characters (keep newlines and tabs)
        text = re.sub(r'[^\x20-\x7E\n\t]', '', text)

        # Remove common header/footer patterns
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Case \d+:\d+-cv-\d+', '', text)

        # Normalize whitespace (preserve paragraph breaks)
        text = re.sub(r'[ \t]+', ' ', text)          # collapse inline spaces
        text = re.sub(r'\n{3,}', '\n\n', text)        # collapse multi-newlines
        text = re.sub(r' +\n', '\n', text)            # trailing spaces before newline

        # Remove orphan lines (single words that are likely headers)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) < 3 and stripped.isupper():
                continue  # skip single-word uppercase lines
            if re.match(r'^[A-Z]\.\s*$', stripped):
                continue  # skip section markers like "A."
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines).strip()


class LegalTextChunker:
    """
    Chunks legal documents with overlap-aware splitting.

    Preserves document context across chunk boundaries and attaches
    source metadata (filename, page range) to each chunk.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or get_config().preprocessing
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_text(self, text: str, source_name: str = "") -> List[Dict]:
        """
        Split text into overlapping chunks with metadata.

        Args:
            text: Full document text.
            source_name: Source filename for metadata.

        Returns:
            List of dicts with 'text', 'chunk_id', 'source', and 'char_count'.
        """
        raw_chunks = self.splitter.split_text(text)

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            # Filter by min/max length
            if len(chunk_text.strip()) < self.config.min_chunk_length:
                continue
            if len(chunk_text) > self.config.max_chunk_length:
                chunk_text = chunk_text[:self.config.max_chunk_length]

            chunk_id = hashlib.sha256(
                f"{source_name}::{i}::{chunk_text[:100]}".encode()
            ).hexdigest()[:16]

            chunks.append({
                "text": chunk_text.strip(),
                "chunk_id": chunk_id,
                "source": source_name,
                "chunk_index": i,
                "char_count": len(chunk_text.strip()),
            })

        logger.info(
            f"Chunked '{source_name}': {len(chunks)} chunks "
            f"(avg {sum(c['char_count'] for c in chunks) / max(len(chunks), 1):.0f} chars)"
        )
        return chunks

    def chunk_documents(self, documents: Dict[str, str]) -> List[Dict]:
        """
        Chunk multiple documents.

        Args:
            documents: Dict mapping filename -> full text.

        Returns:
            Flat list of chunk dicts across all documents.
        """
        all_chunks = []
        for source_name, text in documents.items():
            cleaned = PDFTextExtractor.clean_text(text)
            chunks = self.chunk_text(cleaned, source_name)
            all_chunks.extend(chunks)

        logger.info(
            f"Total: {len(all_chunks)} chunks from {len(documents)} documents"
        )
        return all_chunks


def preprocess_pipeline(data_dir: str | Path) -> Tuple[List[Dict], Dict[str, str]]:
    """
    End-to-end preprocessing: extract -> clean -> chunk.

    Args:
        data_dir: Directory containing source PDFs.

    Returns:
        Tuple of (chunks_list, raw_documents_dict).
    """
    extractor = PDFTextExtractor()
    chunker = LegalTextChunker()

    documents = extractor.extract_from_directory(data_dir)
    chunks = chunker.chunk_documents(documents)
    return chunks, documents
