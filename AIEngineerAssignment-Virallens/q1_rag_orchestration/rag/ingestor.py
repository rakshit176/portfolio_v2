# rag/ingestor.py
import os
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from rag.chunker import RecursiveChunker
from utils.providers import get_embeddings
from utils.logger import get_logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

logger = get_logger("ingestor")

COLLECTION_NAME = "documents"


def load_pdfs_from_directory(directory: str) -> List:
    """Load all PDFs from a directory."""
    docs = []
    dir_path = Path(directory)

    if not dir_path.exists():
        logger.warning("directory_not_found", extra={"directory": directory})
        return docs

    for pdf_file in dir_path.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = pdf_file.name
            docs.extend(pages)
            logger.info("pdf_loaded", extra={"file": pdf_file.name, "pages": len(pages)})
        except Exception as e:
            logger.error("pdf_load_error", extra={"file": pdf_file.name, "error": str(e)})

    return docs


async def ingest_documents(
    data_dir: str = "./data",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    qdrant_url: str = None,
    recreate: bool = False
):
    """Ingest documents into Qdrant vector store.

    Args:
        data_dir: Directory containing PDF files
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        qdrant_url: Qdrant URL (defaults to env QDRANT_URL)
        recreate: If True, delete and recreate collection

    Returns:
        QdrantVectorStore instance
    """
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6334")

    # Load documents
    logger.info("ingestion_start", extra={"data_dir": data_dir})
    documents = load_pdfs_from_directory(data_dir)

    if not documents:
        logger.warning("no_documents_found")
        return None

    # Chunk documents
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.split_documents(documents)

    logger.info("documents_chunked", extra={"chunks": len(chunks)})

    # Get embeddings
    embeddings = get_embeddings()

    # Connect to Qdrant
    client = QdrantClient(url=qdrant_url)

    # Create collection if needed
    if recreate:
        try:
            client.delete_collection(collection_name=COLLECTION_NAME)
            logger.info("collection_deleted", extra={"collection": COLLECTION_NAME})
        except Exception:
            pass

    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    except Exception:
        pass  # Collection exists

    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    # Add documents
    vector_store.add_documents(chunks)

    logger.info("ingestion_complete", extra={"documents": len(chunks)})

    return vector_store
