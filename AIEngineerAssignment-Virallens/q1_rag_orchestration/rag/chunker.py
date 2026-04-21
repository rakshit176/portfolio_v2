# rag/chunker.py
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveChunker:
    """Split documents into overlapping chunks for RAG."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        return self.splitter.split_text(text)

    def split_documents(self, documents: List) -> List:
        """Split documents into chunks."""
        return self.splitter.split_documents(documents)
