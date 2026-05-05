from pathlib import Path
from typing import Optional
import sys

from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    from langchain_community.document_loaders import PyPDFLoader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from .config import Config

_retriever: Optional[BM25Retriever] = None
_is_initialized: bool = False


def get_retriever(cfg: Config) -> Optional[BM25Retriever]:
    """Get or create BM25 retriever (singleton). Returns None if no docs."""
    global _retriever, _is_initialized

    if _is_initialized:
        return _retriever

    _is_initialized = True
    resources_dir = Path(cfg.resources_dir)

    if not resources_dir.exists():
        print(f"Info: '{resources_dir}' not found. Document search disabled.", file=sys.stderr)
        return None

    text_documents: list[Document] = []
    pdf_documents: list[Document] = []

    # Load .txt files one by one
    for path in resources_dir.rglob("*.txt"):
        try:
            text_documents.extend(TextLoader(str(path), encoding="utf-8").load())
        except Exception as e:
            print(
                f"Warning: Error loading file {path}: {type(e).__name__}: {e}",
                file=sys.stderr,
            )

    # Load .pdf files one by one, but combine all pages into one document per PDF
    if HAS_PYPDF:
        for path in resources_dir.rglob("*.pdf"):
            try:
                pages = PyPDFLoader(str(path)).load()
                full_text = "\n\n".join(
                    page.page_content.strip()
                    for page in pages
                    if page.page_content and page.page_content.strip()
                )

                if full_text:
                    pdf_documents.append(
                        Document(
                            page_content=full_text,
                            metadata={"source": str(path)},
                        )
                    )
            except Exception as e:
                print(
                    f"Warning: Error loading PDF {path}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
    else:
        print("Info: PyPDF not available. PDF loading disabled.", file=sys.stderr)

    if not text_documents and not pdf_documents:
        print(f"Info: No documents in '{resources_dir}'. Search disabled.", file=sys.stderr)
        return None

    # Chunk only text documents; keep PDFs whole so recipes stay intact
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    text_chunks = splitter.split_documents(text_documents)

    all_documents = text_chunks + pdf_documents

    _retriever = BM25Retriever.from_documents(all_documents, k=20)

    _retriever = BM25Retriever.from_documents(all_documents, k=10)
    print(
        f"Loaded {len(all_documents)} searchable units "
        f"({len(text_chunks)} text chunks, {len(pdf_documents)} full PDFs)",
        file=sys.stderr,
    )

    return _retriever


def reset_retriever() -> None:
    """Reset singleton (for testing or after uploading new files)."""
    global _retriever, _is_initialized
    _retriever = None
    _is_initialized = False
