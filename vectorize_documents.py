import os
import logging
import torch
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKING_DIR, "data")
VECTOR_DB_DIR = os.path.join(WORKING_DIR, "vector_db_dir")
COLLECTION_NAME = "rag_documents"

# Detect GPU automatically
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device.upper()}")

# Shared embeddings instance — imported by main.py
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)


def load_documents():
    """Load all PDFs from the data directory with error resilience."""
    logger.info(f"Loading PDFs from: {DATA_DIR}")
    loader = PyPDFDirectoryLoader(DATA_DIR, silent_errors=True)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages from PDFs")
    return docs


def split_documents(documents):
    """Split documents into chunks with overlap, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} text chunks")
    return chunks


def build_vectorstore(chunks):
    """Create or update the Chroma vector store idempotently."""
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME,
    )
    logger.info(f"Vector store saved to: {VECTOR_DB_DIR}")
    return vectordb


def load_vectorstore():
    """Load an existing vector store from disk."""
    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def vectorstore_exists():
    """Check if a vector store has already been built."""
    return os.path.exists(VECTOR_DB_DIR) and any(
        f.endswith(".sqlite3") for f in os.listdir(VECTOR_DB_DIR)
    )


if __name__ == "__main__":
    if vectorstore_exists():
        logger.info("Vector store already exists. Skipping re-vectorization.")
        logger.info("Delete 'vector_db_dir/' to force a rebuild.")
    else:
        documents = load_documents()
        if not documents:
            logger.error("No documents found. Add PDFs to the 'data/' folder.")
        else:
            chunks = split_documents(documents)
            build_vectorstore(chunks)
            logger.info("Documents vectorized successfully.")