"""Split loaded documents into chunks, embed them, and persist to a local Chroma index.

Run this once (and again whenever the documents change) before using the assistant:
    python -m src.ingest.build_index
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingest.load_documents import load_all_documents

load_dotenv()

PERSIST_DIR = Path(__file__).resolve().parents[2] / "chroma_db"
COLLECTION_NAME = "ironstore_docs"


def build_index():
  print("Loading documents...")
  documents = load_all_documents()
  print(f"Loaded {len(documents)} pages.")

  # Chunk size/overlap tuned for policy prose and catalog tables: small enough
  # for precise retrieval, with overlap so a fact split across a chunk boundary
  # isn't lost entirely.
  splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
  chunks = splitter.split_documents(documents)
  print(f"Split into {len(chunks)} chunks.")

  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

  print("Embedding and persisting to Chroma (this calls the OpenAI API)...")
  Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=str(PERSIST_DIR),
  )
  print(f"Index built and saved to {PERSIST_DIR}")


if __name__ == "__main__":
  build_index()
