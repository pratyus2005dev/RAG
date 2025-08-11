import os
import tempfile
from typing import List
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
import faiss
import numpy as np

client = OpenAI(api_key=os.getenv(""))

# Store FAISS index and metadata globally
faiss_index = None
documents_metadata = []

def extract_text_from_file(file_path: str) -> str:
    """Extract text from PDF, Excel, or TXT files."""
    ext = file_path.split(".")[-1].lower()
    # The following block is redundant and uses undefined variables; it should be removed.

import os
import argparse
from pathlib import Path
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def load_documents(input_dir: Path):
    docs = []
    for file_path in input_dir.glob("**/*"):
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs.extend(loader.load())
        elif file_path.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs.extend(loader.load())
        else:
            print(f"Skipping unsupported file type: {file_path}")
    return docs


def chunk_documents(documents, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def main(input_dir: str, persist_dir: str):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required")

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Load documents
    print(f"Loading documents from {input_dir}...")
    documents = load_documents(input_path)
    print(f"Loaded {len(documents)} documents")

    # Chunk documents
    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # The following block was removed because file extraction is handled by load_documents and chunk_documents.


def embed_text(text: str):
    """Create an embedding for the given text using OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def add_to_faiss(texts: List[str], source: str):
    """Add multiple texts to the FAISS index."""
    global faiss_index, documents_metadata

    vectors = [embed_text(t) for t in texts]
    vectors = [v for v in vectors if v]

    if faiss_index is None:
        faiss_index = faiss.IndexFlatL2(len(vectors[0]))

    faiss_index.add(np.array(vectors, dtype="float32"))
    documents_metadata.extend([{"text": t, "source": source} for t in texts])


def search_faiss(query: str, k: int = 3):
    """Search FAISS index for most similar documents."""
    global faiss_index, documents_metadata

    if faiss_index is None:
        return []

    query_vector = np.array([embed_text(query)], dtype="float32")
    distances, indices = faiss_index.search(query_vector, k)

    results = []
    for idx in indices[0]:
        if idx < len(documents_metadata):
            results.append(documents_metadata[idx]["text"])

    return results
