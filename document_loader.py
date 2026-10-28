"""
document_loader.py
-------------------
Stage 1 & 2 of the pipeline: LOAD raw files off disk, then SPLIT them
into overlapping text chunks that are small enough to embed meaningfully.

Why chunk at all? Embedding an entire 50-page PDF as one vector would
average out its meaning into mush. Chunking keeps each vector focused
on one coherent idea, so similarity search actually works.
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_single_document(file_path: str) -> List[Document]:
    """
    Pick the right LangChain loader based on file extension and load it.
    Each loader returns a list of Document objects (LangChain's standard
    unit: page_content + metadata like source file / page number).
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    documents = loader.load()

    # Tag every chunk with its originating filename so we can cite sources later.
    for doc in documents:
        doc.metadata["source_file"] = os.path.basename(file_path)

    return documents


def load_all_documents(directory: str) -> List[Document]:
    """Load every supported file found in `directory`."""
    all_docs: List[Document] = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if not os.path.isfile(file_path):
            continue
        try:
            all_docs.extend(load_single_document(file_path))
        except ValueError:
            # Skip files we don't know how to parse instead of crashing the whole batch.
            print(f"Skipping unsupported file: {filename}")
    return all_docs


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Break loaded documents into overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    then sentences, then words -- only falling back to a hard character cut
    as a last resort. This keeps chunks semantically coherent.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def process_documents(directory: str) -> List[Document]:
    """Convenience wrapper: load every file in `directory` and chunk it. Returns chunks ready for embedding."""
    raw_docs = load_all_documents(directory)
    if not raw_docs:
        return []
    chunks = split_documents(raw_docs)
    return chunks
