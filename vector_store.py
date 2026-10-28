"""
vector_store.py
----------------
Stage 3 & 4 of the pipeline: EMBED each chunk into a vector, and STORE
those vectors in a FAISS index so we can later do fast similarity search
over them.

FAISS (Facebook AI Similarity Search) keeps everything in a local file
on disk -- no external vector DB service needed to get started.
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import config

# The embedding model is expensive to load, so we create it once and reuse it.
_embeddings = None


def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Lazily load (and cache) the sentence-transformer embedding model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    return _embeddings


def build_vector_store(chunks: List[Document]) -> FAISS:
    """
    Embed every chunk and build a brand-new FAISS index from them.
    Use this the first time you ingest documents.
    """
    embeddings = get_embeddings_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def add_to_vector_store(vector_store: FAISS, chunks: List[Document]) -> FAISS:
    """
    Embed new chunks and merge them into an existing FAISS index.
    Use this when the user uploads more documents later, so you don't
    have to re-embed everything from scratch.
    """
    embeddings = get_embeddings_model()
    new_store = FAISS.from_documents(chunks, embeddings)
    vector_store.merge_from(new_store)
    return vector_store


def save_vector_store(vector_store: FAISS, path: str = config.VECTOR_STORE_DIR) -> None:
    """Persist the FAISS index to disk so it survives an app restart."""
    vector_store.save_local(path)


def load_vector_store(path: str = config.VECTOR_STORE_DIR) -> Optional[FAISS]:
    """
    Load a previously saved FAISS index from disk, if one exists.
    Returns None if no index has been built yet.
    """
    if not os.path.exists(path):
        return None
    embeddings = get_embeddings_model()
    # allow_dangerous_deserialization=True is required by LangChain because
    # FAISS indexes are loaded via pickle. Safe here since WE wrote the file ourselves.
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
