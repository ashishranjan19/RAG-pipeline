"""
config.py
---------
Single place for every tunable knob in the pipeline.
Keeping config centralized means you can tune retrieval quality
(chunk size, k, model choice) without hunting through every file.
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env file (ANTHROPIC_API_KEY=..., etc.)
load_dotenv()

# --- Ollama (local LLM server) ---
# Ollama must be running locally: https://ollama.com -> `ollama serve`
# Pull the model once with:  ollama pull mistral
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Chunking ---
# CHUNK_SIZE: how many characters go into one chunk before it's embedded.
#   Smaller = more precise retrieval, more chunks, more embedding cost.
#   Larger  = more context per chunk, but retrieval gets "noisier".
CHUNK_SIZE = 1000

# CHUNK_OVERLAP: characters shared between consecutive chunks.
#   Prevents a sentence/idea from being awkwardly cut in half at a chunk boundary.
CHUNK_OVERLAP = 150

# --- Embeddings ---
# A small, fast, local sentence-transformer model. No API key required.
# Swap to "sentence-transformers/all-mpnet-base-v2" for higher quality (slower).
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Retrieval ---
# How many chunks to pull back from the vector store per query.
TOP_K = 4

# --- LLM ---
# Local Mistral model served by Ollama. Must match a model you've pulled:
#   ollama pull mistral
LLM_MODEL_NAME = "mistral"
LLM_TEMPERATURE = 0.2  # low temperature = more grounded, less "creative" answers

# --- Storage paths ---
UPLOAD_DIR = "uploaded_docs"        # where raw uploaded files are saved
VECTOR_STORE_DIR = "vector_store"   # where the FAISS index is persisted to disk
