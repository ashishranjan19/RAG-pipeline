"""
app.py
------
Streamlit UI for the RAG pipeline. Lets someone:
  1. Upload documents (PDF / DOCX / TXT / MD)
  2. Click "Process" to run them through ingestion (load -> chunk -> embed -> store)
  3. Ask questions and get answers grounded in those documents, via Mistral

Run with:  streamlit run app.py
"""

import os
import streamlit as st

import config
from document_loader import process_documents
from vector_store import (
    build_vector_store,
    add_to_vector_store,
    save_vector_store,
    load_vector_store,
)
from rag_chain import get_answer_with_sources

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 Chat with your documents")

os.makedirs(config.UPLOAD_DIR, exist_ok=True)

# --- Session state ---
# `vector_store` lives in session_state so it survives Streamlit's rerun-on-every-
# interaction behavior. We try to load a previously saved index on first run,
# so uploaded docs persist across app restarts. Wrapped in try/except so a
# corrupted/incompatible saved index doesn't crash the whole app.
if "vector_store" not in st.session_state:
    try:
        st.session_state.vector_store = load_vector_store()
    except Exception as e:
        st.warning(f"Could not load existing vector store, starting fresh. ({e})")
        st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer, sources)


# ---------------------------------------------------------------------------
# Sidebar: document upload + ingestion
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT, or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Process documents", disabled=not uploaded_files):
        with st.spinner("Loading, chunking, and embedding your documents..."):
            try:
                # Save uploaded files to disk so our loaders (which expect file paths) can read them.
                for uploaded_file in uploaded_files:
                    save_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                # Run the ingestion pipeline: load -> split into chunks.
                chunks = process_documents(config.UPLOAD_DIR)

                if not chunks:
                    st.error("No readable text found in the uploaded files.")
                else:
                    # Embed + store. If a vector store already exists, append to it
                    # instead of throwing away previously ingested documents.
                    if st.session_state.vector_store is None:
                        st.session_state.vector_store = build_vector_store(chunks)
                    else:
                        st.session_state.vector_store = add_to_vector_store(
                            st.session_state.vector_store, chunks
                        )

                    save_vector_store(st.session_state.vector_store)
                    st.success(f"Processed {len(uploaded_files)} file(s) into {len(chunks)} chunks.")
            except Exception as e:
                # Surfaces real causes: bad file, embedding model download failure, etc.
                st.error(f"Ingestion failed: {e}")

    st.divider()
    st.caption(
        f"Chunk size: {config.CHUNK_SIZE} · Overlap: {config.CHUNK_OVERLAP} · "
        f"Top-k retrieved: {config.TOP_K}"
    )


# ---------------------------------------------------------------------------
# Main panel: chat interface
# ---------------------------------------------------------------------------
st.header("2. Ask a question")

if st.session_state.vector_store is None:
    st.info("Upload and process at least one document to start asking questions.")
else:
    question = st.text_input("Your question about the uploaded documents:")

    if st.button("Ask", disabled=not question):
        with st.spinner("Retrieving relevant chunks and generating an answer..."):
            try:
                answer, sources = get_answer_with_sources(st.session_state.vector_store, question)
                st.session_state.chat_history.append((question, answer, sources))
            except Exception as e:
                # Common causes: missing/invalid MISTRAL_API_KEY, or a Mistral API error.
                st.error(f"Failed to generate answer: {e}")

    # Show conversation history, most recent first.
    for q, a, srcs in reversed(st.session_state.chat_history):
        st.markdown(f"**Q: {q}**")
        st.write(a)
        if srcs:
            st.caption(f"Sources: {', '.join(srcs)}")
        st.divider()
