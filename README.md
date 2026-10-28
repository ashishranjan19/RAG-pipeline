# RAG document Q&A pipeline (LangChain + FAISS + Streamlit + Ollama/Mistral)

Upload your own documents and ask questions grounded in their content.
The LLM (Mistral) runs **locally via Ollama** — no API key, no internet
required after the model is pulled.

## How it works

**Ingestion (runs once per upload):**
`document_loader.py` loads PDFs/DOCX/TXT and splits them into overlapping
chunks -> `vector_store.py` embeds each chunk with a local sentence-transformer
model and stores the vectors in a FAISS index on disk.

**Query (runs per question):**
`rag_chain.py` embeds your question, retrieves the top-k most similar chunks
from FAISS, stuffs them into a prompt template, and sends it to a local
Mistral model (served by Ollama) to generate a grounded answer with source
attribution.

**`app.py`** is the Streamlit UI tying it all together.

## Setup

### 1. Install and start Ollama
Download from https://ollama.com, then:

```bash
ollama serve            # starts the local server on http://localhost:11434
ollama pull mistral     # downloads the Mistral model (~4GB, one-time)
```

Leave `ollama serve` running in a terminal (or it may already run as a
background service depending on your OS install).

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

No `.env` file or API key is needed for the LLM. If your Ollama server runs
on a non-default host/port, set:

```
OLLAMA_BASE_URL=http://localhost:11434
```

## Run

```bash
streamlit run app.py
```

Then in the browser:
1. Upload one or more documents in the sidebar and click **Process documents**.
2. Ask a question in the main panel — the answer is generated only from what
   was found in your documents, with the source file(s) listed underneath.

## File map

| File | Responsibility |
|---|---|
| `config.py` | All tunable settings: chunk size, top-k, model names, Ollama URL |
| `document_loader.py` | Load files (PDF/DOCX/TXT/MD) and split into chunks |
| `vector_store.py` | Embed chunks, build/load/save the FAISS index |
| `rag_chain.py` | LCEL chain: retrieve -> prompt -> local Mistral (Ollama) -> answer |
| `app.py` | Streamlit UI tying it all together |

## Tuning tips

- **Answers missing detail?** Lower `CHUNK_SIZE` in `config.py` for more
  precise retrieval, or raise `TOP_K` to pull back more chunks.
- **Answers too slow / noisy?** Raise `CHUNK_SIZE` and lower `TOP_K`.
- **"Connection refused" errors?** Ollama isn't running — start it with
  `ollama serve` and confirm `ollama list` shows `mistral`.
- **Want a different local model?** `ollama pull mistral-nemo` (or any other
  model) then set `LLM_MODEL_NAME` in `config.py` to match.
- **Want a hosted vector DB instead of local FAISS?** Swap the `FAISS`
  import in `vector_store.py` for `Pinecone`, `Chroma`, `Qdrant`, etc.
