"""
rag_chain.py
------------
Stage 5, 6 & 7 of the pipeline: RETRIEVE relevant chunks for a user's
question, AUGMENT a prompt with that context, and GENERATE the final
answer with the LLM.

This uses LangChain's LCEL (LangChain Expression Language) "|" pipe
syntax, which wires components together into a single runnable chain.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

import config

# The prompt template that grounds the LLM in ONLY the retrieved context.
# This is the key guardrail against hallucination: we explicitly instruct
# the model to say "I don't know" rather than invent an answer.
RAG_PROMPT_TEMPLATE = """You are a helpful assistant answering questions based ONLY on the context below.
If the answer is not contained in the context, say "I don't have enough information in the uploaded documents to answer that."
Do not use outside knowledge. Be concise and cite which part of the context you used when relevant.

Context:
{context}

Question:
{question}

Answer:"""


def format_docs(docs) -> str:
    """
    Turn the list of retrieved Document chunks into one plain-text block,
    tagging each chunk with its source file so the LLM (and the user)
    can trace an answer back to a specific document.
    """
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(vector_store: FAISS):
    """
    Wire together: retriever -> prompt -> LLM -> string output.

    retriever: turns the user's question into a vector, does similarity
               search against FAISS, and returns the top-k closest chunks.
    prompt:    fills the RAG_PROMPT_TEMPLATE with those chunks + the question.
    llm:       Mistral (running locally via Ollama) generates the final answer.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": config.TOP_K})

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    llm = ChatOllama(
        model=config.LLM_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
        base_url=config.OLLAMA_BASE_URL,
    )

    # LCEL chain: retrieve context in parallel with passing the question through,
    # format it into the prompt, send to the LLM, then parse the output to a plain string.
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def get_answer_with_sources(vector_store: FAISS, question: str):
    """
    Run the chain AND separately return the raw retrieved docs, so the UI
    can show the user which document(s) the answer came from.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": config.TOP_K})
    retrieved_docs = retriever.invoke(question)

    chain = build_rag_chain(vector_store)
    answer = chain.invoke(question)

    sources = sorted({doc.metadata.get("source_file", "unknown") for doc in retrieved_docs})
    return answer, sources
