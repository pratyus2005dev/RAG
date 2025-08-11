

import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


# -------------------- Configuration --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # We raise at startup to fail fast if key is missing — change if you want local mocking
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))  # deterministic by default
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "4"))


# -------------------- App & Models --------------------
app = FastAPI(title="AI Study Helper (RAG)", version="0.1")


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = None  # override default number of retrieved documents


class Source(BaseModel):
    doc_id: Optional[str]
    page: Optional[int]
    text: str
    score: Optional[float]


class QueryWithSourcesResponse(BaseModel):
    answer: str
    sources: List[Source]


# -------------------- Initialize embeddings, llm, and vectorstore --------------------

# Create embedding function
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create LLM instance
llm = OpenAI(temperature=LLM_TEMPERATURE, openai_api_key=OPENAI_API_KEY)

# Try to load an existing Chroma vectorstore
if not os.path.isdir(CHROMA_PERSIST_DIR):
    raise RuntimeError(
        f"Chroma persist directory not found: {CHROMA_PERSIST_DIR}. Run the ingestion pipeline first."
    )

try:
    vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Chroma vectorstore: {e}")

retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVE_K})

# RetrievalQA chain: this will create a simple chain that stuffs the retrieved docs into the prompt
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)


# -------------------- Helper functions --------------------

def retrieve_documents(query: str, k: int) -> List:
    """Return the top-k retrieved documents using the vectordb retriever."""
    used_k = k or RETRIEVE_K
    # We temporarily create a retriever with the requested k if different
    if used_k == RETRIEVE_K:
        docs = retriever.get_relevant_documents(query)
    else:
        temp = vectordb.as_retriever(search_kwargs={"k": used_k})
        docs = temp.get_relevant_documents(query)
    return docs


# -------------------- Endpoints --------------------

@app.get("/health")
def health():
    return {"status": "ok", "chroma_dir": CHROMA_PERSIST_DIR}


@app.post("/query")
def query(req: QueryRequest):
    """Return the answer string for a user question."""
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=422, detail="question must be a non-empty string")

    # Option A: use the high-level chain which manages retrieval internally
    # If the user supplied k, we will build a temporary chain so that the search k is used
    if req.k and req.k != RETRIEVE_K:
        temp_retriever = vectordb.as_retriever(search_kwargs={"k": req.k})
        temp_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=temp_retriever)
        answer = temp_qa.run(req.question)
    else:
        answer = qa_chain.run(req.question)

    return {"answer": answer}


@app.post("/query_with_sources", response_model=QueryWithSourcesResponse)
def query_with_sources(req: QueryRequest):
    """Return the answer AND the retrieved source chunks so the user can verify.

    Note: this endpoint re-runs retrieval explicitly to gather source texts and scores.
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=422, detail="question must be a non-empty string")

    k = req.k or RETRIEVE_K
    docs = retrieve_documents(req.question, k)

    # Build the answer by using the LLM directly with the top documents in context.
    # We use the same simple strategy as RetrievalQA: concatenate the docs and ask the LLM.
    # For more sophisticated prompting, replace the prompt below with a template.

    context_texts = [d.page_content for d in docs]
    system_prompt = (
        "You are an expert study assistant. Use the provided context excerpts to answer the question. "
        "If the answer is not contained in the context, say you don't know and suggest where to look."
    )

    prompt = """Context:
{context}

Question: {question}

Answer concisely and cite which context piece (by index) you used when relevant.
""".format(context="\n---\n".join(context_texts), question=req.question)

    # Call LLM
    try:
        answer = llm(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    # Prepare source objects
    sources = []
    for i, d in enumerate(docs):
        meta = getattr(d, "metadata", {}) or {}
        src = Source(
            doc_id=meta.get("source") or meta.get("doc_id") or None,
            page=meta.get("page") or None,
            text=(d.page_content[:1000] + "...") if len(d.page_content) > 1000 else d.page_content,
            score=None,
        )
        sources.append(src)

    return QueryWithSourcesResponse(answer=str(answer), sources=sources)


# -------------------- Startup shutdown hooks (optional) --------------------

@app.on_event("shutdown")
def shutdown_event():
    try:
        # If Chroma has a persist method available, call it to ensure clean shutdown
        if hasattr(vectordb, "persist"):
            vectordb.persist()
    except Exception:
        # We intentionally swallow errors on shutdown to avoid noisy failures
        pass


# -------------------- If run as script --------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
