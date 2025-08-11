AI Study Helper (RAG) — Retrieval-Augmented Generation Study Assistant
This project is a Retrieval-Augmented Generation (RAG) AI-powered study helper that allows students to query their study materials (PDFs, text files, notes) and get grounded answers by retrieving relevant document chunks and generating answers using OpenAI’s LLM.

Features
Ingest study materials (PDF, TXT, Markdown) into a vector database (Chroma)

Chunk documents intelligently with overlap for context

Generate vector embeddings via OpenAI embeddings API

Perform semantic search over documents using vector similarity

Use a RetrievalQA chain to combine retrieved context with LLM for grounded answers

Serve answers via a FastAPI REST API

Dockerized for easy deployment

Optional endpoint to return sources alongside answers for verification

Project Structure
graphql
Copy
Edit
ai-study-helper/
├─ app/
│  ├─ main.py           # FastAPI app serving the RAG QA endpoints
│  └─ rag_chain.py      # RAG chain setup & query helper class
├─ ingest/
│  └─ ingest.py         # Script to ingest and embed documents into Chroma vectorstore
├─ data/
│  ├─ raw_docs/         # Place your PDFs, txt, md files here before ingestion
│  └─ chroma_db/        # Persisted vectorstore directory (created on ingest)
├─ requirements.txt
├─ Dockerfile
└─ README.md
Prerequisites
Python 3.10+

OpenAI API key with access to embeddings and LLM models

Docker (optional, for containerized deployment)

Setup and Usage
1. Clone the repo
bash
Copy
Edit
git clone https://github.com/yourusername/ai-study-helper.git
cd ai-study-helper
2. Install dependencies
bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
# or
.venv\Scripts\activate        # Windows

pip install -r requirements.txt
3. Add your study materials
Place your PDF, TXT, or Markdown files inside the folder:

bash
Copy
Edit
data/raw_docs/
4. Set environment variable
Export your OpenAI API key:

bash
Copy
Edit
export OPENAI_API_KEY="your_openai_api_key"    # Linux/macOS
set OPENAI_API_KEY=your_openai_api_key         # Windows CMD
$Env:OPENAI_API_KEY="your_openai_api_key"      # Windows PowerShell
5. Ingest documents
Run the ingestion pipeline to embed and index your documents:

bash
Copy
Edit
python ingest/ingest.py --input_dir ./data/raw_docs --persist_dir ./data/chroma_db
This creates a vector store with embedded chunks in ./data/chroma_db.

6. Run the API server
bash
Copy
Edit
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
The API will be available at http://localhost:8000.

API Endpoints
POST /query
Input: JSON { "question": "your question here" }
Output: { "answer": "generated answer" }

POST /query_with_sources
Input: JSON { "question": "your question here" }
Output: { "answer": "...", "sources": [ { "doc_id": "...", "page": ..., "text": "..." }, ... ] }

GET /health
Health check endpoint returns status and vector DB location.

Docker
Build Docker image
bash
Copy
Edit
docker build -t ai-study-helper .
Run container
bash
Copy
Edit
docker run -e OPENAI_API_KEY="your_openai_api_key" -p 8000:8000 ai-study-helper
Deployment
You can deploy the Docker container to any cloud platform supporting Docke
