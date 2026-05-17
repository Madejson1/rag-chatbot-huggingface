# RAG Chatbot API - HuggingFace Local

A free, local document-based chatbot using **LangChain**, **HuggingFace**, **FAISS**, **FastAPI** and **Docker**.

## What it does

1. Loads documents from `data/docs/`
2. Splits them into text chunks
3. Creates embeddings using HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
4. Stores chunks in FAISS vector database
5. Answers questions through a FastAPI REST API using `google/flan-t5-small`

## Tech stack

- Python
- LangChain
- HuggingFace Transformers
- HuggingFace Embeddings
- FAISS
- FastAPI
- Docker

## Run locally

### 1. Create virtual environment

Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Mac/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Build vector database

```bash
python -m scripts.ingest
```

### 4. Run API

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

### 5. Test endpoint

In Swagger UI open `/ask`, click `Try it out`, and use:

```json
{
  "question": "What are the remote work rules?",
  "k": 4
}
```

## Docker

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8000/docs
```
