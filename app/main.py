from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import ask_question, build_vectorstore


app = FastAPI(
    title="RAG Chatbot API - HuggingFace Local",
    description="Local document-based chatbot using LangChain, HuggingFace, FAISS and FastAPI.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What are the remote work rules?"])
    k: int = Field(2, ge=1, le=10)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/")
def root():
    return {
        "message": "RAG Chatbot API is running.",
        "docs": "/docs",
        "endpoints": ["/ask", "/ingest"],
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        return ask_question(request.question, request.k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest")
def ingest():
    try:
        build_vectorstore()
        return {"status": "success", "message": "Vectorstore rebuilt."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
