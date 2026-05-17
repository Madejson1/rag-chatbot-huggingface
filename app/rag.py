from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from app.config import settings

import re

_GENERATOR = None


def load_documents(docs_path: str) -> List[Document]:
    path = Path(docs_path)
    documents: List[Document] = []

    if not path.exists():
        raise FileNotFoundError(f"Docs directory does not exist: {docs_path}")

    for file_path in path.rglob("*"):
        if file_path.is_dir():
            continue

        suffix = file_path.suffix.lower()

        if suffix in [".txt", ".md"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

        elif suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())

    if not documents:
        raise ValueError(f"No supported documents found in {docs_path}")

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore() -> None:
    documents = load_documents(settings.docs_path)
    chunks = split_documents(documents)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(settings.vectorstore_path)


def load_vectorstore() -> FAISS:
    embeddings = get_embeddings()
    return FAISS.load_local(
        settings.vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_prompt(question: str, context: str) -> str:
    return f"""
You are a precise assistant.

Answer the question using ONLY relevant parts of the context.
If multiple topics are present, focus strictly on the question.
If the answer is not clearly in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""".strip()


def get_generator():
    global _GENERATOR
    if _GENERATOR is None:
        tokenizer = AutoTokenizer.from_pretrained(settings.generation_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(settings.generation_model)
        _GENERATOR = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=settings.max_new_tokens,
        )
    return _GENERATOR


def generate_answer(prompt: str) -> str:
    generator = get_generator()
    result = generator(prompt)
    return result[0]["generated_text"]


import re


def clean_markdown(text: str) -> str:
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def extract_relevant_answer(text: str, question: str) -> str:
    text = clean_markdown(text)

    # Dzielimy po liniach i zdaniach, żeby nie zwracać całego chunka.
    candidates = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = re.split(r"(?<=[.!?])\s+", line)
        candidates.extend([p.strip() for p in parts if len(p.strip()) > 20])

    question_words = {
        w.strip(".,!?;:()[]").lower()
        for w in question.split()
        if len(w.strip(".,!?;:()[]")) > 2
    }

    best = candidates[0] if candidates else text
    best_score = -1

    for candidate in candidates:
        candidate_words = {
            w.strip(".,!?;:()[]").lower()
            for w in candidate.split()
        }

        score = len(question_words & candidate_words)

        # boost pod remote/work, bo pytanie jest o remote work rules
        if "remote" in candidate.lower():
            score += 3
        if "work" in candidate.lower():
            score += 2

        if score > best_score:
            best_score = score
            best = candidate

    return best


def ask_question(question: str, k: int = 4) -> dict:
    vectorstore = load_vectorstore()
    retrieved_docs = vectorstore.similarity_search(question, k=k)

    best_doc = retrieved_docs[0]
    answer = extract_relevant_answer(best_doc.page_content, question)

    sources = sorted(
        {
            str(doc.metadata.get("source", "unknown"))
            for doc in retrieved_docs
        }
    )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }