from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    docs_path: str = Field("data/docs")
    vectorstore_path: str = Field("vectorstore/faiss_index")

    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2")
    generation_model: str = Field("google/flan-t5-small")

    chunk_size: int = 250
    chunk_overlap: int = 40
    max_new_tokens: int = Field(180)

    class Config:
        env_file = ".env"


settings = Settings()
