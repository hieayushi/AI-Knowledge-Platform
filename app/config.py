import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Internal AI Knowledge Platform"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # MongoDB Local Configuration
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "ai_knowledge_platform")
    
    # Qdrant Local Persistent Storage
    QDRANT_STORAGE_PATH: str = os.getenv("QDRANT_STORAGE_PATH", "./qdrant_data")
    QDRANT_COLLECTION_NAME: str = "knowledge_chunks"
    
    # Embedding Model (Local SentenceTransformer)
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = 384
    
    # Processing Configuration
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50
    
    # LLM Gateway Settings (Local HuggingFace Model)
    LOCAL_LLM_MODEL_NAME: str = os.getenv("LOCAL_LLM_MODEL_NAME", "google/flan-t5-base")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
