import logging
from typing import List
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger("ai_platform.embedding")

class EmbeddingService:
    """
    100% Local Embedding Service powered by SentenceTransformers.
    Generates dense vector embeddings for semantic search without cloud dependencies.
    """
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.model = None

    def load_model(self):
        if self.model is None:
            logger.info(f"Loading SentenceTransformer model '{self.model_name}' locally...")
            self.model = SentenceTransformer(self.model_name)
            logger.info("SentenceTransformer model loaded successfully.")

    def embed_text(self, text: str) -> List[float]:
        self.load_model()
        cleaned_text = text.replace("\n", " ").strip()
        vector = self.model.encode(cleaned_text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self.load_model()
        cleaned_texts = [t.replace("\n", " ").strip() for t in texts]
        vectors = self.model.encode(cleaned_texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

embedding_service = EmbeddingService()
