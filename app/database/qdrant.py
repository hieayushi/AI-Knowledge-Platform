import os
import shutil
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from app.config import settings

logger = logging.getLogger("ai_platform.qdrant")

class QdrantManager:
    """
    Qdrant Local Persistent Vector Database Client Manager.
    Stores vector embeddings and metadata payloads locally without Docker.
    """
    def __init__(self):
        self.client: QdrantClient = None

    def initialize(self):
        if self.client is not None:
            return

        os.makedirs(settings.QDRANT_STORAGE_PATH, exist_ok=True)
        
        # Remove stale lock files if left behind by previous crashed instances
        lock_file = os.path.join(settings.QDRANT_STORAGE_PATH, ".lock")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                logger.info("Removed stale Qdrant lock file.")
            except Exception:
                pass

        try:
            logger.info(f"Initializing embedded Qdrant store at '{settings.QDRANT_STORAGE_PATH}'...")
            self.client = QdrantClient(path=settings.QDRANT_STORAGE_PATH)
        except Exception as e:
            logger.warning(f"Qdrant file storage locked or unavailable ({e}). Falling back to local in-memory Qdrant instance.")
            self.client = QdrantClient(location=":memory:")

        # Check or create collection
        try:
            collections = self.client.get_collections().collections
            existing_names = [c.name for c in collections]
            
            if settings.QDRANT_COLLECTION_NAME not in existing_names:
                logger.info(f"Creating Qdrant collection '{settings.QDRANT_COLLECTION_NAME}'...")
                self.client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    vectors_config=rest_models.VectorParams(
                        size=settings.EMBEDDING_DIMENSION,
                        distance=rest_models.Distance.COSINE
                    )
                )
                
                # Payload indexes for metadata filtering
                self.client.create_payload_index(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    field_name="document_id",
                    field_schema=rest_models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    field_name="file_type",
                    field_schema=rest_models.PayloadSchemaType.KEYWORD
                )
            logger.info("Qdrant store initialized successfully.")
        except Exception as e:
            logger.error(f"Error configuring Qdrant collection: {e}")

qdrant_manager = QdrantManager()
