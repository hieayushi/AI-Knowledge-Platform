import logging
from pymongo import MongoClient
from pymongo.collection import Collection
from app.config import settings

logger = logging.getLogger("ai_platform.mongo")

class MongoManager:
    """
    MongoDB Connection and Database Management.
    Handles storage for document metadata, chunks, and audit query logs.
    """
    def __init__(self):
        self.client: MongoClient = None
        self.db = None

    def connect(self):
        try:
            logger.info(f"Connecting to MongoDB at {settings.MONGO_URI}...")
            self.client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=3000)
            # Ping database to confirm connection
            self.client.admin.command('ping')
            self.db = self.client[settings.MONGO_DB_NAME]
            self._ensure_indexes()
            logger.info("Successfully connected to MongoDB.")
        except Exception as e:
            logger.warning(f"Failed to connect to local MongoDB server: {e}. Operating with local fallback mode.")
            self.client = None
            self.db = None

    def _ensure_indexes(self):
        if self.db is not None:
            # Index documents by document_id and status
            self.db.documents.create_index("document_id", unique=True)
            self.db.documents.create_index("status")
            self.db.documents.create_index("is_deleted")
            
            # Index chunks by document_id and chunk_id
            self.db.chunks.create_index("chunk_id", unique=True)
            self.db.chunks.create_index("document_id")

            # Index query logs by timestamp
            self.db.query_logs.create_index("timestamp")

    def disconnect(self):
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection.")

    @property
    def documents(self) -> Collection:
        if self.db is None:
            return None
        return self.db.documents

    @property
    def chunks(self) -> Collection:
        if self.db is None:
            return None
        return self.db.chunks

    @property
    def query_logs(self) -> Collection:
        if self.db is None:
            return None
        return self.db.query_logs

mongo_manager = MongoManager()
