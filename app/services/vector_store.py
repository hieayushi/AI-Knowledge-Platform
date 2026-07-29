import uuid
import logging
from typing import List, Dict, Any, Optional
from qdrant_client.http import models as rest_models

from app.database.qdrant import qdrant_manager
from app.services.embedding import embedding_service
from app.config import settings

logger = logging.getLogger("ai_platform.vector_store")

class VectorStoreService:
    """
    Qdrant Vector Store Operations Manager.
    Handles vector upsert, similarity search with payload filtering, and deletion.
    """
    
    def upsert_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        file_name: str,
        file_type: str,
        tags: Optional[List[str]] = None
    ) -> List[str]:
        """
        Embeds chunk texts and stores them in Qdrant with metadata payloads.
        """
        if not chunks:
            return []

        client = qdrant_manager.client
        if client is None:
            qdrant_manager.initialize()
            client = qdrant_manager.client

        texts = [c["content"] for c in chunks]
        embeddings = embedding_service.embed_batch(texts)

        points = []
        chunk_ids = []
        
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_{idx}"))
            chunk_ids.append(chunk_id)

            payload = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_index": idx,
                "file_name": file_name,
                "file_type": file_type,
                "content": chunk["content"],
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "char_count": chunk.get("char_count"),
                "tags": tags or []
            }

            points.append(
                rest_models.PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload=payload
                )
            )

        client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points
        )
        logger.info(f"Successfully stored {len(points)} vector chunks for document '{document_id}' in Qdrant.")
        return chunk_ids

    def search_vectors(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Performs Cosine vector similarity search over Qdrant with optional payload filters.
        """
        client = qdrant_manager.client
        if client is None:
            qdrant_manager.initialize()
            client = qdrant_manager.client

        query_vector = embedding_service.embed_text(query_text)

        # Build Qdrant Filter
        must_conditions = []
        if filters:
            if "document_ids" in filters and filters["document_ids"]:
                must_conditions.append(
                    rest_models.FieldCondition(
                        key="document_id",
                        match=rest_models.MatchAny(any=filters["document_ids"])
                    )
                )
            if "file_type" in filters and filters["file_type"]:
                must_conditions.append(
                    rest_models.FieldCondition(
                        key="file_type",
                        match=rest_models.MatchValue(value=filters["file_type"])
                    )
                )

        qdrant_filter = rest_models.Filter(must=must_conditions) if must_conditions else None

        search_response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            score_threshold=min_score_threshold if min_score_threshold > 0 else None
        )

        results = []
        for point in search_response.points:
            results.append({
                "chunk_id": point.id,
                "document_id": point.payload.get("document_id"),
                "file_name": point.payload.get("file_name"),
                "file_type": point.payload.get("file_type"),
                "chunk_index": point.payload.get("chunk_index"),
                "content": point.payload.get("content"),
                "similarity_score": point.score,
                "start_line": point.payload.get("start_line"),
                "end_line": point.payload.get("end_line"),
                "metadata": {
                    "tags": point.payload.get("tags", []),
                    "char_count": point.payload.get("char_count")
                }
            })

        return results

    def delete_document_vectors(self, document_id: str) -> int:
        """
        Purges all chunk vectors belonging to a document from Qdrant.
        """
        client = qdrant_manager.client
        if client is None:
            return 0

        filter_condition = rest_models.Filter(
            must=[
                rest_models.FieldCondition(
                    key="document_id",
                    match=rest_models.MatchValue(value=document_id)
                )
            ]
        )
        
        client.delete(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points_selector=rest_models.FilterSelector(filter=filter_condition)
        )
        logger.info(f"Purged vector chunks for document '{document_id}' from Qdrant.")
        return 1

vector_store_service = VectorStoreService()
