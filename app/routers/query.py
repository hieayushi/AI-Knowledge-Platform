import logging
from fastapi import APIRouter, HTTPException
from app.models.query import QueryRequest, QueryResponse
from app.services.rag_service import rag_service

logger = logging.getLogger("ai_platform.routers.query")

router = APIRouter(tags=["Semantic Search & Retrieval"])

@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query Document Knowledge Base via Natural Language.
    Performs vector similarity search in Qdrant, applies metadata filtering, and returns hybrid ranked chunks.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    try:
        response = rag_service.query(request)
        return response
    except Exception as e:
        logger.error(f"Error executing semantic search query: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")
