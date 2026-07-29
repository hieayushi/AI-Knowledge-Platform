from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class MetadataFilter(BaseModel):
    document_ids: Optional[List[str]] = None
    file_type: Optional[str] = None
    tags: Optional[List[str]] = None

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of relevant chunks to retrieve")
    filters: Optional[MetadataFilter] = None
    min_score_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    enable_reranking: bool = Field(default=True, description="Enable hybrid BM25 + Vector reranking")

class SearchResultChunk(BaseModel):
    chunk_id: str
    document_id: str
    file_name: str
    file_type: str
    chunk_index: int
    content: str
    similarity_score: float
    rerank_score: Optional[float] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    metadata: Dict[str, Any] = {}

class QueryResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResultChunk]
    execution_time_ms: float

class AIGatewayRequest(BaseModel):
    question: str = Field(..., description="User question")
    top_k: int = Field(default=5, description="Number of context chunks to fetch")
    filters: Optional[MetadataFilter] = None
    system_instructions: Optional[str] = "You are an AI Assistant serving as an internal knowledge platform guide."

class AIGatewayResponse(BaseModel):
    answer: str
    retrieved_sources: List[SearchResultChunk]
    model_used: str = "Internal-RAG-Gateway"
    execution_time_ms: float
