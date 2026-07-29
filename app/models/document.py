from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class FileType(str, Enum):
    PDF = "pdf"
    CODE = "code"
    MARKDOWN = "markdown"
    TEXT = "text"
    WEB = "web"

class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    char_count: int
    token_estimate: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DocumentRecord(BaseModel):
    document_id: str
    title: str
    file_name: str
    file_type: FileType
    file_size_bytes: int
    content_hash: str
    status: DocumentStatus = DocumentStatus.PENDING
    error_message: Optional[str] = None
    chunk_count: int = 0
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    is_deleted: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class DocumentUploadResponse(BaseModel):
    document_id: str
    file_name: str
    file_type: str
    status: DocumentStatus
    chunk_count: int
    message: str

class ScrapeUrlRequest(BaseModel):
    url: str
    tags: Optional[List[str]] = Field(default_factory=list)
    chunk_size: Optional[int] = 500

class DeleteDocumentResponse(BaseModel):
    document_id: str
    status: str
    soft_deleted_in_mongo: bool
    purged_from_qdrant: bool
    deleted_chunks_count: int
    message: str
