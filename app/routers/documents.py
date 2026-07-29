import uuid
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, BackgroundTasks

from app.models.document import (
    FileType, DocumentStatus, DocumentRecord, ChunkRecord,
    DocumentUploadResponse, ScrapeUrlRequest, DeleteDocumentResponse
)
from app.services.ingestion import ingestion_service
from app.services.vector_store import vector_store_service
from app.database.mongo import mongo_manager

logger = logging.getLogger("ai_platform.routers.documents")

router = APIRouter(prefix="/documents", tags=["Document Management"])

# In-memory document fallback store if MongoDB is offline locally
in_memory_docs: dict = {}
in_memory_chunks: dict = {}

@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(default=""),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and Ingest a Document or Code file.
    Supports PDF, Markdown, Plain Text, and Source Code (.py, .js, .ts, etc.).
    Extracts text, chunks data, generates embeddings, and indexes in Qdrant & MongoDB.
    """
    filename = file.filename.lower()
    content_bytes = await file.read()
    
    if not content_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Determine file type
    if filename.endswith(".pdf"):
        file_type = FileType.PDF
    elif filename.endswith((".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".html", ".css", ".json")):
        file_type = FileType.CODE
    elif filename.endswith((".md", ".markdown")):
        file_type = FileType.MARKDOWN
    else:
        file_type = FileType.TEXT

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    content_hash = ingestion_service.calculate_hash(content_bytes)
    doc_id = str(uuid.uuid4())

    # Extraction & Chunking based on type
    if file_type == FileType.PDF:
        full_text, chunks = ingestion_service.process_pdf(content_bytes, file.filename)
    elif file_type == FileType.CODE:
        code_str = content_bytes.decode("utf-8", errors="replace")
        full_text, chunks = ingestion_service.process_code(code_str, file.filename)
    else:
        text_str = content_bytes.decode("utf-8", errors="replace")
        full_text = text_str
        chunks = ingestion_service.chunk_text(text_str)

    # Store in Qdrant Vector Store
    chunk_ids = vector_store_service.upsert_chunks(
        document_id=doc_id,
        chunks=chunks,
        file_name=file.filename,
        file_type=file_type.value,
        tags=tag_list
    )

    # Build document record
    doc_record = DocumentRecord(
        document_id=doc_id,
        title=file.filename,
        file_name=file.filename,
        file_type=file_type,
        file_size_bytes=len(content_bytes),
        content_hash=content_hash,
        status=DocumentStatus.COMPLETED,
        chunk_count=len(chunks),
        tags=tag_list,
        metadata={"chunk_ids": chunk_ids, "extracted_char_count": len(full_text)},
        created_at=datetime.utcnow()
    )

    # Save to MongoDB or Fallback
    if mongo_manager.documents is not None:
        try:
            mongo_manager.documents.insert_one(doc_record.dict())
        except Exception as e:
            logger.error(f"Error saving to MongoDB: {e}")
            in_memory_docs[doc_id] = doc_record.dict()
    else:
        in_memory_docs[doc_id] = doc_record.dict()

    return DocumentUploadResponse(
        document_id=doc_id,
        file_name=file.filename,
        file_type=file_type.value,
        status=DocumentStatus.COMPLETED,
        chunk_count=len(chunks),
        message=f"Document '{file.filename}' processed and indexed with {len(chunks)} chunks."
    )


@router.post("/scrape", response_model=DocumentUploadResponse, status_code=201)
async def scrape_and_ingest_url(request: ScrapeUrlRequest):
    """
    Scrape a Web URL using BeautifulSoup, extract body content, chunk, embed & store.
    """
    try:
        full_text, chunks, title = ingestion_service.process_web_url(request.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to scrape URL '{request.url}': {str(e)}")

    doc_id = str(uuid.uuid4())
    content_hash = ingestion_service.calculate_hash(full_text.encode("utf-8"))

    # Vector store indexing
    chunk_ids = vector_store_service.upsert_chunks(
        document_id=doc_id,
        chunks=chunks,
        file_name=title,
        file_type=FileType.WEB.value,
        tags=request.tags
    )

    doc_record = DocumentRecord(
        document_id=doc_id,
        title=title,
        file_name=request.url,
        file_type=FileType.WEB,
        file_size_bytes=len(full_text.encode("utf-8")),
        content_hash=content_hash,
        status=DocumentStatus.COMPLETED,
        chunk_count=len(chunks),
        tags=request.tags or [],
        metadata={"url": request.url, "scraped_by": "BeautifulSoup4"},
        created_at=datetime.utcnow()
    )

    if mongo_manager.documents is not None:
        mongo_manager.documents.insert_one(doc_record.dict())
    else:
        in_memory_docs[doc_id] = doc_record.dict()

    return DocumentUploadResponse(
        document_id=doc_id,
        file_name=request.url,
        file_type=FileType.WEB.value,
        status=DocumentStatus.COMPLETED,
        chunk_count=len(chunks),
        message=f"Web content from '{request.url}' successfully scraped via BeautifulSoup4 and indexed."
    )


@router.get("", response_model=List[DocumentRecord])
async def list_documents(include_deleted: bool = False):
    """
    List all uploaded documents with metadata and status.
    """
    query = {} if include_deleted else {"is_deleted": {"$ne": True}}
    
    if mongo_manager.documents is not None:
        docs = list(mongo_manager.documents.find(query, {"_id": 0}))
        return [DocumentRecord(**d) for d in docs]
    
    # Fallback in memory
    filtered = [
        DocumentRecord(**d) for d in in_memory_docs.values()
        if include_deleted or not d.get("is_deleted", False)
    ]
    return filtered


@router.get("/{document_id}", response_model=DocumentRecord)
async def get_document(document_id: str):
    """
    Get detailed metadata for a specific document by ID.
    """
    if mongo_manager.documents is not None:
        doc = mongo_manager.documents.find_one({"document_id": document_id}, {"_id": 0})
        if doc:
            return DocumentRecord(**doc)
            
    if document_id in in_memory_docs:
        return DocumentRecord(**in_memory_docs[document_id])

    raise HTTPException(status_code=404, detail=f"Document with ID '{document_id}' not found.")


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(document_id: str, hard_delete: bool = Query(default=False, description="Purge document record permanently")):
    """
    Delete a Document:
    - **Soft Delete (default)**: Sets `is_deleted=True` in MongoDB while purging vector embeddings from Qdrant to immediately disable search.
    - **Hard Delete**: Removes document record completely from MongoDB and purges vector embeddings from Qdrant.
    """
    # 1. Verify existence
    doc_exists = False
    if mongo_manager.documents is not None:
        doc_exists = mongo_manager.documents.find_one({"document_id": document_id}) is not None
    else:
        doc_exists = document_id in in_memory_docs

    if not doc_exists:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    # 2. Purge vector embeddings from Qdrant
    vector_store_service.delete_document_vectors(document_id)

    # 3. Handle Mongo record update/deletion
    soft_deleted = False
    deleted_count = 0

    if mongo_manager.documents is not None:
        if hard_delete:
            mongo_manager.documents.delete_one({"document_id": document_id})
            mongo_manager.chunks.delete_many({"document_id": document_id})
        else:
            mongo_manager.documents.update_one(
                {"document_id": document_id},
                {"$set": {"is_deleted": True, "updated_at": datetime.utcnow()}}
            )
            soft_deleted = True
    else:
        if hard_delete:
            in_memory_docs.pop(document_id, None)
        else:
            in_memory_docs[document_id]["is_deleted"] = True
            soft_deleted = True

    delete_mode_str = "Hard deleted (purged)" if hard_delete else "Soft deleted (flagged)"
    return DeleteDocumentResponse(
        document_id=document_id,
        status="DELETED",
        soft_deleted_in_mongo=soft_deleted,
        purged_from_qdrant=True,
        deleted_chunks_count=deleted_count,
        message=f"Document '{document_id}' successfully {delete_mode_str}. Vectors removed from Qdrant vector store."
    )
