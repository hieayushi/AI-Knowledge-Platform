# Internal AI Knowledge Platform - Backend RAG Engine

> **Interview System Design & Implementation Assignment for AI Engineer Position**  
> **Candidate**: Ayushi Sharma  
> **Target Audience**: Internal Engineering Teams (~100 Developers)

---

## Executive Summary

The **Internal AI Knowledge Platform** is a production-ready RAG (Retrieval-Augmented Generation) backend system designed to ingest, chunk, index, and query internal company documents, source code files, and web-scraped content.

It provides high-precision semantic retrieval using a **hybrid search strategy** (Dense Vector Search via Qdrant + Sparse BM25 Keyword Reranking) and exposes a centralized **AI Gateway** endpoint for seamless natural language query synthesis.

---

## Key System Architecture

```mermaid
flowchart TD
    subgraph Client Layer
        DEV[Internal Developer / IDE]
        TEST[Test Verification Suite]
    end

    subgraph API Gateway Layer / FastAPI
        API[FastAPI Application Backend]
        DOC_ROUTER[/documents Endpoints]
        QUERY_ROUTER[/query Endpoint]
        AI_ROUTER[/ai/chat Gateway Endpoint]
    end

    subgraph Processing & Ingestion Services
        INGEST[Ingestion Service]
        PDF_PARSER[PDF Parser pypdf]
        CODE_PARSER[AST / Line Code Chunker]
        BS4_SCRAPER[BeautifulSoup4 Web Scraper]
        EMBED_SVC[SentenceTransformers local all-MiniLM-L6-v2]
    end

    subgraph Hybrid Retrieval Engine
        BM25_RERANK[BM25 Reranker rank-bm25]
        RAG_ORCH[RAG Orchestrator]
    end

    subgraph Storage Layer / Local Persistent
        MONGO[(MongoDB Local on F:\nMetadata, Chunks, Query Logs)]
        QDRANT[(Qdrant Embedded DB\nVector Storage & Payloads)]
    end

    DEV --> API
    TEST --> API
    API --> DOC_ROUTER
    API --> QUERY_ROUTER
    API --> AI_ROUTER

    DOC_ROUTER --> INGEST
    INGEST --> PDF_PARSER
    INGEST --> CODE_PARSER
    INGEST --> BS4_SCRAPER

    INGEST --> EMBED_SVC
    EMBED_SVC --> QDRANT
    INGEST --> MONGO

    QUERY_ROUTER --> RAG_ORCH
    RAG_ORCH --> QDRANT
    RAG_ORCH --> BM25_RERANK
    RAG_ORCH --> MONGO

    AI_ROUTER --> RAG_ORCH
```

---

## Tech Stack & Local Setup

- **Backend Framework**: Python 3.11 with [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/)
- **Vector Database**: [Qdrant](https://qdrant.tech/) Embedded Local Persistent Storage (Zero Docker dependency; stored locally in `./qdrant_data`)
- **Metadata Database**: [MongoDB](https://www.mongodb.com/) Local Instance (`mongodb://localhost:27017` on Drive F:)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, 100% local CPU execution)
- **Web Scraping**: `beautifulsoup4` (`bs4`) + `requests` for clean web content extraction
- **Code Chunker**: Structural boundary chunking with Python line-number tracking
- **Hybrid Reranker**: `rank-bm25` (70% Vector Cosine + 30% BM25 keyword matching)

---

## Database Schemas & Storage Design

### 1. MongoDB Database Design (`ai_knowledge_platform`)

#### Collection: `documents`
Stores master document metadata, processing status, content hashes, and soft-delete flags.

```json
{
  "document_id": "8f4a1c20-d2b3-4f9e-a812-79011f4219b1",
  "title": "Source_Code_Sample.py",
  "file_name": "Source_Code_Sample.py",
  "file_type": "code",
  "file_size_bytes": 7366,
  "content_hash": "a1b2c3d4e5f6...",
  "status": "COMPLETED",
  "error_message": null,
  "chunk_count": 5,
  "tags": ["code", "python", "rotator"],
  "metadata": {
    "chunk_ids": ["...", "..."],
    "extracted_char_count": 7366
  },
  "is_deleted": false,
  "created_at": "2026-07-28T14:20:00.000Z",
  "updated_at": "2026-07-28T14:20:00.000Z"
}
```

#### Collection: `chunks`
Stores textual content breakdown and exact line boundaries for citations.

```json
{
  "chunk_id": "c7a912e4-56b1-5f2a-b912-34901f420001",
  "document_id": "8f4a1c20-d2b3-4f9e-a812-79011f4219b1",
  "chunk_index": 0,
  "content": "class DecayProxyRotator:\n    def __init__(self, proxy_list):\n...",
  "start_line": 7,
  "end_line": 77,
  "char_count": 1850,
  "token_estimate": 462,
  "created_at": "2026-07-28T14:20:00.000Z"
}
```

#### Collection: `query_logs`
Audit log collection tracking natural language queries, response latency, and result counts.

```json
{
  "query": "How does DecayProxyRotator handle proxy penalties?",
  "results_count": 3,
  "execution_time_ms": 14.52,
  "timestamp": "2026-07-28T14:25:00.000Z"
}
```

### 2. Qdrant Vector Collection Schema (`knowledge_chunks`)

- **Vector Dimension**: 384 (Normalized Cosine Distance)
- **Indexed Payload Schema**:
  - `document_id` (Keyword Index)
  - `file_type` (Keyword Index)
  - `tags` (Keyword Array)
  - `content` (Full text payload)
  - `start_line` / `end_line` (Integer ranges)

---

## API Specification

### 1. Upload Document API
- **Endpoint**: `POST /api/v1/documents/upload`
- **Content-Type**: `multipart/form-data`
- **Parameters**: `file` (Binary), `tags` (Optional CSV String)
- **Supported Formats**: PDF (`.pdf`), Python/Code (`.py`, `.js`, `.ts`), Markdown (`.md`), Plain Text (`.txt`)
- **Response** (`201 Created`):
```json
{
  "document_id": "8f4a1c20-d2b3-4f9e-a812-79011f4219b1",
  "file_name": "Source_Code_Sample.py",
  "file_type": "code",
  "status": "COMPLETED",
  "chunk_count": 5,
  "message": "Document 'Source_Code_Sample.py' processed and indexed with 5 chunks."
}
```

### 2. Web URL Scraping Ingestion API
- **Endpoint**: `POST /api/v1/documents/scrape`
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
  "url": "https://docs.python.org/3/library/ast.html",
  "tags": ["web", "python_docs"]
}
```

### 3. Query Semantic Search API
- **Endpoint**: `POST /api/v1/query`
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
  "query": "How does DecayProxyRotator handle proxy penalties?",
  "top_k": 3,
  "filters": {
    "file_type": "code"
  },
  "enable_reranking": true
}
```
- **Response** (`200 OK`):
```json
{
  "query": "How does DecayProxyRotator handle proxy penalties?",
  "total_results": 3,
  "results": [
    {
      "chunk_id": "c7a912e4-56b1-5f2a-b912-34901f420001",
      "document_id": "8f4a1c20-d2b3-4f9e-a812-79011f4219b1",
      "file_name": "Source_Code_Sample.py",
      "file_type": "code",
      "chunk_index": 0,
      "content": "def report_failure(self, proxy):\n    with self.lock:\n        stats = self.proxy_pool[proxy]\n        stats['score'] = 0\n        stats['failure_count'] += 1\n        stats['penalty_factor'] += self.penalty_increment...",
      "similarity_score": 0.8412,
      "rerank_score": 0.8885,
      "start_line": 57,
      "end_line": 68
    }
  ],
  "execution_time_ms": 18.42
}
```

### 4. Delete Document API
- **Endpoint**: `DELETE /api/v1/documents/{document_id}?hard_delete=false`
- **Behavior**:
  - **Soft Delete** (`hard_delete=false`): Immediately purges points from Qdrant vector storage so search results omit the document instantly, while setting `is_deleted: true` in MongoDB for auditing.
  - **Hard Delete** (`hard_delete=true`): Permanently removes document records from MongoDB and purges vector embeddings from Qdrant.

### 5. AI Gateway Endpoint (Local LLM Synthesis)
- **Endpoint**: `POST /api/v1/ai/chat`
- **Behavior**: Uses the retrieved Top-K chunks and feeds them into a local HuggingFace sequence-to-sequence model (`google/flan-t5-base` via the `transformers` pipeline) running natively on CPU to synthesize a natural language response offline.
- **Request Body**:
```json
{
  "prompt": "How does the DecayProxyRotator handle proxy penalties and score recovery?",
  "top_k": 3
}
```

---

## Architectural Deep Dive: Scaling & Trade-offs

### 1. Scaling Strategy (100 -> 10,000 Developers)
- **Asynchronous Task Queuing**: For high-volume file ingestion, delegate document parsing and embedding generation to **Celery / Redis / RabbitMQ** workers to avoid blocking HTTP API workers.
- **Vector DB Sharding**: Migrate Qdrant from embedded mode to a distributed **Qdrant Cluster** with collection replication and HNSW indexing in RAM.
- **Read-Heavy Query Caching**: Implement Redis caching for frequent natural language queries with similarity hashing to serve identical queries in < 2ms.
- **Database Indexing**: Compound MongoDB indexes on `(is_deleted, file_type, created_at)` to support fast document listings.

### 2. Trade-offs & Engineering Decisions
- **Embedded Qdrant vs Docker Qdrant**: Chosen embedded local persistent Qdrant for strict zero-Docker compliance and portable single-host deployment.
- **SentenceTransformer (`all-MiniLM-L6-v2`) vs OpenAI**: Chosen local HuggingFace embeddings for complete privacy, zero latency over network APIs, and 100% cost control.
- **Hybrid Reranking (Vector + BM25)**: Pure vector search can miss exact code symbol names (e.g. `UAFreshnessRotator` or specific variable names). Combining BM25 keyword scoring ensures technical symbols rank at the top.
- **Soft Delete vs Hard Delete**: Soft delete preserves document upload metadata for audit trails while instantly purging Qdrant vectors to fulfill data compliance and retrieval privacy.

---

## Assignment Explanations (Part 2 & Part 3)

### Part 2: Semantic Search Service Details

- **Chunking Strategy**: 
  - **Code**: Uses structural boundary chunking based on Abstract Syntax Trees (AST) and line numbers. It extracts classes, functions, and docstrings intact to prevent breaking logical code blocks.
  - **Text/PDF**: Uses recursive character text splitting with an overlap of 15% to maintain semantic context between adjacent chunks.
- **Embedding Lifecycle**:
  - Documents are ingested, text is cleaned and chunked, and then pushed into a background task queue (FastAPI `BackgroundTasks`).
  - The `SentenceTransformer` model processes the chunks locally on CPU to generate 384-dimensional dense vectors. 
  - Vectors are persisted into Qdrant along with their metadata payload.
- **Similarity Search Approach**: 
  - Qdrant executes an Approximate Nearest Neighbor (ANN) search using **Cosine Distance**. 
  - A pre-filtering step is applied within Qdrant if metadata filters (like `file_type`) are provided.
- **Vector Database Choice**: 
  - **Qdrant (Local Embedded)** was chosen for its zero-dependency portability (doesn't require Docker for local testing), fast HNSW index, and rich payload filtering capabilities.
- **Ranking/Reranking Strategy**: 
  - **Hybrid Search**: We use a two-stage approach. First, Qdrant retrieves the top 50 semantic matches based on dense vectors. Then, `rank-bm25` (Sparse Keyword Matching) reranks these matches to boost chunks containing exact technical keyword matches. Final score = (70% Vector Cosine) + (30% BM25).
- **Failure Handling**: 
  - If embedding generation or OCR fails for a specific document, the database transaction is rolled back, the document status in MongoDB is marked as `FAILED`, and the error stack trace is logged. Partial failures during chunk processing are caught gracefully without crashing the server.

### Part 3: Database Design Details

- **Indexing Strategy**: 
  - **MongoDB**: Compound B-Tree indexes on `(is_deleted, file_type)` and `(document_id)` for lightning-fast queries and status polling.
  - **Qdrant**: HNSW (Hierarchical Navigable Small World) index for vectors, and Keyword payload indexes on `document_id` and `file_type` to allow rapid pre-filtering before vector distance calculation.
- **Partitioning**: 
  - As the system scales to 100+ developers, MongoDB can be partitioned (sharded) based on `file_type` or `team_id`. Qdrant supports horizontal scaling across distributed nodes by partitioning collections based on document hashes.
- **Metadata Modeling**: 
  - Metadata is decoupled. Qdrant holds only the metadata required for filtering (e.g., tags, file_type) and the actual text payload. MongoDB acts as the source of truth for application state, holding extensive document lifecycle data, audit trails, and pagination details.
- **Query Patterns**: 
  - The most common pattern is `Read-Heavy (Search)`. The system is optimized for fast read retrieval by leveraging Qdrant's in-memory HNSW index. 
- **Caching**: 
  - For repeated identical queries, a Redis caching layer (not implemented in the local PoC but designed for production) would hash the incoming query string and filters. If a match is found, the cached Top-K chunks are instantly returned.

---

## Step-by-Step Proof of Execution Guide

Follow these steps in VS Code to run and test the complete system on your local machine:

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start API Server
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Step 3: Execute Automated Verification Test Suite
Open a new terminal tab and run:
```bash
python test_ingest_and_query.py
```

This test script automatically:
1. Ingests `Knowledge_Base_Sample (2).pdf` via `POST /documents/upload`
2. Ingests `Source_Code_Sample (2).py` via `POST /documents/upload`
3. Queries natural language questions regarding `DecayProxyRotator` and document specifications.
4. Synthesizes an answer via the centralized AI Gateway!
