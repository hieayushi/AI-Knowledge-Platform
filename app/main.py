import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database.mongo import mongo_manager
from app.database.qdrant import qdrant_manager
from app.routers import documents, query, ai_gateway

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ai_platform.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Internal AI Knowledge Platform Services...")
    # Initialize MongoDB connection
    mongo_manager.connect()
    # Initialize Qdrant persistent storage
    qdrant_manager.initialize()
    yield
    logger.info("Shutting down platform services...")
    mongo_manager.disconnect()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-Ready RAG & AI Knowledge Backend for Internal Developers",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for internal tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(documents.router, prefix=settings.API_V1_PREFIX)
app.include_router(query.router, prefix=settings.API_V1_PREFIX)
app.include_router(ai_gateway.router, prefix=settings.API_V1_PREFIX)

@app.get("/health", tags=["Health & Operations"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "mongo_connected": mongo_manager.client is not None,
        "qdrant_initialized": qdrant_manager.client is not None
    }

@app.get("/", tags=["Health & Operations"])
async def root_info():
    return {
        "message": "Welcome to the Internal AI Knowledge Platform Backend API",
        "docs": "/docs",
        "api_v1": settings.API_V1_PREFIX
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
