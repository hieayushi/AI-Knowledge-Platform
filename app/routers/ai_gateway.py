import logging
from fastapi import APIRouter, HTTPException
from app.models.query import AIGatewayRequest, AIGatewayResponse
from app.services.rag_service import rag_service

logger = logging.getLogger("ai_platform.routers.ai_gateway")

router = APIRouter(prefix="/ai", tags=["Centralized AI Gateway"])

@router.post("/chat", response_model=AIGatewayResponse)
@router.post("/gateway", response_model=AIGatewayResponse)
async def ai_gateway_chat(request: AIGatewayRequest):
    """
    Centralized AI Gateway Endpoint for Internal Developers.
    Retrieves semantically relevant document & code context and returns formatted AI synthesis.
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        response = rag_service.generate_ai_gateway_answer(request)
        return response
    except Exception as e:
        logger.error(f"Error in AI Gateway execution: {e}")
        raise HTTPException(status_code=500, detail=f"AI Gateway processing failed: {str(e)}")
