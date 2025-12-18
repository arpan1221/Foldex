"""Chat and query API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional

from app.api.deps import get_current_user
from app.services.chat_service import ChatService

router = APIRouter()


class ChatRequest(BaseModel):
    """Request model for chat query."""

    query: str
    folder_id: Optional[str] = None
    conversation_id: Optional[str] = None


class Citation(BaseModel):
    """Citation model."""

    file_id: str
    file_name: str
    chunk_id: str
    page_number: Optional[int] = None
    timestamp: Optional[float] = None
    confidence: float


class ChatResponse(BaseModel):
    """Response model for chat query."""

    response: str
    citations: List[Citation]
    conversation_id: str


@router.post("/query", response_model=ChatResponse)
async def query_chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
) -> ChatResponse:
    """Process a chat query with RAG.

    Args:
        request: Chat query request
        current_user: Current authenticated user

    Returns:
        Chat response with citations

    Raises:
        HTTPException: If query processing fails
    """
    try:
        chat_service = ChatService()
        result = await chat_service.process_query(
            query=request.query,
            folder_id=request.folder_id,
            user_id=current_user.get("sub"),
            conversation_id=request.conversation_id,
        )

        return ChatResponse(
            response=result["response"],
            citations=result["citations"],
            conversation_id=result["conversation_id"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )

