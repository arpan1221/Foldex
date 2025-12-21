"""Chat and query API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import structlog
import json
import asyncio

from app.api.deps import get_current_user
from app.services.chat_service import ChatService

logger = structlog.get_logger(__name__)

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
    citations: List[dict]  # Use dict for flexibility
    conversation_id: str


class ConversationResponse(BaseModel):
    """Response model for conversation metadata."""

    conversation_id: str
    folder_id: Optional[str]
    title: str
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    """Response model for chat message."""

    message_id: str
    role: str
    content: str
    citations: Optional[List[dict]]
    timestamp: datetime


@router.get("/folders/{folder_id}/conversations", response_model=List[ConversationResponse])
async def get_folder_conversations(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get all conversations for a folder."""
    user_id = current_user.get("user_id") or current_user.get("sub")
    chat_service = ChatService()
    conversations = await chat_service.db.get_folder_conversations(user_id, folder_id)
    return conversations


@router.post("/folders/{folder_id}/conversations", response_model=ConversationResponse)
async def create_folder_conversation(
    folder_id: str,
    title: Optional[str] = "New Chat",
    current_user: dict = Depends(get_current_user),
):
    """Create a new conversation for a folder."""
    user_id = current_user.get("user_id") or current_user.get("sub")
    chat_service = ChatService()
    conv_id = await chat_service.db.create_conversation(user_id, folder_id, title)
    
    # Get the newly created conversation metadata
    # (In a real app, we'd have a get_conversation method)
    return {
        "conversation_id": conv_id,
        "folder_id": folder_id,
        "title": title,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get all messages for a conversation."""
    chat_service = ChatService()
    messages = await chat_service.db.get_conversation_messages(conversation_id)
    return messages


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a conversation."""
    user_id = current_user.get("user_id") or current_user.get("sub")
    chat_service = ChatService()
    success = await chat_service.db.delete_conversation(conversation_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "success"}


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
        # Extract user_id from current_user dict (from get_current_user)
        user_id = current_user.get("user_id") or current_user.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token",
            )
        
        chat_service = ChatService()
        result = await chat_service.process_query(
            query=request.query,
            folder_id=request.folder_id,
            user_id=user_id,
            conversation_id=request.conversation_id,
        )
        
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Query processing returned None result",
            )
        
        if "response" not in result or "citations" not in result or "conversation_id" not in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Invalid result structure. Expected keys: response, citations, conversation_id. Got: {list(result.keys()) if result else None}",
            )

        return ChatResponse(
            response=result["response"],
            citations=result["citations"],
            conversation_id=result["conversation_id"],
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Chat query failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )


@router.post("/query/stream")
async def query_chat_stream(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """Process a chat query with RAG and stream the response.

    Args:
        request: Chat query request
        current_user: Current authenticated user

    Returns:
        StreamingResponse with Server-Sent Events (SSE) format

    Raises:
        HTTPException: If query processing fails
    """
    try:
        # Extract user_id from current_user dict
        user_id = current_user.get("user_id") or current_user.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token",
            )
        
        chat_service = ChatService()
        
        # Accumulate response for final result
        accumulated_response = ""
        citations = []
        conversation_id = None
        
        async def generate_stream():
            """Generator for streaming response chunks."""
            nonlocal accumulated_response, citations, conversation_id
            
            try:
                # Queue for streaming chunks
                chunk_queue = asyncio.Queue()
                processing_error = None
                final_result = None
                
                # Callback to stream tokens
                def stream_callback(chunk: str):
                    nonlocal accumulated_response
                    accumulated_response += chunk
                    # Put chunk in queue (non-blocking)
                    try:
                        chunk_queue.put_nowait(("token", chunk))
                        logger.debug("Token queued for streaming", chunk_length=len(chunk))
                    except Exception as e:
                        logger.warning("Failed to queue token", error=str(e))

                # Callback to send status updates
                def status_callback(message: str):
                    try:
                        chunk_queue.put_nowait(("status", message))
                        logger.debug("Status update queued", message=message)
                    except Exception as e:
                        logger.warning("Failed to queue status update", error=str(e))

                # Callback to send progressive citations
                def citations_callback(early_citations: list):
                    try:
                        if early_citations and len(early_citations) > 0:
                            chunk_queue.put_nowait(("citations_early", early_citations))
                            logger.info("Citations queued for streaming", count=len(early_citations))
                        else:
                            logger.debug("Empty citations list received, skipping")
                    except Exception as e:
                        logger.warning("Failed to queue citations", error=str(e), exc_info=True)

                # Process query in background
                async def process_query():
                    nonlocal final_result, processing_error
                    try:
                        result = await chat_service.process_query(
                            query=request.query,
                            folder_id=request.folder_id,
                            user_id=user_id,
                            conversation_id=request.conversation_id,
                            streaming_callback=stream_callback,
                            status_callback=status_callback,
                            citations_callback=citations_callback,
                            use_graph_intelligence=True,
                        )
                        final_result = result
                        chunk_queue.put_nowait(("done", None))
                    except Exception as e:
                        processing_error = str(e)
                        chunk_queue.put_nowait(("error", str(e)))
                
                # Start query processing
                query_task = asyncio.create_task(process_query())
                
                # Stream chunks as they arrive
                citations_sent = False
                while True:
                    try:
                        # Wait for chunk with timeout
                        chunk_type, chunk_data = await asyncio.wait_for(chunk_queue.get(), timeout=300.0)

                        if chunk_type == "status":
                            # Status update
                            yield f"data: {json.dumps({'type': 'status', 'message': chunk_data})}\n\n"
                        elif chunk_type == "citations_early":
                            # Progressive citations (sent after retrieval, before generation)
                            citations_sent = True
                            yield f"data: {json.dumps({'type': 'citations', 'citations': chunk_data})}\n\n"
                        elif chunk_type == "token":
                            # Stream token chunk
                            yield f"data: {json.dumps({'type': 'token', 'content': chunk_data})}\n\n"
                        elif chunk_type == "done":
                            # Processing complete
                            await query_task  # Wait for task to complete

                            if final_result is None:
                                yield f"data: {json.dumps({'type': 'error', 'content': 'Query processing returned None result'})}\n\n"
                                break

                            # Store final result
                            citations = final_result.get("citations", [])
                            conversation_id = final_result.get("conversation_id")
                            answer = final_result.get("response", "") or final_result.get("answer", "")

                            # If no tokens were streamed but we have an answer, stream it word by word
                            # This ensures the user always sees streaming, even if callbacks weren't triggered
                            if not accumulated_response and answer:
                                logger.warning(
                                    "No streaming occurred via callbacks, streaming full response word-by-word",
                                    answer_length=len(answer),
                                    accumulated_length=len(accumulated_response)
                                )
                                # Stream the answer word by word to maintain streaming UX
                                words = answer.split()
                                for i, word in enumerate(words):
                                    if i > 0:
                                        yield f"data: {json.dumps({'type': 'token', 'content': ' '})}\n\n"
                                    yield f"data: {json.dumps({'type': 'token', 'content': word})}\n\n"
                                    # Small delay every few words
                                    if i % 5 == 0:
                                        await asyncio.sleep(0.01)

                            # Send citations only if not already sent early (as final fallback)
                            # This ensures citations are always sent, even if early citation callback didn't fire
                            if not citations_sent and citations:
                                logger.info("Sending citations as fallback (not sent early)", citation_count=len(citations))
                                yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
                            elif citations_sent:
                                logger.debug("Citations already sent early, skipping final send")

                            # Send completion
                            done_data = {'type': 'done', 'conversation_id': conversation_id}
                            yield f"data: {json.dumps(done_data)}\n\n"
                            break
                        elif chunk_type == "error":
                            # Error occurred
                            yield f"data: {json.dumps({'type': 'error', 'content': chunk_data})}\n\n"
                            break
                    except asyncio.TimeoutError:
                        yield f"data: {json.dumps({'type': 'error', 'content': 'Request timeout'})}\n\n"
                        break
                
            except Exception as e:
                import traceback
                error_detail = f"{str(e)}\n{traceback.format_exc()}"
                logger.error("Streaming query failed", error=str(e), exc_info=True, traceback=error_detail)
                yield f"data: {json.dumps({'type': 'error', 'content': f'Streaming failed: {str(e)}'})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Streaming chat query failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Streaming query processing failed: {str(e)}",
        )

