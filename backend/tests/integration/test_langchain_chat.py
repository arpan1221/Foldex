"""Integration tests for LangChain chat functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from app.services.chat_service import ChatService
from app.services.langgraph_orchestrator import LangGraphOrchestrator
from app.database.sqlite_manager import SQLiteManager


@pytest.fixture
def mock_orchestrator():
    """Create a mock LangGraph orchestrator."""
    orchestrator = MagicMock(spec=LangGraphOrchestrator)
    orchestrator.process_query_with_graph_intelligence = AsyncMock(
        return_value={
            "answer": "This is a test response from LangGraph.",
            "citations": [
                {
                    "file_id": "test_file_1",
                    "file_name": "test.pdf",
                    "chunk_id": "chunk_1",
                    "page_number": 1,
                }
            ],
            "query_intent": "information_retrieval",
            "confidence": 0.85,
        }
    )
    return orchestrator


@pytest.fixture
def mock_db():
    """Create a mock database manager."""
    db = MagicMock(spec=SQLiteManager)
    db.store_message = AsyncMock()
    db.get_conversation_messages = AsyncMock(return_value=[])
    return db


@pytest.mark.asyncio
async def test_chat_service_initialization():
    """Test that ChatService initializes with LangGraph orchestrator."""
    with patch('app.services.chat_service.LangGraphOrchestrator') as mock_orch_class:
        mock_orch_class.return_value = MagicMock()
        
        service = ChatService()
        
        assert service.orchestrator is not None
        assert service.db is not None


@pytest.mark.asyncio
async def test_process_query_with_langgraph(mock_orchestrator, mock_db):
    """Test processing a query through LangGraph orchestrator."""
    service = ChatService(orchestrator=mock_orchestrator, db=mock_db)
    
    result = await service.process_query(
        query="What is the capital of France?",
        folder_id="test_folder_123",
        user_id="test_user_456",
    )
    
    # Verify orchestrator was called
    mock_orchestrator.process_query_with_graph_intelligence.assert_called_once()
    call_args = mock_orchestrator.process_query_with_graph_intelligence.call_args
    assert call_args.kwargs["query"] == "What is the capital of France?"
    assert call_args.kwargs["folder_id"] == "test_folder_123"
    
    # Verify result structure
    assert "response" in result
    assert "citations" in result
    assert "conversation_id" in result
    assert result["response"] == "This is a test response from LangGraph."
    assert len(result["citations"]) == 1
    assert result["query_intent"] == "information_retrieval"
    assert result["confidence"] == 0.85
    
    # Verify messages were stored
    assert mock_db.store_message.call_count == 2  # User message + assistant response


@pytest.mark.asyncio
async def test_process_query_without_folder(mock_orchestrator, mock_db):
    """Test processing a query without a folder context."""
    service = ChatService(orchestrator=mock_orchestrator, db=mock_db)
    
    result = await service.process_query(
        query="What is AI?",
        folder_id=None,
        user_id="test_user_456",
    )
    
    # Should not call orchestrator without folder
    mock_orchestrator.process_query_with_graph_intelligence.assert_not_called()
    
    # Should return a message asking for folder
    assert "response" in result
    assert "folder" in result["response"].lower()


@pytest.mark.asyncio
async def test_process_query_with_conversation_history(mock_orchestrator, mock_db):
    """Test processing a query with conversation history."""
    # Mock conversation history
    mock_db.get_conversation_messages = AsyncMock(
        return_value=[
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
    )
    
    service = ChatService(orchestrator=mock_orchestrator, db=mock_db)
    
    result = await service.process_query(
        query="Follow-up question",
        folder_id="test_folder_123",
        user_id="test_user_456",
        conversation_id="test_conversation_789",
    )
    
    # Verify conversation history was retrieved
    mock_db.get_conversation_messages.assert_called_once_with("test_conversation_789", limit=10)
    
    # Verify orchestrator received conversation history
    call_args = mock_orchestrator.process_query_with_graph_intelligence.call_args
    assert call_args.kwargs["conversation_history"] is not None
    assert len(call_args.kwargs["conversation_history"]) == 2


@pytest.mark.asyncio
async def test_process_query_error_handling(mock_orchestrator, mock_db):
    """Test error handling in query processing."""
    # Make orchestrator raise an error
    mock_orchestrator.process_query_with_graph_intelligence = AsyncMock(
        side_effect=Exception("Test error")
    )
    
    service = ChatService(orchestrator=mock_orchestrator, db=mock_db)
    
    with pytest.raises(Exception) as exc_info:
        await service.process_query(
            query="Test query",
            folder_id="test_folder_123",
            user_id="test_user_456",
        )
    
    assert "Test error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_citations_structure(mock_orchestrator, mock_db):
    """Test that citations are properly formatted."""
    service = ChatService(orchestrator=mock_orchestrator, db=mock_db)
    
    result = await service.process_query(
        query="Test query",
        folder_id="test_folder_123",
        user_id="test_user_456",
    )
    
    # Verify citations structure
    assert isinstance(result["citations"], list)
    if len(result["citations"]) > 0:
        citation = result["citations"][0]
        assert "file_id" in citation
        assert "file_name" in citation
        assert "chunk_id" in citation


@pytest.mark.asyncio
async def test_streaming_callback_parameter():
    """Test that streaming callback parameter is accepted."""
    with patch('app.services.chat_service.LangGraphOrchestrator') as mock_orch_class:
        mock_orch = MagicMock()
        mock_orch.process_query_with_graph_intelligence = AsyncMock(
            return_value={
                "answer": "Test response",
                "citations": [],
            }
        )
        mock_orch_class.return_value = mock_orch
        
        mock_db = MagicMock(spec=SQLiteManager)
        mock_db.store_message = AsyncMock()
        mock_db.get_conversation_messages = AsyncMock(return_value=[])
        
        service = ChatService(db=mock_db)
        
        # Define a streaming callback
        streamed_chunks = []
        def callback(chunk: str):
            streamed_chunks.append(chunk)
        
        # Process query with streaming callback
        result = await service.process_query(
            query="Test query",
            folder_id="test_folder_123",
            user_id="test_user_456",
            streaming_callback=callback,
        )
        
        # Note: LangGraph orchestrator doesn't currently support streaming,
        # so streamed_chunks will be empty, but the call should not error
        assert "response" in result


@pytest.mark.asyncio
async def test_graph_intelligence_flag(mock_orchestrator, mock_db):
    """Test that graph intelligence flag is passed correctly."""
    service = ChatService(orchestrator=mock_orchestrator, db=mock_db)
    
    # Test with graph intelligence enabled
    await service.process_query(
        query="Test query",
        folder_id="test_folder_123",
        user_id="test_user_456",
        use_graph_intelligence=True,
    )
    
    call_args = mock_orchestrator.process_query_with_graph_intelligence.call_args
    assert call_args.kwargs["use_graph_enhancement"] is True
    
    # Test with graph intelligence disabled
    mock_orchestrator.reset_mock()
    await service.process_query(
        query="Test query",
        folder_id="test_folder_123",
        user_id="test_user_456",
        use_graph_intelligence=False,
    )
    
    call_args = mock_orchestrator.process_query_with_graph_intelligence.call_args
    assert call_args.kwargs["use_graph_enhancement"] is False

