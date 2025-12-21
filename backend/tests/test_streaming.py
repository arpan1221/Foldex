"""
Test script to verify streaming works correctly for both LangChain and LangGraph paths.

This script tests:
1. Regular query (LangChain path) - should stream tokens
2. Cross-document synthesis query (LangGraph path) - should stream synthesis

Run with:
    python backend/test_streaming.py <folder_id> <query>
"""

import asyncio
import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.chat_service import ChatService
from app.database.sqlite_manager import SQLiteManager


async def test_streaming(folder_id: str, query: str):
    """Test streaming for a query."""
    
    print("=" * 80)
    print("Streaming Test")
    print("=" * 80)
    print(f"Folder ID: {folder_id}")
    print(f"Query: {query}")
    print()
    
    # Track streaming events
    tokens_received = []
    status_updates = []
    citations_received = []
    
    def streaming_callback(token: str):
        """Callback for streaming tokens."""
        tokens_received.append(token)
        print(f"📝 Token: {repr(token)}")
    
    def status_callback(message: str):
        """Callback for status updates."""
        status_updates.append(message)
        print(f"📊 Status: {message}")
    
    def citations_callback(citations: list):
        """Callback for citations."""
        citations_received.extend(citations)
        print(f"📚 Citations: {len(citations)} received")
    
    try:
        chat_service = ChatService()
        
        print("Starting query processing...")
        print()
        
        result = await chat_service.process_query(
            query=query,
            folder_id=folder_id,
            user_id="test_user",
            streaming_callback=streaming_callback,
            status_callback=status_callback,
            citations_callback=citations_callback,
        )
        
        print()
        print("=" * 80)
        print("Results")
        print("=" * 80)
        print()
        
        print(f"✅ Status Updates: {len(status_updates)}")
        for status in status_updates:
            print(f"   - {status}")
        print()
        
        print(f"✅ Tokens Streamed: {len(tokens_received)}")
        total_chars = sum(len(t) for t in tokens_received)
        print(f"   Total characters: {total_chars}")
        if tokens_received:
            print(f"   First token: {repr(tokens_received[0][:50])}")
            print(f"   Last token: {repr(tokens_received[-1][:50])}")
        print()
        
        print(f"✅ Citations: {len(citations_received)}")
        print()
        
        print(f"✅ Final Response Length: {len(result.get('response', ''))}")
        print()
        
        # Verify streaming worked
        if tokens_received:
            print("✅ STREAMING WORKED - Tokens were received via callback")
        else:
            print("❌ STREAMING FAILED - No tokens received via callback")
            print(f"   Response was: {result.get('response', '')[:100]}...")
        
        return {
            "tokens_received": len(tokens_received),
            "status_updates": len(status_updates),
            "citations_received": len(citations_received),
            "streaming_worked": len(tokens_received) > 0,
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_streaming.py <folder_id> <query>")
        print()
        print("Examples:")
        print("  python test_streaming.py <folder_id> 'What is in this folder?'")
        print("  python test_streaming.py <folder_id> 'What are common entities between files?'")
        sys.exit(1)
    
    folder_id = sys.argv[1]
    query = " ".join(sys.argv[2:])
    
    result = asyncio.run(test_streaming(folder_id, query))
    
    if result:
        print()
        print("=" * 80)
        if result["streaming_worked"]:
            print("✅ STREAMING TEST PASSED")
        else:
            print("❌ STREAMING TEST FAILED")
        print("=" * 80)

