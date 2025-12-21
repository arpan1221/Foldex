"""Test individual RAG components to diagnose issues."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
if not os.getenv("SECRET_KEY"):
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-purposes-only-123456"

from sqlalchemy import select
from app.database.base import DatabaseSessionManager, get_database_url
from app.models.database import FolderRecord
from app.rag.vector_store import LangChainVectorStore
from app.rag.retrievers import AdaptiveRetriever
import structlog

logger = structlog.get_logger(__name__)


async def test_components():
    """Test individual RAG components."""
    
    print(f"\n{'='*80}")
    print("🔍 TESTING RAG COMPONENTS")
    print(f"{'='*80}\n")
    
    # Initialize database
    db_url = get_database_url()
    db_manager = DatabaseSessionManager(database_url=db_url)
    await db_manager.initialize()
    
    try:
        async with db_manager.get_session() as session:
            stmt = select(FolderRecord).order_by(FolderRecord.created_at.desc()).limit(1)
            result = await session.execute(stmt)
            folder = result.scalar_one_or_none()
            
            if not folder:
                print("❌ No folders found")
                return
            
            folder_id = folder.folder_id
            folder_name = folder.folder_name
            print(f"📁 Folder: {folder_name} (ID: {folder_id})\n")
        
        # Initialize vector store
        print("🔧 Initializing vector store...")
        vector_store = LangChainVectorStore()
        print("✅ Vector store initialized\n")
        
        # Test 1: Direct retrieval
        print("="*80)
        print("TEST 1: Direct Vector Store Retrieval")
        print("="*80)
        query = "What does the audio file say?"
        
        results = await vector_store.similarity_search_with_score(
            query=query,
            k=5,
            filter={"folder_id": folder_id}
        )
        
        print(f"\n✅ Retrieved {len(results)} documents")
        for i, (doc, score) in enumerate(results[:3], 1):
            print(f"\n[{i}] Score: {score:.4f}")
            if hasattr(doc, 'metadata'):
                meta = doc.metadata
                print(f"    File: {meta.get('file_name', 'Unknown')}")
                print(f"    File Type: {meta.get('file_type', 'Unknown')}")
                print(f"    Content: {doc.page_content[:100]}...")
        
        # Test 2: Retrieval with content type filter
        print("\n" + "="*80)
        print("TEST 2: Retrieval with Audio Content Type Filter")
        print("="*80)
        
        results_audio = await vector_store.similarity_search_with_score(
            query=query,
            k=5,
            filter={"folder_id": folder_id, "file_type": {"$in": ["audio"]}}
        )
        
        print(f"\n✅ Retrieved {len(results_audio)} audio documents")
        for i, (doc, score) in enumerate(results_audio[:3], 1):
            print(f"\n[{i}] Score: {score:.4f}")
            if hasattr(doc, 'metadata'):
                meta = doc.metadata
                print(f"    File: {meta.get('file_name', 'Unknown')}")
                print(f"    Content: {doc.page_content[:100]}...")
        
        # Test 3: Adaptive Retriever
        print("\n" + "="*80)
        print("TEST 3: Adaptive Retriever (BM25)")
        print("="*80)
        
        # Load documents for BM25
        all_docs = await vector_store.similarity_search(
            query="",
            k=10000,
            filter={"folder_id": folder_id}
        )
        
        retriever = AdaptiveRetriever(
            vector_store=vector_store,
            documents=all_docs[:100],  # Limit for BM25
            folder_id=folder_id,
            k=5,
        )
        
        docs = await retriever.aget_relevant_documents(query)
        print(f"\n✅ Retrieved {len(docs)} documents via AdaptiveRetriever")
        for i, doc in enumerate(docs[:3], 1):
            print(f"\n[{i}]")
            if hasattr(doc, 'metadata'):
                meta = doc.metadata
                print(f"    File: {meta.get('file_name', 'Unknown')}")
                print(f"    File Type: {meta.get('file_type', 'Unknown')}")
                print(f"    Content: {doc.page_content[:100]}...")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(test_components())

