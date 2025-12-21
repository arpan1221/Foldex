"""Test queries with similarity scores for Test Folder."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
if not os.getenv("SECRET_KEY"):
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-purposes-only-123456"

from sqlalchemy import select, func
from app.database.base import DatabaseSessionManager, get_database_url
from app.models.database import FolderRecord, FileRecord, ChunkRecord
from app.rag.vector_store import LangChainVectorStore
from app.services.chat_service import ChatService
import structlog

logger = structlog.get_logger(__name__)

async def test_queries_with_scores(folder_name: str = "Test Folder"):
    """Test queries and show similarity scores for each chunk."""
    print(f"\n{'='*60}")
    print(f"🔍 TESTING QUERIES WITH SIMILARITY SCORES")
    print(f"{'='*60}\n")
    
    # Initialize database
    database_url = get_database_url()
    db_manager = DatabaseSessionManager(database_url=database_url)
    await db_manager.initialize()
    
    try:
        async with db_manager.get_session() as session:
            # Find folder
            stmt = select(FolderRecord).where(FolderRecord.folder_name == folder_name)
            result = await session.execute(stmt)
            folder = result.scalar_one_or_none()
            
            if not folder:
                print(f"❌ Folder '{folder_name}' not found")
                return
            
            folder_id = folder.folder_id
            user_id = folder.user_id
            
            print(f"📁 Folder: {folder.folder_name}")
            print(f"   ID: {folder_id}")
            print(f"   Status: {folder.status}\n")
            
            # Check indexing status
            print(f"{'='*60}")
            print("INDEXING STATUS")
            print("="*60)
            
            files_stmt = select(FileRecord).where(FileRecord.folder_id == folder_id)
            files_result = await session.execute(files_stmt)
            files = files_result.scalars().all()
            
            indexed_files = []
            total_chunks = 0
            
            for file in files:
                chunks_stmt = select(func.count(ChunkRecord.chunk_id)).where(
                    ChunkRecord.file_id == file.file_id
                )
                chunks_result = await session.execute(chunks_stmt)
                chunk_count = chunks_result.scalar() or 0
                total_chunks += chunk_count
                
                status = "✅" if chunk_count > 0 else "❌"
                print(f"{status} {file.file_name}")
                print(f"   MIME: {file.mime_type}")
                print(f"   Size: {file.size:,} bytes")
                print(f"   Chunks: {chunk_count}")
                print()
                
                if chunk_count > 0:
                    indexed_files.append(file)
            
            print(f"📊 Summary: {len(indexed_files)}/{len(files)} files indexed, {total_chunks} total chunks\n")
            
            if total_chunks == 0:
                print("❌ No chunks found. Please process the folder first.")
                return
            
            # Initialize vector store
            print(f"{'='*60}")
            print("INITIALIZING VECTOR STORE")
            print("="*60)
            
            vector_store = LangChainVectorStore()
            
            # Test queries
            test_queries = [
                "What are the different file types in this folder?",
                "What is the main topic of the README file?",
                "What information is in the LICENSE file?",
                "Summarize the key content across all documents.",
            ]
            
            for query in test_queries:
                print(f"\n{'='*60}")
                print(f"QUERY: {query}")
                print("="*60)
                
                try:
                    # Get similar documents with scores
                    results = await vector_store.similarity_search_with_score(
                        query=query,
                        k=10,
                        filter={"folder_id": folder_id}
                    )
                    
                    if not results:
                        print("❌ No results found")
                        continue
                    
                    print(f"\n✅ Found {len(results)} results:\n")
                    
                    for i, (doc, score) in enumerate(results, 1):
                        metadata = doc.metadata if hasattr(doc, "metadata") else {}
                        file_name = metadata.get("file_name", "Unknown")
                        chunk_id = metadata.get("chunk_id", "Unknown")
                        content_preview = doc.page_content[:200] if hasattr(doc, "page_content") else str(doc)[:200]
                        
                        # Similarity score (higher is more similar for cosine similarity)
                        # For some implementations, score might be distance (lower is better)
                        similarity_score = float(score)
                        
                        print(f"   {i}. Score: {similarity_score:.4f}")
                        print(f"      File: {file_name}")
                        print(f"      Chunk ID: {chunk_id[:16]}...")
                        print(f"      Content: {content_preview}...")
                        print()
                    
                    # Also test via ChatService to see full response
                    print(f"\n   🤖 Full AI Response:")
                    print(f"   {'-'*56}")
                    
                    chat_service = ChatService()
                    result = await chat_service.process_query(
                        query=query,
                        folder_id=folder_id,
                        user_id=user_id,
                    )
                    
                    response = result.get("response", "")
                    citations = result.get("citations", [])
                    
                    print(f"   {response[:300]}{'...' if len(response) > 300 else ''}")
                    print(f"\n   📚 Citations: {len(citations)}")
                    for cit in citations[:3]:
                        print(f"      - {cit.get('file_name', 'Unknown')}")
                    print()
                    
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
    finally:
        await db_manager.close()

if __name__ == "__main__":
    folder_name = sys.argv[1] if len(sys.argv) > 1 else "Test Folder"
    asyncio.run(test_queries_with_scores(folder_name))

