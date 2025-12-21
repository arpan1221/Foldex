"""Comprehensive workflow test script for Test Folder."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set SECRET_KEY before importing settings
import os
if not os.getenv("SECRET_KEY"):
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-purposes-only-123456"

from sqlalchemy import select, func
from app.database.base import DatabaseSessionManager, get_database_url
from app.models.database import FolderRecord, FileRecord, ChunkRecord
from app.services.chat_service import ChatService
from app.database.sqlite_manager import SQLiteManager
import structlog

logger = structlog.get_logger(__name__)

async def test_workflow(folder_name="Test Folder"):
    """Test complete workflow."""
    print(f"\n{'='*60}")
    print("🚀 FOLDEX WORKFLOW TEST SUITE")
    print(f"{'='*60}\n")
    
    # Initialize database
    database_url = get_database_url()
    db_manager = DatabaseSessionManager(database_url=database_url)
    await db_manager.initialize()
    
    try:
        # Find folder
        async with db_manager.get_session() as session:
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
            print(f"   User: {user_id[:8]}...")
            print(f"   Status: {folder.status}\n")
            
            # Check indexing
            print(f"{'='*60}")
            print("STEP 1: Checking Indexing Status")
            print("="*60)
            
            files_stmt = select(FileRecord).where(FileRecord.folder_id == folder_id)
            files_result = await session.execute(files_stmt)
            files = files_result.scalars().all()
            
            indexed_count = 0
            for file in files:
                chunks_stmt = select(func.count(ChunkRecord.chunk_id)).where(
                    ChunkRecord.file_id == file.file_id
                )
                chunks_result = await session.execute(chunks_stmt)
                chunk_count = chunks_result.scalar() or 0
                if chunk_count > 0:
                    indexed_count += 1
                    print(f"✅ {file.file_name} ({chunk_count} chunks)")
                else:
                    print(f"❌ {file.file_name} (not indexed)")
            
            print(f"\n📊 {indexed_count}/{len(files)} files indexed\n")
            
            if indexed_count == 0:
                print("❌ No files indexed. Cannot test queries.")
                return
            
            # Test queries
            chat_service = ChatService()
            
            queries = [
                ("Simple Query", "What is the main topic of this folder?"),
                ("Cross-Document Query", "Summarize the key information across all documents."),
            ]
            
            for query_type, query in queries:
                print(f"{'='*60}")
                print(f"STEP: {query_type}")
                print("="*60)
                print(f"Query: {query}\n")
                
                try:
                    result = await chat_service.process_query(
                        query=query,
                        folder_id=folder_id,
                        user_id=user_id,
                    )
                    
                    response = result.get("response", "")
                    citations = result.get("citations", [])
                    
                    print(f"✅ Response ({len(response)} chars)")
                    print(f"\n📝 Answer:")
                    print("-" * 60)
                    print(response[:400] + ("..." if len(response) > 400 else ""))
                    print("-" * 60)
                    print(f"\n📚 Citations: {len(citations)}")
                    for i, cit in enumerate(citations[:3], 1):
                        print(f"   {i}. {cit.get('file_name', 'Unknown')}")
                    print()
                except Exception as e:
                    print(f"❌ Error: {str(e)}\n")
                    import traceback
                    traceback.print_exc()
        
    finally:
        await db_manager.close()

if __name__ == "__main__":
    folder_name = sys.argv[1] if len(sys.argv) > 1 else "Test Folder"
    asyncio.run(test_workflow(folder_name))
