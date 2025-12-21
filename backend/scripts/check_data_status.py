"""Check data status across database, cache, and vector store."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.database.base import initialize_database, get_db_manager
from app.models.database import (
    FolderRecord,
    FileRecord,
    ChunkRecord,
    ConversationRecord,
    MessageRecord,
    UserRecord,
)
from app.rag.vector_store import LangChainVectorStore
from app.utils.caching import get_cache

logger = structlog.get_logger(__name__)


async def check_database():
    """Check database tables for data."""
    print("\n" + "=" * 80)
    print("DATABASE STATUS")
    print("=" * 80)
    
    try:
        await initialize_database()
        db_manager = get_db_manager()
        
        async with db_manager.get_session() as session:
            # Check folders
            folder_count = await session.execute(select(func.count(FolderRecord.folder_id)))
            folders = folder_count.scalar() or 0
            
            # Check files
            file_count = await session.execute(select(func.count(FileRecord.file_id)))
            files = file_count.scalar() or 0
            
            # Check chunks
            chunk_count = await session.execute(select(func.count(ChunkRecord.chunk_id)))
            chunks = chunk_count.scalar() or 0
            
            # Check conversations
            conv_count = await session.execute(select(func.count(ConversationRecord.conversation_id)))
            conversations = conv_count.scalar() or 0
            
            # Check messages
            msg_count = await session.execute(select(func.count(MessageRecord.message_id)))
            messages = msg_count.scalar() or 0
            
            # Check users
            user_count = await session.execute(select(func.count(UserRecord.user_id)))
            users = user_count.scalar() or 0
            
            # Get folder details
            folder_stmt = select(FolderRecord).limit(10)
            folder_result = await session.execute(folder_stmt)
            folder_records = folder_result.scalars().all()
            
            print(f"\n📊 Record Counts:")
            print(f"   Folders:      {folders}")
            print(f"   Files:        {files}")
            print(f"   Chunks:       {chunks}")
            print(f"   Conversations: {conversations}")
            print(f"   Messages:     {messages}")
            print(f"   Users:        {users}")
            
            if folder_records:
                print(f"\n📁 Sample Folders (showing up to 10):")
                for folder in folder_records:
                    print(f"   - {folder.folder_id[:20]}... | {folder.folder_name or 'N/A'} | Status: {folder.status}")
                    
                    # Get file count for this folder
                    file_stmt = select(func.count(FileRecord.file_id)).where(
                        FileRecord.folder_id == folder.folder_id
                    )
                    file_result = await session.execute(file_stmt)
                    folder_file_count = file_result.scalar() or 0
                    
                    # Get chunk count for this folder
                    chunk_stmt = (
                        select(func.count(ChunkRecord.chunk_id))
                        .join(FileRecord, ChunkRecord.file_id == FileRecord.file_id)
                        .where(FileRecord.folder_id == folder.folder_id)
                    )
                    chunk_result = await session.execute(chunk_stmt)
                    folder_chunk_count = chunk_result.scalar() or 0
                    
                    print(f"     Files: {folder_file_count}, Chunks: {folder_chunk_count}")
                    
                    # Check if summary exists
                    if folder.summary:
                        summary_preview = folder.summary[:50] + "..." if len(folder.summary) > 50 else folder.summary
                        print(f"     Summary: {summary_preview}")
            
            # Check knowledge graphs table
            try:
                kg_result = await session.execute(
                    text("SELECT COUNT(*) FROM knowledge_graphs")
                )
                kg_count = kg_result.scalar() or 0
                print(f"\n📈 Knowledge Graphs: {kg_count}")
            except Exception as e:
                print(f"\n📈 Knowledge Graphs: Table may not exist or error: {str(e)}")
            
            total_records = folders + files + chunks + conversations + messages + users
            print(f"\n✅ Total records across all tables: {total_records}")
            
            if total_records == 0:
                print("\n⚠️  Database is empty - no data found")
            else:
                print("\n✅ Database contains data")
                
    except Exception as e:
        print(f"\n❌ Error checking database: {str(e)}")
        import traceback
        traceback.print_exc()


async def check_vector_store():
    """Check vector store (ChromaDB) for data."""
    print("\n" + "=" * 80)
    print("VECTOR STORE STATUS")
    print("=" * 80)
    
    try:
        vector_store = LangChainVectorStore()
        
        # Get all collections
        try:
            collections = vector_store._collection.get() if hasattr(vector_store, '_collection') else None
            
            if collections and collections.get('ids'):
                ids = collections['ids']
                count = len(ids)
                print(f"\n📊 Vector Store Contents:")
                print(f"   Total embeddings: {count}")
                
                if count > 0:
                    # Sample some IDs
                    sample_ids = ids[:5]
                    print(f"\n   Sample chunk IDs (first 5):")
                    for chunk_id in sample_ids:
                        print(f"     - {chunk_id}")
                    
                    # Try to get metadata for a sample
                    if 'metadatas' in collections and collections['metadatas']:
                        sample_metadata = collections['metadatas'][0] if collections['metadatas'] else None
                        if sample_metadata:
                            print(f"\n   Sample metadata:")
                            for key, value in list(sample_metadata.items())[:5]:
                                print(f"     {key}: {value}")
                else:
                    print("\n⚠️  Vector store is empty - no embeddings found")
            else:
                print("\n⚠️  Could not retrieve collection data or collection is empty")
                print("   This might mean ChromaDB hasn't been initialized yet")
                
        except Exception as e:
            print(f"\n⚠️  Error accessing vector store collection: {str(e)}")
            print("   Vector store may not be initialized or collection may not exist")
            
    except Exception as e:
        print(f"\n❌ Error checking vector store: {str(e)}")
        import traceback
        traceback.print_exc()


async def check_cache():
    """Check cache status."""
    print("\n" + "=" * 80)
    print("CACHE STATUS")
    print("=" * 80)
    
    try:
        cache = get_cache()
        stats = cache.get_stats()
        
        print(f"\n📊 Cache Statistics:")
        print(f"   Hits:        {stats.get('hits', 0)}")
        print(f"   Misses:      {stats.get('misses', 0)}")
        print(f"   Size:        {stats.get('size', 0)}")
        print(f"   Max Size:    {stats.get('max_size', 0)}")
        
        if stats.get('size', 0) > 0:
            print("\n✅ Cache contains data")
        else:
            print("\n⚠️  Cache is empty")
            
    except Exception as e:
        print(f"\n❌ Error checking cache: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function to check all data stores."""
    print("\n" + "=" * 80)
    print("FOLDEX DATA STATUS CHECK")
    print("=" * 80)
    
    await check_database()
    await check_vector_store()
    await check_cache()
    
    print("\n" + "=" * 80)
    print("CHECK COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
