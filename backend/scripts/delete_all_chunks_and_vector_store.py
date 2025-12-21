"""Delete all chunks from database and vector store."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
if not os.getenv("SECRET_KEY"):
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-purposes-only-123456"

from sqlalchemy import delete, func, select
from app.database.base import DatabaseSessionManager, get_database_url
from app.models.database import ChunkRecord
from app.rag.vector_store import LangChainVectorStore
import structlog

logger = structlog.get_logger(__name__)

async def delete_all_chunks_and_vector_store():
    """Delete all chunks from database and vector store."""
    print(f"\n{'='*60}")
    print(f"🗑️  DELETING ALL CHUNKS FROM DATABASE AND VECTOR STORE")
    print(f"{'='*60}\n")
    
    # Initialize database
    database_url = get_database_url()
    db_manager = DatabaseSessionManager(database_url=database_url)
    await db_manager.initialize()
    
    try:
        async with db_manager.get_session() as session:
            # Count chunks before deletion
            count_stmt = select(func.count(ChunkRecord.chunk_id))
            result = await session.execute(count_stmt)
            chunk_count = result.scalar() or 0
            
            print(f"Found {chunk_count:,} chunks in database")
            
            # Delete all chunks from database
            if chunk_count > 0:
                print(f"\nDeleting all chunks from database...")
                delete_stmt = delete(ChunkRecord)
                await session.execute(delete_stmt)
                await session.commit()
                print(f"✅ Successfully deleted {chunk_count:,} chunks from database")
            else:
                print("No chunks in database to delete.")
        
        # Reset vector store
        print(f"\nResetting vector store...")
        vector_store = LangChainVectorStore()
        vector_store._reset_chromadb_database()
        print(f"✅ Successfully reset vector store\n")
        
    finally:
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(delete_all_chunks_and_vector_store())

