"""Show top N chunks for each file in the vector store."""

import asyncio
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
if not os.getenv("SECRET_KEY"):
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-purposes-only-123456"

from sqlalchemy import select
from app.database.base import DatabaseSessionManager, get_database_url
from app.models.database import FolderRecord, FileRecord, ChunkRecord
from app.rag.vector_store import LangChainVectorStore
import structlog

logger = structlog.get_logger(__name__)


async def show_top_chunks_per_file(top_n: int = 2):
    """Show top N chunks for each file in the vector store."""
    
    print(f"\n{'='*80}")
    print(f"📊 TOP {top_n} CHUNKS PER FILE IN VECTOR STORE")
    print(f"{'='*80}\n")
    
    # Get folder
    db_url = get_database_url()
    db_manager = DatabaseSessionManager(database_url=db_url)
    await db_manager.initialize()
    
    try:
        async with db_manager.get_session() as session:
            # Get Test Folder by name
            stmt = select(FolderRecord).where(FolderRecord.folder_name == "Test Folder")
            result = await session.execute(stmt)
            folder = result.scalar_one_or_none()
            
            if not folder:
                print('❌ "Test Folder" not found in database')
                # Fallback to most recent folder
                stmt = select(FolderRecord).order_by(FolderRecord.created_at.desc()).limit(1)
                result = await session.execute(stmt)
                folder = result.scalar_one_or_none()
                if not folder:
                    print("❌ No folders found in database")
                    return
            
            folder_id = folder.folder_id
            folder_name = folder.folder_name
            print(f"📁 Folder: {folder_name} (ID: {folder_id})\n")
            
            # Get all files in this folder
            files_stmt = select(FileRecord).where(FileRecord.folder_id == folder_id)
            files_result = await session.execute(files_stmt)
            files = files_result.scalars().all()
            
            print(f"📄 Found {len(files)} files in folder\n")
            
            # Get all chunks for this folder by joining with files
            chunks_stmt = select(ChunkRecord).join(
                FileRecord, ChunkRecord.file_id == FileRecord.file_id
            ).where(FileRecord.folder_id == folder_id)
            chunks_result = await session.execute(chunks_stmt)
            all_chunks = chunks_result.scalars().all()
            
            # Group chunks by file
            chunks_by_file = defaultdict(list)
            for chunk in all_chunks:
                chunks_by_file[chunk.file_id].append(chunk)
            
            # Map file IDs to file names
            file_map = {f.file_id: f.file_name for f in files}
            
    finally:
        await db_manager.close()
    
    # Initialize vector store
    vector_store = LangChainVectorStore()
    
    # For each file, get top chunks using a query based on file name
    print(f"{'='*80}")
    print(f"TOP {top_n} CHUNKS PER FILE")
    print(f"{'='*80}\n")
    print("📊 Note: The 'Score' shown is a similarity score (0.0-1.0) indicating how")
    print("   similar the chunk is to the query used for retrieval. Higher scores")
    print("   (closer to 1.0) indicate better matches. This uses cosine similarity")
    print("   between the query embedding and the chunk embedding.\n")
    
    for file_id, file_name in file_map.items():
        chunks = chunks_by_file.get(file_id, [])
        if not chunks:
            print(f"❌ {file_name}: No chunks found")
            continue
        
        print(f"\n📄 {file_name}")
        print(f"   Total chunks: {len(chunks)}")
        print("-" * 80)
        
        # Try to retrieve chunks using file name as query
        try:
            # Use file name (without extension) as query
            query = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
            
            # Retrieve top chunks for this file
            docs = await vector_store.similarity_search_with_score(
                query=query,
                k=top_n * 2,  # Get more to filter by file
                filter={'folder_id': folder_id, 'file_id': file_id}
            )
            
            # Filter to only this file and take top N
            file_docs = [(doc, score) for doc, score in docs if doc.metadata.get('file_id') == file_id][:top_n]
            
            if not file_docs:
                # Fallback: try without file_id filter
                docs = await vector_store.similarity_search_with_score(
                    query=query,
                    k=top_n * 3,
                    filter={'folder_id': folder_id}
                )
                file_docs = [(doc, score) for doc, score in docs if doc.metadata.get('file_id') == file_id][:top_n]
            
            if file_docs:
                for i, (doc, score) in enumerate(file_docs, 1):
                    chunk_id = doc.metadata.get('chunk_id', 'unknown')
                    chunk_type = doc.metadata.get('chunk_type', 'unknown')
                    content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                    
                    print(f"\n   [{i}] Chunk ID: {chunk_id}")
                    print(f"       Type: {chunk_type}")
                    print(f"       Similarity Score: {score:.4f} (0.0 = dissimilar, 1.0 = very similar)")
                    print(f"       Content: {content_preview}")
                    
                    # Print full metadata
                    print(f"       Full Metadata:")
                    for key, value in sorted(doc.metadata.items()):
                        if isinstance(value, (str, int, float, bool, type(None))):
                            value_str = str(value)
                            # Truncate very long values
                            if len(value_str) > 200:
                                value_str = value_str[:200] + "..."
                            print(f"         {key}: {value_str}")
                        elif isinstance(value, (list, dict)):
                            # Show summary for complex types
                            if isinstance(value, list):
                                print(f"         {key}: [list with {len(value)} items]")
                            else:
                                print(f"         {key}: [dict with {len(value)} keys]")
                        else:
                            print(f"         {key}: <{type(value).__name__}>")
            else:
                # Fallback: show first N chunks from database
                print(f"   (Using database chunks as fallback - no similarity scores available)")
                for i, chunk in enumerate(chunks[:top_n], 1):
                    content_preview = chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
                    chunk_type = chunk.chunk_metadata.get('chunk_type', 'unknown') if chunk.chunk_metadata else 'unknown'
                    print(f"\n   [{i}] Chunk ID: {chunk.chunk_id}")
                    print(f"       Type: {chunk_type}")
                    print(f"       Similarity Score: N/A (from database, not vector store)")
                    print(f"       Content: {content_preview}")
                    
                    # Print full metadata from database
                    if chunk.chunk_metadata:
                        print(f"       Full Metadata:")
                        for key, value in sorted(chunk.chunk_metadata.items()):
                            if isinstance(value, (str, int, float, bool, type(None))):
                                value_str = str(value)
                                if len(value_str) > 200:
                                    value_str = value_str[:200] + "..."
                                print(f"         {key}: {value_str}")
                            elif isinstance(value, (list, dict)):
                                if isinstance(value, list):
                                    print(f"         {key}: [list with {len(value)} items]")
                                else:
                                    print(f"         {key}: [dict with {len(value)} keys]")
                            else:
                                print(f"         {key}: <{type(value).__name__}>")
                    else:
                        print(f"       Full Metadata: (empty)")
                    
        except Exception as e:
            print(f"   ⚠️  Error retrieving chunks: {str(e)}")
            # Fallback: show first N chunks from database
            print(f"   (Using database chunks as fallback - no similarity scores available)")
            for i, chunk in enumerate(chunks[:top_n], 1):
                content_preview = chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
                chunk_type = chunk.chunk_metadata.get('chunk_type', 'unknown') if chunk.chunk_metadata else 'unknown'
                print(f"\n   [{i}] Chunk ID: {chunk.chunk_id}")
                print(f"       Type: {chunk_type}")
                print(f"       Similarity Score: N/A (from database, not vector store)")
                print(f"       Content: {content_preview}")
                
                # Print full metadata from database
                if chunk.chunk_metadata:
                    print(f"       Full Metadata:")
                    for key, value in sorted(chunk.chunk_metadata.items()):
                        if isinstance(value, (str, int, float, bool, type(None))):
                            value_str = str(value)
                            if len(value_str) > 200:
                                value_str = value_str[:200] + "..."
                            print(f"         {key}: {value_str}")
                        elif isinstance(value, (list, dict)):
                            if isinstance(value, list):
                                print(f"         {key}: [list with {len(value)} items]")
                            else:
                                print(f"         {key}: [dict with {len(value)} keys]")
                        else:
                            print(f"         {key}: <{type(value).__name__}>")
                else:
                    print(f"       Full Metadata: (empty)")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(show_top_chunks_per_file(top_n=2))

