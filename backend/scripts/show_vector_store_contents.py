"""Show contents of vector store with sample chunks for each file."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ['SECRET_KEY'] = 'test-secret-key-for-testing-purposes-only-123456'

from app.rag.vector_store import LangChainVectorStore
from app.database.base import DatabaseSessionManager, get_database_url
from sqlalchemy import select, func
from app.models.database import FolderRecord, FileRecord, ChunkRecord


async def show_vector_store_contents():
    """Show all files in vector store with sample chunks."""
    # Get folder
    db_url = get_database_url()
    db_manager = DatabaseSessionManager(database_url=db_url)
    await db_manager.initialize()
    
    try:
        async with db_manager.get_session() as session:
            # Get Test Folder by name
            stmt = select(FolderRecord).where(FolderRecord.folder_name == "Test Folder")
            folder = (await session.execute(stmt)).scalar_one_or_none()
            
            if not folder:
                print('❌ "Test Folder" not found in database')
                # Fallback to most recent folder
                stmt = select(FolderRecord).order_by(FolderRecord.created_at.desc()).limit(1)
                folder = (await session.execute(stmt)).scalar_one_or_none()
                if not folder:
                    print('❌ No folders found in database')
                    return
            
            folder_id = folder.folder_id
            print(f'\n📁 Folder: {folder.folder_name} (ID: {folder_id})\n')
            
            # Get all files
            files_stmt = select(FileRecord).where(FileRecord.folder_id == folder_id)
            files_result = await session.execute(files_stmt)
            files = files_result.scalars().all()
            
            vector_store = LangChainVectorStore()
            
            print('=' * 80)
            print('VECTOR STORE CONTENTS')
            print('=' * 80)
            print()
            
            for file in files:
                print('-' * 80)
                print(f'📄 File: {file.file_name}')
                print(f'   File ID: {file.file_id}')
                print(f'   MIME Type: {file.mime_type}')
                print(f'   Size: {file.size:,} bytes')
                
                # Get chunk count from database
                chunk_count_stmt = select(func.count(ChunkRecord.chunk_id)).where(
                    ChunkRecord.file_id == file.file_id
                )
                chunk_count_result = await session.execute(chunk_count_stmt)
                db_chunk_count = chunk_count_result.scalar_one()
                print(f'   Chunks in DB: {db_chunk_count}')
                
                # Get chunks from vector store using get() method to avoid HNSW index issues
                # This directly queries by metadata instead of using similarity search
                try:
                    chroma_collection = vector_store.vector_store._collection
                    if hasattr(chroma_collection, 'get'):
                        # Use get() method directly with where filter
                        results = chroma_collection.get(
                            where={"$and": [{"folder_id": folder_id}, {"file_id": file.file_id}]},
                            include=['documents', 'metadatas']
                        )
                        
                        # Convert to Document objects
                        from langchain_core.documents import Document
                        docs = []
                        if results and "documents" in results and results["documents"]:
                            for i, doc_text in enumerate(results["documents"]):
                                metadata = {}
                                if "metadatas" in results and results["metadatas"] and i < len(results["metadatas"]):
                                    metadata = results["metadatas"][i]
                                
                                doc = Document(page_content=doc_text, metadata=metadata)
                                docs.append((doc, 1.0))  # Use 1.0 as a placeholder score for get() method
                        
                        print(f'   Chunks in Vector Store (using get()): {len(docs)}')
                    else:
                        # Fallback to similarity_search if get() is not available
                        docs = await vector_store.similarity_search_with_score(
                            query='document',
                            k=100,
                            filter={'folder_id': folder_id, 'file_id': file.file_id}
                        )
                        print(f'   Chunks in Vector Store: {len(docs)}')
                except Exception as e:
                    print(f'   ⚠️  Error retrieving chunks: {str(e)}')
                    docs = []
                    print(f'   Chunks in Vector Store: 0')
                
                print(f'   Chunks in Vector Store: {len(docs)}')
                
                # Show first 3 chunks with full details
                if docs:
                    print(f'\n   Sample chunks (showing first {min(3, len(docs))}):')
                    for i, doc_tuple in enumerate(docs[:3], 1):
                        # Handle both (doc, score) tuples and just doc objects
                        if isinstance(doc_tuple, tuple) and len(doc_tuple) == 2:
                            doc, score = doc_tuple
                        else:
                            doc = doc_tuple
                            score = None
                        print(f'\n   Chunk {i}:')
                        if score is not None:
                            print(f'     Similarity Score: {score:.4f}')
                        else:
                            print(f'     Similarity Score: N/A (retrieved via get() method)')
                        print(f'     Chunk ID: {doc.metadata.get("chunk_id", "N/A")}')
                        print(f'     File ID: {doc.metadata.get("file_id", "N/A")}')
                        print(f'     File Name: {doc.metadata.get("file_name", "N/A")}')
                        print(f'     Element Type: {doc.metadata.get("element_type", "N/A")}')
                        print(f'     Page Number: {doc.metadata.get("page_number", "N/A")}')
                        print(f'     Source File: {doc.metadata.get("source_file", "N/A")}')
                        
                        # Show all metadata keys
                        all_keys = list(doc.metadata.keys())
                        print(f'     All Metadata Keys: {all_keys}')
                        
                        # Show content
                        content_preview = doc.page_content[:400].replace('\n', ' ')
                        print(f'     Content ({len(doc.page_content)} chars): {content_preview}...')
                        
                        # Show full metadata for this chunk
                        print(f'     Full Metadata:')
                        for key, value in doc.metadata.items():
                            if isinstance(value, (str, int, float, bool, type(None))):
                                value_str = str(value)[:100]
                                print(f'       {key}: {value_str}')
                            else:
                                print(f'       {key}: <{type(value).__name__}>')
                    
                    if len(docs) > 3:
                        print(f'\n   ... and {len(docs) - 3} more chunks')
                else:
                    print('   ⚠️  No chunks found in vector store for this file')
                    
    finally:
        await db_manager.close()


if __name__ == '__main__':
    asyncio.run(show_vector_store_contents())

