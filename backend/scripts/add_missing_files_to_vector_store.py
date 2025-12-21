"""Add missing files from database to vector store.

This script finds files that have chunks in the SQLite database but are missing
from the ChromaDB vector store, and adds them.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
if not os.getenv("SECRET_KEY"):
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-purposes-only-123456"

from sqlalchemy import select
from app.database.base import DatabaseSessionManager, get_database_url
from app.models.database import FolderRecord, FileRecord, ChunkRecord
from app.rag.vector_store import LangChainVectorStore
from collections import defaultdict

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


async def add_missing_files_to_vector_store(folder_id: str = None):
    """Add missing files from database to vector store.
    
    Args:
        folder_id: Optional specific folder ID. If None, processes all folders.
    """
    print(f"\n{'='*80}")
    print("🔍 FINDING MISSING FILES IN VECTOR STORE")
    print(f"{'='*80}\n")
    
    db_url = get_database_url()
    db_manager = DatabaseSessionManager(database_url=db_url)
    await db_manager.initialize()
    
    try:
        async with db_manager.get_session() as session:
            # Get folder(s)
            if folder_id:
                stmt = select(FolderRecord).where(FolderRecord.folder_id == folder_id)
            else:
                stmt = select(FolderRecord).order_by(FolderRecord.created_at.desc())
            
            folders_result = await session.execute(stmt)
            folders = folders_result.scalars().all()
            
            if not folders:
                print("❌ No folders found")
                return
            
            # Process each folder
            for folder in folders:
                folder_id = folder.folder_id
                print(f"📁 Processing folder: {folder.folder_name} ({folder_id})\n")
                
                # Get all files in this folder
                files_stmt = select(FileRecord).where(FileRecord.folder_id == folder_id)
                files_result = await session.execute(files_stmt)
                files = files_result.scalars().all()
                
                print(f"Found {len(files)} files in database\n")
                
                # Get chunks from database for each file
                db_files_chunks = {}
                for file in files:
                    chunks_stmt = select(ChunkRecord).where(ChunkRecord.file_id == file.file_id)
                    chunks_result = await session.execute(chunks_stmt)
                    chunks = chunks_result.scalars().all()
                    if chunks:
                        db_files_chunks[file.file_name] = {
                            'file': file,
                            'chunks': chunks
                        }
                
                print(f"Files with chunks in DB: {len(db_files_chunks)}\n")
                
    finally:
        await db_manager.close()
    
    # Check vector store
    vector_store = LangChainVectorStore()
    collection = vector_store.vector_store._collection
    
    # Get all documents in vector store for this folder
    try:
        all_docs = collection.get(
            limit=10000,
            where={"folder_id": folder_id},
            include=['metadatas']
        )
    except Exception as e:
        print(f"⚠️  Error querying vector store: {str(e)}")
        print("   Trying without filter...")
        all_docs = collection.get(limit=10000, include=['metadatas'])
    
    # Group by file_name
    vs_files_chunks = defaultdict(list)
    for meta in all_docs['metadatas']:
        file_name = meta.get('file_name', 'unknown')
        vs_files_chunks[file_name].append(meta.get('chunk_id'))
    
    print(f"Files in vector store: {len(vs_files_chunks)}\n")
    
    # Find missing files
    missing_files = {}
    for file_name, data in db_files_chunks.items():
        db_chunk_ids = {chunk.chunk_id for chunk in data['chunks']}
        vs_chunk_ids = set(vs_files_chunks.get(file_name, []))
        
        missing_chunks = db_chunk_ids - vs_chunk_ids
        if missing_chunks:
            missing_files[file_name] = {
                'file': data['file'],
                'chunks': [c for c in data['chunks'] if c.chunk_id in missing_chunks],
                'missing_count': len(missing_chunks),
                'total_count': len(db_chunk_ids)
            }
    
    if not missing_files:
        print("✅ All files are already in vector store!\n")
        return
    
    print(f"{'='*80}")
    print(f"📊 MISSING FILES FOUND: {len(missing_files)}")
    print(f"{'='*80}\n")
    
    for file_name, data in missing_files.items():
        print(f"📄 {file_name}:")
        print(f"   Missing: {data['missing_count']}/{data['total_count']} chunks")
        print(f"   File ID: {data['file'].file_id}\n")
    
    # Add missing chunks to vector store
    print(f"{'='*80}")
    print("➕ ADDING MISSING CHUNKS TO VECTOR STORE")
    print(f"{'='*80}\n")
    
    total_added = 0
    for file_name, data in missing_files.items():
        file = data['file']
        chunks = data['chunks']
        
        print(f"Adding {len(chunks)} chunks for {file_name}...")
        
        try:
            # Convert chunks to LangChain Documents
            documents = []
            for chunk in chunks:
                # Build metadata (filter None values)
                clean_metadata = {
                    "chunk_id": chunk.chunk_id,
                    "file_id": chunk.file_id,
                    "folder_id": folder_id,
                    "file_name": file_name,
                }
                
                # Add chunk metadata, filtering out None values
                if chunk.chunk_metadata:
                    for key, value in chunk.chunk_metadata.items():
                        if value is not None:
                            # Convert value to string if it's not a basic type
                            if isinstance(value, (str, int, float, bool)):
                                clean_metadata[key] = value
                            else:
                                clean_metadata[key] = str(value)
                
                doc = Document(
                    page_content=chunk.content,
                    metadata=clean_metadata
                )
                documents.append(doc)
            
            # Add to vector store
            result_ids = await vector_store.add_documents(documents)
            total_added += len(result_ids)
            
            print(f"✅ Added {len(result_ids)} chunks for {file_name}\n")
            
        except Exception as e:
            print(f"❌ Error adding {file_name}: {str(e)}\n")
            import traceback
            traceback.print_exc()
            print()
    
    print(f"{'='*80}")
    print(f"✅ COMPLETE: Added {total_added} chunks to vector store")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    folder_id = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(add_missing_files_to_vector_store(folder_id))

