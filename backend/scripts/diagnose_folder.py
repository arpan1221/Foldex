"""Diagnose folder indexing issues - check both database and vector store."""

import sqlite3
import sys
from pathlib import Path
import asyncio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set SECRET_KEY before importing settings
import os
if not os.getenv("SECRET_KEY"):
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-purposes-only-123456"

from app.database.base import DatabaseSessionManager, get_database_url
from app.models.database import FolderRecord, FileRecord, ChunkRecord
from app.rag.vector_store import LangChainVectorStore
from app.config.settings import settings
from sqlalchemy import select, func

async def diagnose_folder(folder_name: str = "Test Folder"):
    """Diagnose folder indexing issues."""
    print(f"\n{'='*60}")
    print(f"🔍 DIAGNOSING FOLDER: {folder_name}")
    print("="*60)
    
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
            print(f"\n📁 Folder ID: {folder_id}")
            print(f"   Status: {folder.status}")
            
            # Check database chunks
            print(f"\n{'='*60}")
            print("DATABASE CHECK (SQLite)")
            print("="*60)
            
            files_stmt = select(FileRecord).where(FileRecord.folder_id == folder_id)
            files_result = await session.execute(files_stmt)
            files = files_result.scalars().all()
            
            db_chunks = {}
            for file in files:
                chunks_stmt = select(func.count(ChunkRecord.chunk_id)).where(
                    ChunkRecord.file_id == file.file_id
                )
                chunks_result = await session.execute(chunks_stmt)
                chunk_count = chunks_result.scalar() or 0
                db_chunks[file.file_id] = {
                    "file_name": file.file_name,
                    "chunk_count": chunk_count,
                }
                status = "✅" if chunk_count > 0 else "❌"
                print(f"{status} {file.file_name}: {chunk_count} chunks")
            
            total_db_chunks = sum(f["chunk_count"] for f in db_chunks.values())
            print(f"\n📊 Total chunks in database: {total_db_chunks}")
            
            # Check vector store
            print(f"\n{'='*60}")
            print("VECTOR STORE CHECK (ChromaDB)")
            print("="*60)
            
            try:
                vector_store = LangChainVectorStore()
                
                # Try to get all documents for this folder
                print(f"Searching for documents with folder_id: {folder_id}...")
                
                # Use similarity_search with empty query to get all
                documents = await vector_store.similarity_search(
                    query="",
                    k=10000,
                    filter={"folder_id": folder_id}
                )
                
                print(f"✅ Found {len(documents)} documents in vector store")
                
                if documents:
                    # Group by file
                    file_groups = {}
                    for doc in documents[:20]:  # Show first 20
                        metadata = doc.metadata if hasattr(doc, "metadata") else {}
                        file_id = metadata.get("file_id", "unknown")
                        file_name = metadata.get("file_name", "unknown")
                        
                        if file_id not in file_groups:
                            file_groups[file_id] = {"file_name": file_name, "count": 0}
                        file_groups[file_id]["count"] += 1
                    
                    print(f"\nDocuments by file (first 20):")
                    for file_id, info in file_groups.items():
                        print(f"   📄 {info['file_name']}: {info['count']} chunks")
                    
                    if len(documents) > 20:
                        print(f"   ... and {len(documents) - 20} more documents")
                else:
                    print("⚠️  No documents found in vector store!")
                    print(f"\n   This explains why queries return 'no documents found'")
                    print(f"   Even though database has {total_db_chunks} chunks")
                    
                    # Check if there are ANY documents in vector store
                    all_docs = await vector_store.similarity_search(
                        query="",
                        k=100,
                        filter={}  # No filter
                    )
                    print(f"\n   Total documents in vector store (any folder): {len(all_docs)}")
                    
                    if all_docs:
                        # Check folder_ids in vector store
                        folder_ids_found = set()
                        for doc in all_docs[:10]:
                            metadata = doc.metadata if hasattr(doc, "metadata") else {}
                            found_folder_id = metadata.get("folder_id")
                            if found_folder_id:
                                folder_ids_found.add(found_folder_id)
                        
                        print(f"   Folder IDs found in vector store: {list(folder_ids_found)}")
                        if folder_id not in folder_ids_found:
                            print(f"\n   ⚠️  Folder ID mismatch!")
                            print(f"      Expected: {folder_id}")
                            print(f"      Found in vector store: {list(folder_ids_found)}")
                
            except Exception as e:
                print(f"❌ Error checking vector store: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Summary
            print(f"\n{'='*60}")
            print("DIAGNOSIS SUMMARY")
            print("="*60)
            
            if total_db_chunks > 0 and len(documents) == 0:
                print("🔴 PROBLEM IDENTIFIED:")
                print("   - Database has chunks")
                print("   - Vector store is empty")
                print("\n   SOLUTION:")
                print("   - Chunks exist in SQLite but weren't added to vector store")
                print("   - This can happen if processing failed partway through")
                print("   - You may need to re-process the folder")
            elif total_db_chunks == 0:
                print("🔴 PROBLEM IDENTIFIED:")
                print("   - No chunks in database")
                print("   - Files were not successfully processed")
                print("\n   SOLUTION:")
                print("   - Re-process the folder")
                print("   - Check processing logs for errors")
            elif total_db_chunks != len(documents):
                print("⚠️  MISMATCH:")
                print(f"   - Database chunks: {total_db_chunks}")
                print(f"   - Vector store documents: {len(documents)}")
                print("\n   This may cause incomplete search results")
            else:
                print("✅ Everything looks good!")
                print(f"   - {total_db_chunks} chunks in database")
                print(f"   - {len(documents)} documents in vector store")
                
    finally:
        await db_manager.close()

if __name__ == "__main__":
    folder_name = sys.argv[1] if len(sys.argv) > 1 else "Test Folder"
    asyncio.run(diagnose_folder(folder_name))

