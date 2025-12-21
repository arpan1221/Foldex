"""Process all files in Test Folder locally and test queries with similarity scores."""

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
from app.services.document_processor import DocumentProcessor
from app.rag.vector_store import LangChainVectorStore
from app.database.sqlite_manager import SQLiteManager
import structlog
import uuid
from datetime import datetime

logger = structlog.get_logger(__name__)

async def process_test_folder():
    """Process all files in Test Folder locally."""
    test_folder_path = Path("/Users/arpannookala/Documents/Foldex/Test Folder")
    folder_name = "Test Folder"
    
    print(f"\n{'='*60}")
    print(f"📁 PROCESSING TEST FOLDER LOCALLY")
    print(f"{'='*60}\n")
    print(f"Path: {test_folder_path}")
    
    if not test_folder_path.exists():
        print(f"❌ Folder not found: {test_folder_path}")
        return
    
    # Initialize database
    database_url = get_database_url()
    db_manager = DatabaseSessionManager(database_url=database_url)
    await db_manager.initialize()
    db = SQLiteManager()
    vector_store = LangChainVectorStore()
    doc_processor = DocumentProcessor()
    
    try:
        async with db_manager.get_session() as session:
            # Find or create folder
            stmt = select(FolderRecord).where(FolderRecord.folder_name == folder_name)
            result = await session.execute(stmt)
            folder = result.scalar_one_or_none()
            
            if not folder:
                folder_id = str(uuid.uuid4())
                user_id = "test_user"
                await db.store_folder(
                    folder_id=folder_id,
                    user_id=user_id,
                    folder_name=folder_name,
                    file_count=0,
                    status="processing"
                )
            else:
                folder_id = folder.folder_id
                user_id = folder.user_id
            
            print(f"📁 Folder ID: {folder_id}\n")
            
            # Get all files in folder
            files = list(test_folder_path.iterdir())
            files = [f for f in files if f.is_file()]
            
            print(f"📄 Found {len(files)} files to process:\n")
            
            # Process each file
            for file_path in files:
                file_name = file_path.name
                file_size = file_path.stat().st_size
                
                print(f"{'='*60}")
                print(f"Processing: {file_name}")
                print(f"   Size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
                
                # Check if already processed
                async with db_manager.get_session() as session:
                    stmt = select(FileRecord).where(
                        FileRecord.folder_id == folder_id,
                        FileRecord.file_name == file_name
                    )
                    result = await session.execute(stmt)
                    existing_file = result.scalar_one_or_none()
                    
                    if existing_file:
                        # Check chunk count
                        chunks_stmt = select(func.count(ChunkRecord.chunk_id)).where(
                            ChunkRecord.file_id == existing_file.file_id
                        )
                        chunks_result = await session.execute(chunks_stmt)
                        chunk_count = chunks_result.scalar() or 0
                        
                        if chunk_count > 0:
                            print(f"   ✅ Already indexed ({chunk_count} chunks)")
                            continue
                        else:
                            print(f"   ⚠️  File exists but has no chunks, re-processing...")
                            file_id = existing_file.file_id
                    else:
                        # Create new file record
                        file_id = str(uuid.uuid4())
                        await db.store_file_metadata({
                            "id": file_id,
                            "name": file_name,
                            "mimeType": get_mime_type(file_path),
                            "size": file_size,
                        }, folder_id)
                
                # Determine MIME type
                mime_type = get_mime_type(str(file_path))
                print(f"   MIME Type: {mime_type}")
                
                # Process file
                try:
                    chunks = await doc_processor.process_file(
                        str(file_path),
                        {
                            "id": file_id,
                            "name": file_name,
                            "mimeType": mime_type,
                            "size": file_size,
                        }
                    )
                    
                    if not chunks:
                        print(f"   ❌ No chunks extracted")
                        continue
                    
                    # Store chunks in database
                    await db.store_chunks(chunks)
                    
                    # Store in vector store
                    from langchain_core.documents import Document
                    documents = []
                    for chunk in chunks:
                        doc = Document(
                            page_content=chunk.content,
                            metadata={
                                "chunk_id": chunk.chunk_id,
                                "file_id": chunk.file_id,
                                "folder_id": folder_id,
                                "file_name": file_name,
                                **chunk.metadata
                            }
                        )
                        documents.append(doc)
                    
                    await vector_store.add_documents(documents)
                    
                    print(f"   ✅ Processed: {len(chunks)} chunks")
                    
                except Exception as e:
                    print(f"   ❌ Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Update folder status
            await db.update_folder_status(folder_id, "completed")
            
            # Show final status
            print(f"\n{'='*60}")
            print("FINAL INDEXING STATUS")
            print("="*60)
            
            async with db_manager.get_session() as session:
                files_stmt = select(FileRecord).where(FileRecord.folder_id == folder_id)
                files_result = await session.execute(files_stmt)
                files = files_result.scalars().all()
                
                total_chunks = 0
                for file in files:
                    chunks_stmt = select(func.count(ChunkRecord.chunk_id)).where(
                        ChunkRecord.file_id == file.file_id
                    )
                    chunks_result = await session.execute(chunks_stmt)
                    chunk_count = chunks_result.scalar() or 0
                    total_chunks += chunk_count
                    
                    status = "✅" if chunk_count > 0 else "❌"
                    print(f"{status} {file.file_name}: {chunk_count} chunks")
                
                print(f"\n📊 Total: {len(files)} files, {total_chunks} chunks\n")
                
    finally:
        await db_manager.close()

from app.utils.file_utils import get_mime_type

if __name__ == "__main__":
    asyncio.run(process_test_folder())

