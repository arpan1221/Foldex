"""Process all files in Test Folder (without running queries)."""

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
from datetime import datetime
from app.services.document_processor import DocumentProcessor
from app.rag.vector_store import LangChainVectorStore
from app.database.sqlite_manager import SQLiteManager
from app.utils.file_utils import get_mime_type
import structlog
import uuid
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        raise ImportError("LangChain is required. Install with: pip install langchain")

logger = structlog.get_logger(__name__)

async def process_files_only(folder_name: str = "Test Folder", test_folder_path_str: str = None):
    """Process all files in Test Folder without running queries."""
    
    # Default path - works both locally and in Docker if mounted
    if test_folder_path_str is None:
        # Try Docker path first, then local path
        docker_path = Path("/app/../Test Folder")
        local_path = Path("/Users/arpannookala/Documents/Foldex/Test Folder")
        test_folder_path = docker_path if docker_path.exists() else local_path
    else:
        test_folder_path = Path(test_folder_path_str)
    
    print(f"\n{'='*60}")
    print(f"📁 PROCESSING TEST FOLDER")
    print(f"{'='*60}\n")
    print(f"Path: {test_folder_path}")
    
    if not test_folder_path.exists():
        print(f"❌ Folder not found: {test_folder_path}")
        return
    
    # Initialize services
    database_url = get_database_url()
    db_manager = DatabaseSessionManager(database_url=database_url)
    await db_manager.initialize()
    db = SQLiteManager(db_manager=db_manager)  # Pass db_manager to SQLiteManager
    vector_store = LangChainVectorStore()
    doc_processor = DocumentProcessor()
    
    try:
        # Use a single session context for the whole operation
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
            
            # Get all files
            files = [f for f in test_folder_path.iterdir() if f.is_file()]
            print(f"Found {len(files)} files to process\n")
            
            processed_count = 0
            for file_path in files:
                file_name = file_path.name
                file_size = file_path.stat().st_size
                mime_type = get_mime_type(str(file_path))
                
                print(f"📄 {file_name}")
                print(f"   Size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
                print(f"   Type: {mime_type}", end="")
                
                # Check if already processed (reuse session from outer context)
                stmt = select(FileRecord).where(
                    FileRecord.folder_id == folder_id,
                    FileRecord.file_name == file_name
                )
                result = await session.execute(stmt)
                existing_file = result.scalar_one_or_none()
                
                if existing_file:
                    chunks_stmt = select(func.count(ChunkRecord.chunk_id)).where(
                        ChunkRecord.file_id == existing_file.file_id
                    )
                    chunks_result = await session.execute(chunks_stmt)
                    chunk_count = chunks_result.scalar() or 0
                    
                    if chunk_count > 0:
                        print(f" ✅ Already indexed ({chunk_count} chunks)\n")
                        processed_count += 1
                        continue
                    else:
                        file_id = existing_file.file_id
                else:
                    file_id = str(uuid.uuid4())
                    # Store file metadata directly using session
                    file_record = FileRecord(
                        file_id=file_id,
                        folder_id=folder_id,
                        file_name=file_name,
                        mime_type=mime_type,
                        size=file_size,
                        created_at=datetime.utcnow(),
                        modified_at=datetime.utcnow(),
                    )
                    session.add(file_record)
                    await session.commit()
                
                print(f" - Processing...", flush=True)
                
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
                        print(f"   ❌ No chunks extracted\n")
                        continue
                    
                    # Store chunks (filter duplicates first)
                    seen_chunk_ids = set()
                    unique_chunks = []
                    for chunk in chunks:
                        if chunk.chunk_id not in seen_chunk_ids:
                            seen_chunk_ids.add(chunk.chunk_id)
                            unique_chunks.append(chunk)
                        else:
                            print(f"   ⚠️  Skipping duplicate chunk_id: {chunk.chunk_id[:16]}...")
                    
                    if len(unique_chunks) < len(chunks):
                        print(f"   ⚠️  Filtered {len(chunks) - len(unique_chunks)} duplicate chunks")
                    
                    await db.store_chunks(unique_chunks)
                    
                    # Store in vector store
                    # Import Document here (vector_store will handle LangChain availability)
                    try:
                        from langchain_core.documents import Document
                    except ImportError:
                        from langchain.schema import Document
                    
                    documents = []
                    for chunk in chunks:
                        # Filter out None values from metadata (ChromaDB doesn't accept None)
                        clean_metadata = {
                            "chunk_id": chunk.chunk_id,
                            "file_id": chunk.file_id,
                            "folder_id": folder_id,
                            "file_name": file_name,
                        }
                        # Add chunk metadata, filtering out None values
                        if chunk.metadata:
                            for key, value in chunk.metadata.items():
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
                    
                    await vector_store.add_documents(documents)
                    
                    print(f" ✅ Processed: {len(chunks)} chunks\n")
                    processed_count += 1
                    
                except Exception as e:
                    print(f" ❌ Error: {str(e)}\n")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Show final status (using same session)
            print(f"\n{'='*60}")
            print("FINAL STATUS")
            print("="*60)
            
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
            
            print(f"\n📊 Total: {len(files)} files, {total_chunks} chunks")
            print(f"✅ Processed: {processed_count}/{len(files)} files\n")
            
            await db.update_folder_status(folder_id, "completed")
                
    finally:
        await db_manager.close()

if __name__ == "__main__":
    folder_name = sys.argv[1] if len(sys.argv) > 1 else "Test Folder"
    test_folder_path = sys.argv[2] if len(sys.argv) > 2 else None
    asyncio.run(process_files_only(folder_name, test_folder_path))

