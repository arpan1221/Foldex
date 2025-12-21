"""Complete workflow: Process Test Folder and test queries with similarity scores."""

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
from app.services.chat_service import ChatService
from app.database.sqlite_manager import SQLiteManager
from app.utils.file_utils import get_mime_type
import structlog
import uuid
from langchain_core.documents import Document

logger = structlog.get_logger(__name__)

async def process_and_test_folder(folder_name: str = "Test Folder"):
    """Process all files in Test Folder and test queries with similarity scores."""
    
    test_folder_path = Path("/Users/arpannookala/Documents/Foldex/Test Folder")
    
    print(f"\n{'='*60}")
    print(f"🚀 COMPLETE WORKFLOW: PROCESS & TEST")
    print(f"{'='*60}\n")
    
    # Initialize services
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
            
            print(f"📁 Folder: {folder_name}")
            print(f"   ID: {folder_id}\n")
            
            # STEP 1: Process all files
            print(f"{'='*60}")
            print("STEP 1: PROCESSING FILES")
            print("="*60)
            
            if not test_folder_path.exists():
                print(f"❌ Folder not found: {test_folder_path}")
                return
            
            files = [f for f in test_folder_path.iterdir() if f.is_file()]
            print(f"Found {len(files)} files\n")
            
            processed_count = 0
            for file_path in files:
                file_name = file_path.name
                file_size = file_path.stat().st_size
                mime_type = get_mime_type(str(file_path))
                
                print(f"📄 {file_name}")
                print(f"   Size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
                print(f"   Type: {mime_type}")
                
                # Check if already processed
                async with db_manager.get_session() as session:
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
                            print(f"   ✅ Already indexed ({chunk_count} chunks)\n")
                            processed_count += 1
                            continue
                        else:
                            file_id = existing_file.file_id
                    else:
                        file_id = str(uuid.uuid4())
                        await db.store_file_metadata({
                            "id": file_id,
                            "name": file_name,
                            "mimeType": mime_type,
                            "size": file_size,
                        }, folder_id)
                
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
                    
                    # Store chunks
                    await db.store_chunks(chunks)
                    
                    # Store in vector store
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
                    
                    print(f"   ✅ Processed: {len(chunks)} chunks\n")
                    processed_count += 1
                    
                except Exception as e:
                    print(f"   ❌ Error: {str(e)}\n")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # STEP 2: Show indexing status
            print(f"{'='*60}")
            print("STEP 2: INDEXING STATUS")
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
                
                print(f"\n📊 {len([f for f in files])} files, {total_chunks} total chunks\n")
            
            # STEP 3: Test queries with similarity scores
            print(f"{'='*60}")
            print("STEP 3: TESTING QUERIES WITH SIMILARITY SCORES")
            print("="*60)
            
            test_queries = [
                "What are the different file types in this folder?",
                "What is the main topic of the README file?",
                "What information is in the LICENSE file?",
                "What does the CSV file contain?",
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
                    
                    print(f"\n✅ Top {len(results)} Results with Similarity Scores:\n")
                    
                    for i, (doc, score) in enumerate(results, 1):
                        metadata = doc.metadata if hasattr(doc, "metadata") else {}
                        file_name = metadata.get("file_name", "Unknown")
                        chunk_id = metadata.get("chunk_id", "Unknown")
                        content_preview = (doc.page_content[:150] if hasattr(doc, "page_content") 
                                         else str(doc)[:150])
                        
                        # Score interpretation: higher = more similar (for cosine similarity)
                        # Some implementations use distance (lower = better)
                        similarity_score = float(score)
                        
                        print(f"   {i}. Similarity Score: {similarity_score:.4f}")
                        print(f"      File: {file_name}")
                        print(f"      Chunk: {chunk_id[:20]}...")
                        print(f"      Preview: {content_preview}...")
                        print()
                    
                    # Get AI response
                    print(f"   🤖 AI Response:")
                    print(f"   {'-'*56}")
                    
                    chat_service = ChatService()
                    result = await chat_service.process_query(
                        query=query,
                        folder_id=folder_id,
                        user_id=user_id,
                    )
                    
                    response = result.get("response", "")
                    citations = result.get("citations", [])
                    
                    print(f"   {response[:400]}{'...' if len(response) > 400 else ''}")
                    
                    if citations:
                        print(f"\n   📚 Citations ({len(citations)}):")
                        file_citations = {}
                        for cit in citations:
                            file_name = cit.get("file_name", "Unknown")
                            if file_name not in file_citations:
                                file_citations[file_name] = 0
                            file_citations[file_name] += 1
                        
                        for file_name, count in file_citations.items():
                            print(f"      - {file_name}: {count} chunks")
                    print()
                    
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # Update folder status
            await db.update_folder_status(folder_id, "completed")
            
            print(f"\n{'='*60}")
            print("✅ WORKFLOW COMPLETE")
            print("="*60)
            print(f"Processed: {processed_count} files")
            print(f"Total chunks: {total_chunks}")
            print()
                
    finally:
        await db_manager.close()

if __name__ == "__main__":
    folder_name = sys.argv[1] if len(sys.argv) > 1 else "Test Folder"
    asyncio.run(process_and_test_folder(folder_name))

