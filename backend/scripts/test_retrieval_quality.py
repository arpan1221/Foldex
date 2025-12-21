"""Test retrieval quality by checking what chunks are retrieved for specific queries."""

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
from app.rag.query_classifier import QueryClassifier, QueryType, get_query_classifier
import structlog

logger = structlog.get_logger(__name__)


async def test_retrieval_quality():
    """Test retrieval quality for specific queries."""
    
    print(f"\n{'='*80}")
    print("🔍 TESTING RETRIEVAL QUALITY")
    print(f"{'='*80}\n")
    
    # Get folder
    db_url = get_database_url()
    db_manager = DatabaseSessionManager(database_url=db_url)
    await db_manager.initialize()
    
    try:
        async with db_manager.get_session() as session:
            # Get the most recent folder (or Test Folder)
            stmt = select(FolderRecord).order_by(FolderRecord.created_at.desc()).limit(1)
            result = await session.execute(stmt)
            folder = result.scalar_one_or_none()
            
            if not folder:
                print("❌ No folders found in database")
                return
            
            folder_id = folder.folder_id
            folder_name = folder.folder_name
            print(f"📁 Testing folder: {folder_name} (ID: {folder_id})\n")
            
            # Get all files in this folder
            files_stmt = select(FileRecord).where(FileRecord.folder_id == folder_id)
            files_result = await session.execute(files_stmt)
            files = files_result.scalars().all()
            
            print(f"📄 Found {len(files)} files in folder\n")
            
            # Map file names to IDs
            file_map = {f.file_name: f.file_id for f in files}
            
            # Get chunk counts for each file
            print("📊 Chunk counts per file:")
            for file in files:
                chunk_stmt = select(ChunkRecord).where(ChunkRecord.file_id == file.file_id)
                chunk_result = await session.execute(chunk_stmt)
                chunks = chunk_result.scalars().all()
                print(f"   {file.file_name}: {len(chunks)} chunks")
            print()
            
            # Prepare available files list for query classifier
            available_files = [{"file_name": f.file_name} for f in files]
            
    finally:
        await db_manager.close()
    
    # Initialize vector store
    vector_store = LangChainVectorStore()
    
    # Initialize query classifier
    classifier = get_query_classifier(available_files=available_files, use_llm=True)
    if classifier.llm:
        print("✅ Query classifier using LLM (llama3.2:1b)")
    else:
        print("⚠️  Query classifier using pattern-based fallback")
    print()
    
    # Test queries related to specific files
    test_queries = [
        ("Describe the contents of this folder", None),
        ("What does the license file state?", "LICENSE.txt"),
        ("What is the screenshot about?", None),  # Should match PNG
        ("What does the audio file say?", None),  # Should match .wav
        ("What is Reconciliation.docx about?", "Reconciliation.docx"),
        ("What is in the README?", "README.md"),
        ("What does the HTML file contain?", None),  # Should match .html
    ]
    
    print(f"{'='*80}")
    print("🔍 TESTING QUERIES")
    print(f"{'='*80}\n")
    
    for query, expected_file in test_queries:
        print(f"Query: \"{query}\"")
        if expected_file:
            print(f"Expected file: {expected_file}")
        print("-" * 80)
        
        # Classify the query
        try:
            query_understanding = await classifier.classify(query)
            print(f"🏷️  Classification:")
            print(f"   Type: {query_understanding.query_type.value}")
            print(f"   Confidence: {query_understanding.confidence:.2%}")
            if query_understanding.content_type:
                print(f"   Content Type: {query_understanding.content_type}")
            if query_understanding.file_references:
                print(f"   File References: {', '.join(query_understanding.file_references)}")
            if query_understanding.entities:
                print(f"   Entities: {', '.join(query_understanding.entities[:5])}")  # First 5
            print(f"   Explanation: {query_understanding.explanation}")
            print()
        except Exception as e:
            print(f"⚠️  Query classification failed: {str(e)}")
            print()
        
        try:
            # Retrieve documents
            docs = await vector_store.similarity_search_with_score(
                query=query,
                k=10,
                filter={'folder_id': folder_id}
            )
            
            print(f"📊 Retrieved {len(docs)} documents\n")
            
            if len(docs) == 0:
                print("❌ NO DOCUMENTS RETRIEVED!\n")
                continue
            
            # Show top 5 results
            print("Top 5 results:")
            for i, (doc, score) in enumerate(docs[:5], 1):
                file_name = doc.metadata.get('file_name', 'unknown')
                chunk_id = doc.metadata.get('chunk_id', 'unknown')[:20]
                content_preview = doc.page_content[:200].replace('\n', ' ')
                
                # Check if this matches expected file
                match_indicator = ""
                if expected_file and file_name == expected_file:
                    match_indicator = " ✅ EXPECTED FILE"
                elif expected_file:
                    match_indicator = " ❌ WRONG FILE"
                
                print(f"  {i}. Score: {score:.4f} | File: {file_name}{match_indicator}")
                print(f"     Chunk ID: {chunk_id}...")
                print(f"     Content: {content_preview}...")
                print()
            
            # Check if expected file is in results
            if expected_file:
                found_expected = any(
                    doc.metadata.get('file_name') == expected_file 
                    for doc, _ in docs[:5]
                )
                if found_expected:
                    print(f"✅ Expected file '{expected_file}' found in top 5 results")
                else:
                    print(f"❌ Expected file '{expected_file}' NOT found in top 5 results")
                    # Show all file names in results
                    result_files = [doc.metadata.get('file_name') for doc, _ in docs]
                    print(f"   Files in results: {set(result_files)}")
            print()
            
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")
            import traceback
            traceback.print_exc()
            print()
    
    print(f"{'='*80}")
    print("✅ RETRIEVAL TEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(test_retrieval_quality())

