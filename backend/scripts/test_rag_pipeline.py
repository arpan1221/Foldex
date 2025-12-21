"""Test complete RAG pipeline with end-to-end queries."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
if not os.getenv("SECRET_KEY"):
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-purposes-only-123456"

from sqlalchemy import select
from app.database.base import DatabaseSessionManager, get_database_url
from app.models.database import FolderRecord
from app.rag.vector_store import LangChainVectorStore
from app.services.rag_service import get_rag_service
import structlog

logger = structlog.get_logger(__name__)


async def test_rag_pipeline():
    """Test complete RAG pipeline with multiple queries."""
    
    print(f"\n{'='*80}")
    print("🧪 TESTING COMPLETE RAG PIPELINE")
    print(f"{'='*80}\n")
    
    # Initialize database
    db_url = get_database_url()
    db_manager = DatabaseSessionManager(database_url=db_url)
    await db_manager.initialize()
    
    try:
        async with db_manager.get_session() as session:
            # Get the most recent folder
            stmt = select(FolderRecord).order_by(FolderRecord.created_at.desc()).limit(1)
            result = await session.execute(stmt)
            folder = result.scalar_one_or_none()
            
            if not folder:
                print("❌ No folders found in database")
                return
            
            folder_id = folder.folder_id
            folder_name = folder.folder_name
            print(f"📁 Testing folder: {folder_name} (ID: {folder_id})\n")
        
        # Get RAG service
        print("🔧 Initializing RAG service...")
        vector_store = LangChainVectorStore()
        rag_service = get_rag_service(vector_store=vector_store)
        print("✅ RAG service initialized\n")
        
        # Test queries
        test_queries = [
            "What is the total experience of Arpan based on the resume?",
            "What is the folder about?",
            "What does the audio file say?",
            "What is the image about?",
            "Describe the contents of this folder",
            "What does the license file state?",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"{'='*80}")
            print(f"📝 Query {i}/{len(test_queries)}: {query}")
            print(f"{'='*80}\n")
            
            try:
                # Status callback to show progress
                status_messages = []
                def status_callback(msg: str):
                    status_messages.append(msg)
                    print(f"   ℹ️  {msg}")
                
                # Execute query
                result = await rag_service.query(
                    query=query,
                    folder_id=folder_id,
                    status_callback=status_callback,
                )
                
                # Display results
                answer = result.get("answer", "")
                citations = result.get("citations", [])
                sources = result.get("sources", [])
                
                print(f"\n✅ ANSWER:")
                print(f"   {answer}\n")
                
                if citations:
                    print(f"📚 CITATIONS ({len(citations)}):")
                    for j, citation in enumerate(citations[:5], 1):  # Show first 5
                        file_name = citation.get("file_name", "Unknown")
                        file_id = citation.get("file_id", "")
                        chunk_id = citation.get("chunk_id", "")
                        page = citation.get("page_number", "")
                        print(f"   [{j}] {file_name} (ID: {file_id[:8]}..., Chunk: {chunk_id[:16]}...{f', Page: {page}' if page else ''})")
                    if len(citations) > 5:
                        print(f"   ... and {len(citations) - 5} more citations")
                    print()
                
                if sources:
                    print(f"📄 SOURCES ({len(sources)} documents retrieved):")
                    for j, source in enumerate(sources[:5], 1):  # Show first 5
                        if hasattr(source, 'metadata'):
                            meta = source.metadata
                            file_name = meta.get("file_name", "Unknown")
                            chunk_id = meta.get("chunk_id", "")
                            content_preview = source.page_content[:100] + "..." if len(source.page_content) > 100 else source.page_content
                            print(f"   [{j}] {file_name}")
                            print(f"       Chunk: {chunk_id[:32]}...")
                            print(f"       Content: {content_preview}")
                        else:
                            print(f"   [{j}] {str(source)[:100]}...")
                    if len(sources) > 5:
                        print(f"   ... and {len(sources) - 5} more sources")
                    print()
                
                print(f"✅ Query completed successfully\n")
                
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                print()
            
            # Small delay between queries
            await asyncio.sleep(1)
        
        print(f"{'='*80}")
        print("✅ ALL TESTS COMPLETED")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(test_rag_pipeline())

