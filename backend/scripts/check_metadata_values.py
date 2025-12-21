"""Check actual metadata values in vector store to diagnose filtering issues."""

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
import structlog

logger = structlog.get_logger(__name__)


async def check_metadata():
    """Check actual metadata values in vector store."""
    
    print(f"\n{'='*80}")
    print("🔍 CHECKING METADATA VALUES IN VECTOR STORE")
    print(f"{'='*80}\n")
    
    # Initialize database
    db_url = get_database_url()
    db_manager = DatabaseSessionManager(database_url=db_url)
    await db_manager.initialize()
    
    try:
        async with db_manager.get_session() as session:
            stmt = select(FolderRecord).order_by(FolderRecord.created_at.desc()).limit(1)
            result = await session.execute(stmt)
            folder = result.scalar_one_or_none()
            
            if not folder:
                print("❌ No folders found")
                return
            
            folder_id = folder.folder_id
            folder_name = folder.folder_name
            print(f"📁 Folder: {folder_name} (ID: {folder_id})\n")
        
        # Initialize vector store
        vector_store = LangChainVectorStore()
        
        # Get all documents
        print("Fetching documents from vector store...\n")
        all_docs = await vector_store.similarity_search(
            query="",
            k=10000,
            filter={"folder_id": folder_id}
        )
        
        print(f"Total documents: {len(all_docs)}\n")
        
        # Group by file_type
        file_types = {}
        file_names = {}
        
        for doc in all_docs:
            if hasattr(doc, 'metadata'):
                meta = doc.metadata
                file_type = meta.get('file_type', 'UNKNOWN')
                file_name = meta.get('file_name', 'UNKNOWN')
                
                if file_type not in file_types:
                    file_types[file_type] = []
                file_types[file_type].append({
                    'file_name': file_name,
                    'chunk_type': meta.get('chunk_type', 'UNKNOWN'),
                })
                
                if file_name not in file_names:
                    file_names[file_name] = {
                        'file_type': file_type,
                        'chunk_count': 0,
                        'chunk_types': set(),
                    }
                file_names[file_name]['chunk_count'] += 1
                file_names[file_name]['chunk_types'].add(meta.get('chunk_type', 'UNKNOWN'))
        
        # Print summary
        print("="*80)
        print("FILE TYPE DISTRIBUTION")
        print("="*80)
        for file_type, items in sorted(file_types.items()):
            unique_files = len(set(item['file_name'] for item in items))
            print(f"\n{file_type}:")
            print(f"  Total chunks: {len(items)}")
            print(f"  Unique files: {unique_files}")
            print(f"  Sample files: {', '.join(set(item['file_name'] for item in items[:5]))}")
        
        print("\n" + "="*80)
        print("FILE-BY-FILE BREAKDOWN")
        print("="*80)
        for file_name, info in sorted(file_names.items()):
            print(f"\n📄 {file_name}:")
            print(f"   file_type: {info['file_type']}")
            print(f"   chunk_count: {info['chunk_count']}")
            print(f"   chunk_types: {', '.join(sorted(info['chunk_types']))}")
        
        # Test specific queries
        print("\n" + "="*80)
        print("TESTING CONTENT TYPE FILTERS")
        print("="*80)
        
        test_filters = [
            {"file_type": "audio"},
            {"file_type": {"$in": ["audio"]}},
            {"file_type": "image"},
            {"file_type": {"$in": ["image"]}},
        ]
        
        for filter_clause in test_filters:
            combined_filter = {"folder_id": folder_id}
            combined_filter.update(filter_clause)
            
            results = await vector_store.similarity_search(
                query="test",
                k=10,
                filter=combined_filter
            )
            
            filter_str = str(filter_clause)
            print(f"\nFilter: {filter_str}")
            print(f"  Results: {len(results)}")
            if results:
                meta = results[0].metadata if hasattr(results[0], 'metadata') else {}
                print(f"  Sample file: {meta.get('file_name', 'Unknown')}")
                print(f"  Sample file_type: {meta.get('file_type', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(check_metadata())

