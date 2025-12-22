"""Wipe all data from database and vector store.

This script deletes all folder indexes, user data, chat history, and vector embeddings.
Use with caution - this is irreversible!

IMPORTANT: This will also delete all OAuth refresh tokens. Users will need to
re-authenticate with Google after running this script.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
if not os.getenv("SECRET_KEY"):
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-purposes-only-123456"

from sqlalchemy import delete, func, select
from app.database.base import DatabaseSessionManager, get_database_url
from app.models.database import (
    ChunkRecord,
    FileRecord,
    FolderRecord,
    ConversationRecord,
    MessageRecord,
    UserRecord,
    KnowledgeGraphRecord,
    ProcessingCacheRecord,
)
from app.rag.vector_store import LangChainVectorStore
from app.utils.cache import FileCache, Cache
from app.config.settings import settings
import structlog
import shutil

logger = structlog.get_logger(__name__)


async def wipe_all_data():
    """Wipe all data from database and vector store."""
    print(f"\n{'='*60}")
    print(f"🗑️  WIPING ALL DATA FROM DATABASE AND VECTOR STORE")
    print(f"{'='*60}\n")
    print("⚠️  WARNING: This will delete:")
    print("   - All folder indexes")
    print("   - All file metadata")
    print("   - All document chunks")
    print("   - All chat conversations and messages")
    print("   - All user accounts (including OAuth refresh tokens)")
    print("   - All knowledge graphs")
    print("   - All processing cache")
    print("   - All vector embeddings")
    print("\n⚠️  IMPORTANT: Users will need to re-authenticate with Google")
    print("   after this operation as OAuth refresh tokens will be deleted.\n")
    print("⚠️  This action is IRREVERSIBLE!\n")
    
    # Initialize database
    database_url = get_database_url()
    db_manager = DatabaseSessionManager(database_url=database_url)
    await db_manager.initialize()
    
    try:
        async with db_manager.get_session() as session:
            # Count records before deletion
            counts = {}
            
            # Count chunks
            count_stmt = select(func.count(ChunkRecord.chunk_id))
            counts['chunks'] = (await session.execute(count_stmt)).scalar() or 0
            
            # Count files
            count_stmt = select(func.count(FileRecord.file_id))
            counts['files'] = (await session.execute(count_stmt)).scalar() or 0
            
            # Count folders
            count_stmt = select(func.count(FolderRecord.folder_id))
            counts['folders'] = (await session.execute(count_stmt)).scalar() or 0
            
            # Count messages
            count_stmt = select(func.count(MessageRecord.message_id))
            counts['messages'] = (await session.execute(count_stmt)).scalar() or 0
            
            # Count conversations
            count_stmt = select(func.count(ConversationRecord.conversation_id))
            counts['conversations'] = (await session.execute(count_stmt)).scalar() or 0
            
            # Count users
            count_stmt = select(func.count(UserRecord.user_id))
            counts['users'] = (await session.execute(count_stmt)).scalar() or 0
            
            # Count knowledge graphs
            count_stmt = select(func.count(KnowledgeGraphRecord.folder_id))
            counts['knowledge_graphs'] = (await session.execute(count_stmt)).scalar() or 0
            
            # Count processing cache (if table exists)
            try:
                count_stmt = select(func.count(ProcessingCacheRecord.cache_key))
                counts['processing_cache'] = (await session.execute(count_stmt)).scalar() or 0
            except Exception:
                counts['processing_cache'] = 0
            
            print("📊 Current data counts:")
            for table, count in counts.items():
                print(f"   {table}: {count:,} records")
            print()
            
            # Delete in order (respecting foreign key constraints)
            print("🗑️  Deleting data...\n")
            
            # 1. Delete chunks first (no dependencies)
            if counts['chunks'] > 0:
                print(f"   Deleting {counts['chunks']:,} chunks...")
                delete_stmt = delete(ChunkRecord)
                await session.execute(delete_stmt)
                await session.commit()
                print(f"   ✅ Deleted chunks")
            
            # 2. Delete messages (depends on conversations)
            if counts['messages'] > 0:
                print(f"   Deleting {counts['messages']:,} messages...")
                delete_stmt = delete(MessageRecord)
                await session.execute(delete_stmt)
                await session.commit()
                print(f"   ✅ Deleted messages")
            
            # 3. Delete conversations
            if counts['conversations'] > 0:
                print(f"   Deleting {counts['conversations']:,} conversations...")
                delete_stmt = delete(ConversationRecord)
                await session.execute(delete_stmt)
                await session.commit()
                print(f"   ✅ Deleted conversations")
            
            # 4. Delete files (depends on folders)
            if counts['files'] > 0:
                print(f"   Deleting {counts['files']:,} files...")
                delete_stmt = delete(FileRecord)
                await session.execute(delete_stmt)
                await session.commit()
                print(f"   ✅ Deleted files")
            
            # 5. Delete knowledge graphs (depends on folders)
            if counts['knowledge_graphs'] > 0:
                print(f"   Deleting {counts['knowledge_graphs']:,} knowledge graphs...")
                delete_stmt = delete(KnowledgeGraphRecord)
                await session.execute(delete_stmt)
                await session.commit()
                print(f"   ✅ Deleted knowledge graphs")
            
            # 6. Delete processing cache
            if counts['processing_cache'] > 0:
                print(f"   Deleting {counts['processing_cache']:,} cache entries...")
                delete_stmt = delete(ProcessingCacheRecord)
                await session.execute(delete_stmt)
                await session.commit()
                print(f"   ✅ Deleted processing cache")
            
            # 7. Delete folders
            if counts['folders'] > 0:
                print(f"   Deleting {counts['folders']:,} folders...")
                delete_stmt = delete(FolderRecord)
                await session.execute(delete_stmt)
                await session.commit()
                print(f"   ✅ Deleted folders")
            
            # 8. Delete users (keep this last, might be referenced elsewhere)
            if counts['users'] > 0:
                print(f"   Deleting {counts['users']:,} users...")
                delete_stmt = delete(UserRecord)
                await session.execute(delete_stmt)
                await session.commit()
                print(f"   ✅ Deleted users")
        
        # Reset vector store
        print(f"\n🗑️  Resetting vector store...")
        vector_store = LangChainVectorStore()
        
        # First, try to delete all documents from the collection
        try:
            collection = vector_store.vector_store._collection
            count = collection.count()
            if count > 0:
                print(f"   Deleting {count:,} documents from collection...")
                all_ids = collection.get()['ids']
                if all_ids:
                    collection.delete(ids=all_ids)
                print(f"   ✅ Deleted all documents from collection")
        except Exception as e:
            print(f"   ⚠️  Could not delete documents directly: {str(e)}")
        
        # Reset the database (removes SQLite files)
        vector_store._reset_chromadb_database()
        
        # Also clean up any remaining collection directories
        persist_dir = vector_store.persist_directory
        if persist_dir and persist_dir.exists():
            print(f"   Cleaning up persist directory: {persist_dir}")
            removed_count = 0
            for item in persist_dir.iterdir():
                if item.is_dir() and item.name != "__pycache__" and not item.name.startswith('.'):
                    # Remove any UUID-named directories (likely ChromaDB collections)
                    # or directories containing ChromaDB files
                    should_remove = False
                    if len(item.name) == 36 and item.name.count('-') == 4:  # UUID format
                        should_remove = True
                    elif any(f.name.startswith("chroma") or f.name.endswith(".sqlite3") 
                            for f in item.iterdir() if f.is_file()):
                        should_remove = True
                    elif (item / "chroma.sqlite3").exists():
                        should_remove = True
                    
                    if should_remove:
                        print(f"   Removing collection directory: {item.name}")
                        try:
                            shutil.rmtree(item)
                            removed_count += 1
                        except Exception as e:
                            print(f"   ⚠️  Could not remove {item.name}: {str(e)}")
            
            if removed_count > 0:
                print(f"   ✅ Removed {removed_count} collection directories")
        
        print(f"✅ Successfully reset vector store")
        
        # Clear file-based caches
        print(f"\n🗑️  Clearing file-based caches...")
        try:
            # Clear FileCache (downloaded files)
            file_cache = FileCache()
            # Clear all user caches by iterating through metadata
            cleared_count = 0
            user_ids = set()
            for cache_key, metadata in file_cache.metadata.items():
                user_id = metadata.get("user_id")
                if user_id:
                    user_ids.add(user_id)
            
            for user_id in user_ids:
                cleared_count += file_cache.clear_user_cache(user_id)
            
            # Clear metadata file
            if file_cache.metadata_file.exists():
                file_cache.metadata_file.unlink()
                file_cache.metadata = {}
            
            # Also remove any remaining files/directories in cache dir
            cache_dir = Path(settings.CACHE_DIR)
            if cache_dir.exists():
                for item in cache_dir.iterdir():
                    if item.is_file() and item.name != ".metadata.json":
                        item.unlink()
                        cleared_count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        cleared_count += 1
            
            print(f"   ✅ Cleared file cache ({cleared_count} items)")
        except Exception as e:
            print(f"   ⚠️  Could not clear file cache: {str(e)}")
        
        try:
            # Clear Cache (JSON cache files)
            cache = Cache()
            cleared_count = cache.clear()
            print(f"   ✅ Cleared {cleared_count} cache files")
        except Exception as e:
            print(f"   ⚠️  Could not clear cache: {str(e)}")
        
        # Clear knowledge graph files directory
        print(f"\n🗑️  Clearing knowledge graph files...")
        try:
            kg_dir = Path(settings.KNOWLEDGE_GRAPH_DIR)
            if kg_dir.exists():
                removed_count = 0
                for item in kg_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                        removed_count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        removed_count += 1
                if removed_count > 0:
                    print(f"   ✅ Removed {removed_count} knowledge graph files/directories")
                else:
                    print(f"   ℹ️  No knowledge graph files to remove")
            else:
                print(f"   ℹ️  Knowledge graph directory does not exist")
        except Exception as e:
            print(f"   ⚠️  Could not clear knowledge graph files: {str(e)}")
        
        # Note: Cache directory already cleared above with FileCache
        
        # Clear any LangChain SQLite caches
        print(f"\n🗑️  Clearing LangChain caches...")
        try:
            from app.rag.chain_caching import LangChainCacheManager
            cache_manager = LangChainCacheManager()
            cache_manager.clear_all_caches()
            print(f"   ✅ Cleared LangChain in-memory caches")
        except Exception as e:
            print(f"   ⚠️  Could not clear LangChain caches (may not be initialized): {str(e)}")
        
        # Check for LangChain SQLite cache files
        try:
            langchain_cache_paths = [
                Path("./data/langchain_cache.db"),
                Path("./data/.langchain_cache.db"),
                Path(settings.CACHE_DIR) / "langchain_cache.db",
            ]
            for cache_path in langchain_cache_paths:
                if cache_path.exists():
                    cache_path.unlink()
                    print(f"   ✅ Removed LangChain SQLite cache: {cache_path}")
        except Exception as e:
            print(f"   ⚠️  Could not check for LangChain SQLite caches: {str(e)}")
        
        print(f"\n{'='*60}")
        print("✅ ALL DATA WIPED SUCCESSFULLY")
        print(f"{'='*60}\n")
        print("The database, vector store, and all caches are now empty.")
        print("\n📋 Next Steps:")
        print("   1. Restart the backend server to clear in-memory caches")
        print("      (In-memory caches like entity extraction cache will persist until restart)")
        print("   2. Users will need to re-authenticate with Google")
        print("      (OAuth refresh tokens have been deleted)")
        print("   3. Re-index folders through the frontend")
        print("   4. All file types (HTML, DOCX, PNG, etc.) should process correctly")
        print("   5. Everything will be indexed fresh with no cached data\n")
        
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(wipe_all_data())

