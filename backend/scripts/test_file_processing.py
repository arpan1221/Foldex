"""Test file processing for different file types in Test Folder."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
if not os.getenv("SECRET_KEY"):
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-purposes-only-123456"

from sqlalchemy import select
from app.database.base import DatabaseSessionManager, get_database_url
from app.models.database import FolderRecord, FileRecord
from app.services.document_processor import DocumentProcessor
from app.services.google_drive import GoogleDriveService
from app.database.sqlite_manager import SQLiteManager
import structlog

logger = structlog.get_logger(__name__)

async def test_file_processing():
    """Test processing each file type in Test Folder."""
    print(f"\n{'='*60}")
    print("TESTING FILE PROCESSING FOR EACH FILE TYPE")
    print("="*60)
    
    # Initialize database
    database_url = get_database_url()
    db_manager = DatabaseSessionManager(database_url=database_url)
    await db_manager.initialize()
    
    try:
        async with db_manager.get_session() as session:
            # Find Test Folder
            stmt = select(FolderRecord).where(FolderRecord.folder_name == "Test Folder")
            result = await session.execute(stmt)
            folder = result.scalar_one_or_none()
            
            if not folder:
                print("❌ Test Folder not found")
                return
            
            folder_id = folder.folder_id
            user_id = folder.user_id
            
            # Get files
            files_stmt = select(FileRecord).where(FileRecord.folder_id == folder_id)
            files_result = await session.execute(files_stmt)
            files = files_result.scalars().all()
            
            print(f"\n📁 Found {len(files)} files in Test Folder\n")
            
            # Test document processor
            doc_processor = DocumentProcessor()
            
            # Get supported file types
            supported = doc_processor.get_supported_file_types()
            print(f"✅ Supported extensions: {len(supported['extensions'])}")
            print(f"✅ Supported MIME types: {len(supported['mime_types'])}")
            
            # Check each file
            for file in files:
                print(f"\n{'='*60}")
                print(f"File: {file.file_name}")
                print(f"   MIME Type: {file.mime_type}")
                print(f"   Size: {file.size:,} bytes ({file.size / (1024*1024):.2f} MB)")
                
                # Check if supported
                is_supported = await doc_processor.is_supported_file_type(
                    file.file_name, file.mime_type
                )
                print(f"   Supported: {'✅' if is_supported else '❌'}")
                
                if not is_supported:
                    print(f"   ⚠️  File type not supported!")
                    continue
                
                # Try to get processor
                processor = await doc_processor._get_processor(file.file_name, file.mime_type)
                if processor:
                    print(f"   Processor: {processor.__class__.__name__}")
                else:
                    print(f"   ❌ No processor found!")
            
            print(f"\n{'='*60}")
            print("RECOMMENDATIONS")
            print("="*60)
            
            # Check CSV
            csv_file = next((f for f in files if f.file_name.endswith('.csv')), None)
            if csv_file:
                print(f"\n📊 CSV File: {csv_file.file_name}")
                print(f"   Size: {csv_file.size / (1024*1024):.2f} MB")
                if csv_file.size > 10_000_000:
                    print(f"   ⚠️  Large CSV file - may need special handling")
                    print(f"   Recommendation: Process CSV with structured parsing")
            
            # Check HTML
            html_file = next((f for f in files if f.file_name.endswith('.html')), None)
            if html_file:
                print(f"\n🌐 HTML File: {html_file.file_name}")
                print(f"   Size: {html_file.size:,} bytes")
                print(f"   Recommendation: Use UnstructuredProcessor for better HTML parsing")
                
    finally:
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(test_file_processing())

