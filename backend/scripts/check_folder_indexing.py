"""Script to check if all documents in a folder are indexed."""

import sqlite3
import sys
from pathlib import Path


def check_folder_indexing(folder_name: str, db_path: str = "./data/foldex.db"):
    """Check if all documents in a folder are indexed.
    
    Args:
        folder_name: Name of the folder to check
        db_path: Path to the SQLite database
    """
    # Resolve database path relative to project root
    if not Path(db_path).is_absolute():
        # Assume script is run from project root or backend directory
        project_root = Path(__file__).parent.parent.parent
        db_path = project_root / db_path
    
    if not Path(db_path).exists():
        print(f"❌ Database not found at: {db_path}")
        return
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row  # Enable column access by name
    cursor = conn.cursor()
    
    try:
        # Search for folder by name
        cursor.execute("SELECT * FROM folders WHERE folder_name = ?", (folder_name,))
        folders = cursor.fetchall()
        
        if not folders:
            print(f"❌ No folder found with name '{folder_name}'")
            return
        
        if len(folders) > 1:
            print(f"⚠️  Found {len(folders)} folders with name '{folder_name}':")
            for folder in folders:
                print(f"   - Folder ID: {folder['folder_id']}, User ID: {folder['user_id']}, Status: {folder['status']}")
            print(f"\nChecking all folders...\n")
        
        for folder in folders:
            folder_id = folder['folder_id']
            print(f"📁 Folder: {folder['folder_name']}")
            print(f"   Folder ID: {folder_id}")
            print(f"   User ID: {folder['user_id']}")
            print(f"   Status: {folder['status']}")
            print(f"   File Count (metadata): {folder['file_count']}")
            print()
            
            # Get all files in this folder
            cursor.execute("SELECT * FROM files WHERE folder_id = ?", (folder_id,))
            files = cursor.fetchall()
            
            if not files:
                print(f"   ❌ No files found in folder")
                print()
                continue
            
            print(f"   📄 Found {len(files)} files:")
            print()
            
            indexed_count = 0
            not_indexed = []
            
            for file in files:
                file_id = file['file_id']
                # Check if file has chunks
                cursor.execute(
                    "SELECT COUNT(*) as count FROM chunks WHERE file_id = ?",
                    (file_id,)
                )
                chunk_count = cursor.fetchone()['count']
                
                if chunk_count > 0:
                    indexed_count += 1
                    status = "✅"
                else:
                    status = "❌"
                    not_indexed.append(file)
                
                print(f"   {status} {file['file_name']}")
                print(f"      File ID: {file_id}")
                print(f"      MIME Type: {file['mime_type']}")
                print(f"      Size: {file['size']:,} bytes")
                print(f"      Chunks: {chunk_count}")
                print()
            
            # Summary
            print(f"   📊 Summary:")
            print(f"      Total files: {len(files)}")
            print(f"      Indexed: {indexed_count} ({indexed_count/len(files)*100:.1f}%)")
            print(f"      Not indexed: {len(not_indexed)} ({len(not_indexed)/len(files)*100:.1f}%)")
            print()
            
            if not_indexed:
                print(f"   ⚠️  Files not indexed:")
                for file in not_indexed:
                    print(f"      - {file['file_name']} ({file['mime_type']})")
                print()
            else:
                print(f"   ✅ All files are indexed!")
                print()
    
    finally:
        conn.close()


if __name__ == "__main__":
    folder_name = "Test Folder"
    db_path = "./data/foldex.db"
    
    if len(sys.argv) > 1:
        folder_name = sys.argv[1]
    if len(sys.argv) > 2:
        db_path = sys.argv[2]
    
    print(f"🔍 Checking indexing status for folder: '{folder_name}'")
    print("=" * 60)
    print()
    
    check_folder_indexing(folder_name, db_path)
