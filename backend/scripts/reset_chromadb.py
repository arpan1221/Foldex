"""
Utility script to reset ChromaDB database when schema mismatches occur.

Usage:
    python backend/scripts/reset_chromadb.py [--path PATH]

This will delete all ChromaDB database files and allow ChromaDB to recreate
with the current schema version.
"""

import argparse
import shutil
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.settings import settings


def reset_chromadb(persist_directory: Path) -> None:
    """Reset ChromaDB database by removing all database files.
    
    Args:
        persist_directory: Path to ChromaDB persist directory
    """
    print(f"Resetting ChromaDB database at: {persist_directory}")
    print()
    
    if not persist_directory.exists():
        print(f"Directory does not exist: {persist_directory}")
        return
    
    # Find and remove ChromaDB SQLite database files
    db_files = [
        persist_directory / "chroma.sqlite3",
        persist_directory / "chroma.sqlite3-wal",
        persist_directory / "chroma.sqlite3-shm",
    ]
    
    removed_files = []
    for db_file in db_files:
        if db_file.exists():
            print(f"Removing: {db_file}")
            db_file.unlink()
            removed_files.append(db_file.name)
    
    # Also check for subdirectories that might contain old collections
    removed_dirs = []
    for item in persist_directory.iterdir():
        if item.is_dir() and item.name != "__pycache__":
            # Check if it's a ChromaDB collection directory
            if (item / "chroma.sqlite3").exists() or any(
                f.name.startswith("chroma") for f in item.iterdir() if f.is_file()
            ):
                print(f"Removing collection directory: {item}")
                shutil.rmtree(item)
                removed_dirs.append(item.name)
    
    if removed_files or removed_dirs:
        print()
        print("✅ ChromaDB database reset complete!")
        print(f"   Removed {len(removed_files)} database file(s)")
        print(f"   Removed {len(removed_dirs)} collection directory(ies)")
        print()
        print("⚠️  WARNING: All vector embeddings have been deleted.")
        print("   You will need to re-process your folders to rebuild the index.")
    else:
        print()
        print("ℹ️  No ChromaDB database files found to remove.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reset ChromaDB database to fix schema mismatches"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to ChromaDB persist directory (default: from settings)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt (use with caution!)",
    )
    
    args = parser.parse_args()
    
    # Get persist directory
    if args.path:
        persist_directory = Path(args.path)
    else:
        persist_directory = Path(settings.VECTOR_DB_PATH)
    
    # Confirm before deleting
    if not args.confirm:
        print("=" * 80)
        print("ChromaDB Database Reset Utility")
        print("=" * 80)
        print()
        print(f"This will DELETE all data in: {persist_directory}")
        print()
        print("⚠️  WARNING: This action cannot be undone!")
        print("   All vector embeddings will be lost.")
        print("   You will need to re-process your folders.")
        print()
        response = input("Are you sure you want to continue? (yes/no): ").strip().lower()
        
        if response not in ["yes", "y"]:
            print("Cancelled.")
            return
    
    # Reset database
    reset_chromadb(persist_directory)


if __name__ == "__main__":
    main()

