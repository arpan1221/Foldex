#!/usr/bin/env python3
"""Database migration utilities."""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.database.sqlite_manager import SQLiteManager


def migrate_database():
    """Run database migrations."""
    print("Running database migrations...")
    
    db = SQLiteManager()
    # Database schema is created automatically in __init__
    # Add any additional migrations here
    
    print("Database migrations complete!")


if __name__ == "__main__":
    migrate_database()

