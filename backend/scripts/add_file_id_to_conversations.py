#!/usr/bin/env python3
"""Database migration script to add file_id column to conversations table.

This script adds the file_id column to the conversations table to support
file-specific chats.

Usage:
    python scripts/add_file_id_to_conversations.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set SECRET_KEY for database initialization
if not os.getenv("SECRET_KEY"):
    os.environ["SECRET_KEY"] = "test-secret-key-for-migration-only"

from sqlalchemy import text
from app.database.base import DatabaseSessionManager, get_database_url
import structlog

logger = structlog.get_logger(__name__)


async def migrate_database():
    """Add file_id column to conversations table."""
    logger.info("Starting database migration to add file_id column to conversations table")

    # Initialize database manager
    db_url = get_database_url()
    db_manager = DatabaseSessionManager(db_url)
    await db_manager.initialize()

    # Define migrations
    migrations = [
        ("file_id", "ALTER TABLE conversations ADD COLUMN file_id TEXT"),
    ]

    success_count = 0
    failed_count = 0

    # Use raw connection for ALTER TABLE commands
    async with db_manager.engine.begin() as conn:
        for column_name, migration_sql in migrations:
            try:
                await conn.execute(text(migration_sql))
                logger.info(f"✓ Added column: {column_name}")
                success_count += 1
                
                # Create index on file_id for better query performance
                try:
                    await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_conversations_file_id ON conversations(file_id)"))
                    logger.info(f"✓ Created index on: {column_name}")
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"⚠ Could not create index on {column_name} (may already exist): {e}")
                    
            except Exception as e:
                if "duplicate column name" in str(e).lower() or "already exists" in str(e).lower():
                    logger.warning(f"⚠ Column already exists: {column_name}")
                    success_count += 1  # Count as success since column already exists
                else:
                    logger.error(f"✗ Failed to add column: {column_name}", error=str(e))
                    failed_count += 1

    # Close database connections
    await db_manager.close()

    logger.info(
        "Migration completed",
        success=success_count,
        failed=failed_count,
        total=len(migrations)
    )

    if failed_count > 0:
        logger.warning(f"{failed_count} migrations failed - please check errors above")
        return False
    else:
        logger.info("✓ All migrations completed successfully")
        return True


if __name__ == "__main__":
    result = asyncio.run(migrate_database())
    sys.exit(0 if result else 1)
