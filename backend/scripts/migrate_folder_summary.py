#!/usr/bin/env python3
"""Database migration script to add folder summary fields.

This script adds new columns to the folders table to support
the folder knowledge base / post-ingestion summarization feature.

Usage:
    python scripts/migrate_folder_summary.py
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
    """Add new columns to folders table for folder summarization."""
    logger.info("Starting database migration for folder summary fields")

    # Initialize database manager
    db_url = get_database_url()
    db_manager = DatabaseSessionManager(db_url)
    await db_manager.initialize()

    # Define migrations
    migrations = [
        ("summary", "ALTER TABLE folders ADD COLUMN summary TEXT"),
        ("learning_status", "ALTER TABLE folders ADD COLUMN learning_status TEXT DEFAULT 'learning_pending'"),
        ("insights", "ALTER TABLE folders ADD COLUMN insights JSON"),
        ("file_type_distribution", "ALTER TABLE folders ADD COLUMN file_type_distribution JSON"),
        ("entity_summary", "ALTER TABLE folders ADD COLUMN entity_summary JSON"),
        ("relationship_summary", "ALTER TABLE folders ADD COLUMN relationship_summary JSON"),
        ("capabilities", "ALTER TABLE folders ADD COLUMN capabilities JSON"),
        ("graph_statistics", "ALTER TABLE folders ADD COLUMN graph_statistics JSON"),
        ("learning_completed_at", "ALTER TABLE folders ADD COLUMN learning_completed_at DATETIME"),
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
            except Exception as e:
                if "duplicate column name" in str(e).lower():
                    logger.warning(f"⚠ Column already exists: {column_name}")
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
