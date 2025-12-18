"""SQLite database manager."""

from typing import List, Optional, Dict
import sqlite3
import json
import structlog
from datetime import datetime

from app.config.settings import settings
from app.models.documents import DocumentChunk, FileMetadata
from app.core.exceptions import DatabaseError

logger = structlog.get_logger(__name__)


class SQLiteManager:
    """Manages SQLite database operations."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize SQLite manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or settings.DATABASE_PATH
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize database schema."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()

            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    folder_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    mime_type TEXT,
                    size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_id) REFERENCES files(file_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    folder_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_graphs (
                    folder_id TEXT PRIMARY KEY,
                    graph_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            self.conn.commit()
            logger.info("Database initialized", db_path=self.db_path)
        except Exception as e:
            logger.error("Database initialization failed", error=str(e))
            raise DatabaseError(f"Failed to initialize database: {str(e)}")

    async def store_file_metadata(
        self, file_info: Dict, folder_id: str
    ) -> None:
        """Store file metadata.

        Args:
            file_info: File metadata dictionary
            folder_id: Folder ID
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO files 
                (file_id, folder_id, file_name, mime_type, size, modified_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    file_info.get("id"),
                    folder_id,
                    file_info.get("name"),
                    file_info.get("mimeType"),
                    file_info.get("size", 0),
                    datetime.utcnow(),
                ),
            )
            self.conn.commit()
        except Exception as e:
            logger.error("Failed to store file metadata", error=str(e))
            raise DatabaseError(f"Failed to store file metadata: {str(e)}")

    async def store_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Store document chunks.

        Args:
            chunks: List of document chunks
        """
        try:
            cursor = self.conn.cursor()
            for chunk in chunks:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, file_id, content, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.file_id,
                        chunk.content,
                        json.dumps(chunk.metadata),
                    ),
                )
            self.conn.commit()
        except Exception as e:
            logger.error("Failed to store chunks", error=str(e))
            raise DatabaseError(f"Failed to store chunks: {str(e)}")

    async def get_chunks_by_folder(self, folder_id: str) -> List[DocumentChunk]:
        """Get all chunks for a folder.

        Args:
            folder_id: Folder ID

        Returns:
            List of document chunks
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT c.*, f.file_name 
                FROM chunks c
                JOIN files f ON c.file_id = f.file_id
                WHERE f.folder_id = ?
                """,
                (folder_id,),
            )
            rows = cursor.fetchall()
            chunks = []
            for row in rows:
                chunks.append(
                    DocumentChunk(
                        chunk_id=row["chunk_id"],
                        content=row["content"],
                        file_id=row["file_id"],
                        metadata=json.loads(row["metadata"] or "{}"),
                    )
                )
            return chunks
        except Exception as e:
            logger.error("Failed to get chunks", error=str(e))
            raise DatabaseError(f"Failed to get chunks: {str(e)}")

    async def keyword_search(
        self, query: str, folder_id: str, k: int = 10
    ) -> List[DocumentChunk]:
        """Keyword search in chunks.

        Args:
            query: Search query
            folder_id: Folder ID
            k: Number of results

        Returns:
            List of matching chunks
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT c.*, f.file_name
                FROM chunks c
                JOIN files f ON c.file_id = f.file_id
                WHERE f.folder_id = ? AND c.content LIKE ?
                LIMIT ?
                """,
                (folder_id, f"%{query}%", k),
            )
            rows = cursor.fetchall()
            chunks = []
            for row in rows:
                chunks.append(
                    DocumentChunk(
                        chunk_id=row["chunk_id"],
                        content=row["content"],
                        file_id=row["file_id"],
                        metadata=json.loads(row["metadata"] or "{}"),
                    )
                )
            return chunks
        except Exception as e:
            logger.error("Keyword search failed", error=str(e))
            return []

    async def store_message(
        self, conversation_id: str, role: str, content: str, user_id: str
    ) -> None:
        """Store chat message.

        Args:
            conversation_id: Conversation ID
            role: Message role (user/assistant)
            content: Message content
            user_id: User ID
        """
        try:
            import uuid
            cursor = self.conn.cursor()

            # Ensure conversation exists
            cursor.execute(
                """
                INSERT OR IGNORE INTO conversations 
                (conversation_id, user_id, updated_at)
                VALUES (?, ?, ?)
                """,
                (conversation_id, user_id, datetime.utcnow()),
            )

            # Store message
            message_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO messages 
                (message_id, conversation_id, role, content)
                VALUES (?, ?, ?, ?)
                """,
                (message_id, conversation_id, role, content),
            )
            self.conn.commit()
        except Exception as e:
            logger.error("Failed to store message", error=str(e))
            raise DatabaseError(f"Failed to store message: {str(e)}")

    async def store_knowledge_graph(self, folder_id: str, graph) -> None:
        """Store knowledge graph.

        Args:
            folder_id: Folder ID
            graph: NetworkX graph
        """
        try:
            import pickle
            cursor = self.conn.cursor()
            graph_data = pickle.dumps(graph)
            cursor.execute(
                """
                INSERT OR REPLACE INTO knowledge_graphs 
                (folder_id, graph_data)
                VALUES (?, ?)
                """,
                (folder_id, graph_data),
            )
            self.conn.commit()
        except Exception as e:
            logger.error("Failed to store knowledge graph", error=str(e))
            raise DatabaseError(f"Failed to store knowledge graph: {str(e)}")

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

