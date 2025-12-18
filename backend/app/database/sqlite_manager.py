"""SQLite database manager with async session handling."""

from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import uuid

from sqlalchemy import select, update, delete, func, text
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.database.base import Base, get_db_manager, DatabaseSessionManager
from app.models.documents import DocumentChunk, FileMetadata
from app.models.database import (
    FileRecord,
    ChunkRecord,
    ConversationRecord,
    MessageRecord,
    UserRecord,
)
from app.core.exceptions import DatabaseError

logger = structlog.get_logger(__name__)


class SQLiteManager:
    """Manages SQLite database operations with async support.

    This class provides a high-level interface for database operations
    using SQLAlchemy async sessions with proper transaction management.
    """

    def __init__(self, db_manager: Optional[DatabaseSessionManager] = None):
        """Initialize SQLite manager.

        Args:
            db_manager: Optional database session manager (uses global if not provided)
        """
        self.db_manager = db_manager

    def _get_db_manager(self) -> DatabaseSessionManager:
        """Get database session manager.

        Returns:
            Database session manager

        Raises:
            DatabaseError: If manager not available
        """
        if self.db_manager:
            return self.db_manager
        return get_db_manager()

    async def store_file_metadata(
        self, file_info: Dict[str, Any], folder_id: str
    ) -> None:
        """Store file metadata.

        Args:
            file_info: File metadata dictionary from Google Drive
            folder_id: Parent folder ID

        Raises:
            DatabaseError: If storage fails

        Example:
            >>> await manager.store_file_metadata(
            ...     {"id": "file123", "name": "doc.pdf", "mimeType": "application/pdf"},
            ...     "folder456"
            ... )
        """
        try:
            async with self._get_db_manager().get_session() as session:
                file_record = FileRecord(
                    file_id=file_info.get("id", ""),
                    folder_id=folder_id,
                    file_name=file_info.get("name", ""),
                    mime_type=file_info.get("mimeType", ""),
                    size=file_info.get("size", 0),
                    created_at=datetime.utcnow(),
                    modified_at=datetime.utcnow(),
                )

                # Use merge to handle insert or update
                await session.merge(file_record)
                await session.commit()

                logger.info(
                    "File metadata stored",
                    file_id=file_record.file_id,
                    folder_id=folder_id,
                )
        except Exception as e:
            logger.error(
                "Failed to store file metadata",
                file_id=file_info.get("id"),
                error=str(e),
                exc_info=True,
            )
            raise DatabaseError(f"Failed to store file metadata: {str(e)}") from e

    async def store_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Store document chunks in batch.

        Args:
            chunks: List of document chunks to store

        Raises:
            DatabaseError: If storage fails

        Example:
            >>> chunks = [DocumentChunk(...), DocumentChunk(...)]
            >>> await manager.store_chunks(chunks)
        """
        if not chunks:
            logger.warning("Attempted to store empty chunk list")
            return

        try:
            async with self._get_db_manager().get_session() as session:
                chunk_records = []
                for chunk in chunks:
                    chunk_record = ChunkRecord(
                        chunk_id=chunk.chunk_id,
                        file_id=chunk.file_id,
                        content=chunk.content,
                        metadata=chunk.metadata,
                        embedding=None,  # Embeddings stored in vector DB
                        created_at=datetime.utcnow(),
                    )
                    chunk_records.append(chunk_record)

                # Batch insert
                session.add_all(chunk_records)
                await session.commit()

                logger.info(
                    "Chunks stored",
                    count=len(chunks),
                    file_ids=list(set(c.file_id for c in chunks)),
                )
        except Exception as e:
            logger.error(
                "Failed to store chunks",
                chunk_count=len(chunks),
                error=str(e),
                exc_info=True,
            )
            raise DatabaseError(f"Failed to store chunks: {str(e)}") from e

    async def get_chunks_by_folder(
        self, folder_id: str, limit: Optional[int] = None
    ) -> List[DocumentChunk]:
        """Get all chunks for a folder.

        Args:
            folder_id: Folder ID
            limit: Optional limit on number of chunks

        Returns:
            List of document chunks

        Raises:
            DatabaseError: If retrieval fails
        """
        try:
            async with self._get_db_manager().get_session() as session:
                query = (
                    select(ChunkRecord, FileRecord)
                    .join(FileRecord, ChunkRecord.file_id == FileRecord.file_id)
                    .where(FileRecord.folder_id == folder_id)
                )

                if limit:
                    query = query.limit(limit)

                result = await session.execute(query)
                rows = result.all()

                chunks = []
                for chunk_record, file_record in rows:
                    metadata = chunk_record.metadata.copy()
                    metadata["file_name"] = file_record.file_name

                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_record.chunk_id,
                            content=chunk_record.content,
                            file_id=chunk_record.file_id,
                            metadata=metadata,
                        )
                    )

                logger.info(
                    "Chunks retrieved",
                    folder_id=folder_id,
                    count=len(chunks),
                )
                return chunks
        except Exception as e:
            logger.error(
                "Failed to get chunks",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            raise DatabaseError(f"Failed to get chunks: {str(e)}") from e

    async def keyword_search(
        self, query: str, folder_id: str, k: int = 10
    ) -> List[DocumentChunk]:
        """Perform keyword search in chunks.

        Args:
            query: Search query string
            folder_id: Folder ID to search within
            k: Maximum number of results

        Returns:
            List of matching document chunks

        Raises:
            DatabaseError: If search fails
        """
        try:
            async with self._get_db_manager().get_session() as session:
                # Use LIKE for simple keyword matching
                # For production, consider full-text search (FTS5)
                search_pattern = f"%{query}%"

                query_stmt = (
                    select(ChunkRecord, FileRecord)
                    .join(FileRecord, ChunkRecord.file_id == FileRecord.file_id)
                    .where(
                        FileRecord.folder_id == folder_id,
                        ChunkRecord.content.like(search_pattern),
                    )
                    .limit(k)
                )

                result = await session.execute(query_stmt)
                rows = result.all()

                chunks = []
                for chunk_record, file_record in rows:
                    metadata = chunk_record.metadata.copy()
                    metadata["file_name"] = file_record.file_name

                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_record.chunk_id,
                            content=chunk_record.content,
                            file_id=chunk_record.file_id,
                            metadata=metadata,
                        )
                    )

                logger.info(
                    "Keyword search completed",
                    query=query[:50],  # Truncate for logging
                    folder_id=folder_id,
                    results_count=len(chunks),
                )
                return chunks
        except Exception as e:
            logger.error(
                "Keyword search failed",
                query=query[:50],
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            return []  # Return empty list on error rather than raising

    async def store_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        user_id: str,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Store chat message.

        Args:
            conversation_id: Conversation ID
            role: Message role (user/assistant)
            content: Message content
            user_id: User ID
            citations: Optional list of citations

        Returns:
            Message ID

        Raises:
            DatabaseError: If storage fails
        """
        try:
            async with self._get_db_manager().get_session() as session:
                # Ensure conversation exists
                conversation = await session.get(
                    ConversationRecord, conversation_id
                )
                if not conversation:
                    conversation = ConversationRecord(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        folder_id=None,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    )
                    session.add(conversation)

                # Create message
                message_id = str(uuid.uuid4())
                message = MessageRecord(
                    message_id=message_id,
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    citations=json.dumps(citations) if citations else None,
                    timestamp=datetime.utcnow(),
                )

                session.add(message)

                # Update conversation timestamp
                conversation.updated_at = datetime.utcnow()

                await session.commit()

                logger.info(
                    "Message stored",
                    message_id=message_id,
                    conversation_id=conversation_id,
                    role=role,
                )
                return message_id
        except Exception as e:
            logger.error(
                "Failed to store message",
                conversation_id=conversation_id,
                error=str(e),
                exc_info=True,
            )
            raise DatabaseError(f"Failed to store message: {str(e)}") from e

    async def get_conversation_messages(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get messages for a conversation.

        Args:
            conversation_id: Conversation ID
            limit: Optional limit on number of messages

        Returns:
            List of message dictionaries

        Raises:
            DatabaseError: If retrieval fails
        """
        try:
            async with self._get_db_manager().get_session() as session:
                query = select(MessageRecord).where(
                    MessageRecord.conversation_id == conversation_id
                ).order_by(MessageRecord.timestamp)

                if limit:
                    query = query.limit(limit)

                result = await session.execute(query)
                messages = result.scalars().all()

                return [
                    {
                        "message_id": msg.message_id,
                        "role": msg.role,
                        "content": msg.content,
                        "citations": (
                            json.loads(msg.citations) if msg.citations else None
                        ),
                        "timestamp": msg.timestamp,
                    }
                    for msg in messages
                ]
        except Exception as e:
            logger.error(
                "Failed to get conversation messages",
                conversation_id=conversation_id,
                error=str(e),
                exc_info=True,
            )
            raise DatabaseError(
                f"Failed to get conversation messages: {str(e)}"
            ) from e

    async def store_knowledge_graph(
        self, folder_id: str, graph_data: bytes
    ) -> None:
        """Store knowledge graph data.

        Args:
            folder_id: Folder ID
            graph_data: Serialized graph data (pickle bytes)

        Raises:
            DatabaseError: If storage fails
        """
        try:
            async with self._get_db_manager().get_session() as session:
                # Use raw SQL for BLOB storage
                await session.execute(
                    text("""
                        INSERT OR REPLACE INTO knowledge_graphs 
                        (folder_id, graph_data, created_at)
                        VALUES (:folder_id, :graph_data, :created_at)
                    """),
                    {
                        "folder_id": folder_id,
                        "graph_data": graph_data,
                        "created_at": datetime.utcnow(),
                    },
                )
                await session.commit()

                logger.info("Knowledge graph stored", folder_id=folder_id)
        except Exception as e:
            logger.error(
                "Failed to store knowledge graph",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            raise DatabaseError(
                f"Failed to store knowledge graph: {str(e)}"
            ) from e

    async def get_knowledge_graph(self, folder_id: str) -> Optional[bytes]:
        """Get knowledge graph data.

        Args:
            folder_id: Folder ID

        Returns:
            Serialized graph data or None if not found

        Raises:
            DatabaseError: If retrieval fails
        """
        try:
            async with self._get_db_manager().get_session() as session:
                result = await session.execute(
                    text(
                        "SELECT graph_data FROM knowledge_graphs WHERE folder_id = :folder_id"
                    ),
                    {"folder_id": folder_id},
                )
                row = result.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.error(
                "Failed to get knowledge graph",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            raise DatabaseError(
                f"Failed to get knowledge graph: {str(e)}"
            ) from e

    async def get_or_create_user(
        self,
        google_id: str,
        email: str,
        name: Optional[str] = None,
        picture: Optional[str] = None,
        verified_email: bool = False,
    ) -> Dict[str, Any]:
        """Get or create user record.

        Args:
            google_id: Google user ID
            email: User email
            name: User name
            picture: User profile picture URL
            verified_email: Email verification status

        Returns:
            User dictionary

        Raises:
            DatabaseError: If operation fails
        """
        try:
            async with self._get_db_manager().get_session() as session:
                # Try to find existing user by google_id or email
                stmt = select(UserRecord).where(
                    (UserRecord.google_id == google_id) | (UserRecord.email == email)
                )
                result = await session.execute(stmt)
                user = result.scalar_one_or_none()

                if user:
                    # Update existing user
                    user.name = name or user.name
                    user.picture = picture or user.picture
                    user.verified_email = verified_email or user.verified_email
                    user.updated_at = datetime.utcnow()
                    await session.commit()
                    await session.refresh(user)

                    logger.info("User updated", user_id=user.user_id, email=email)
                else:
                    # Create new user
                    import uuid

                    user_id = str(uuid.uuid4())
                    user = UserRecord(
                        user_id=user_id,
                        google_id=google_id,
                        email=email,
                        name=name,
                        picture=picture,
                        verified_email=verified_email,
                    )
                    session.add(user)
                    await session.commit()
                    await session.refresh(user)

                    logger.info("User created", user_id=user_id, email=email)

                return {
                    "user_id": user.user_id,
                    "email": user.email,
                    "name": user.name,
                    "google_id": user.google_id,
                    "picture": user.picture,
                    "verified_email": user.verified_email,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "updated_at": user.updated_at.isoformat() if user.updated_at else None,
                }

        except Exception as e:
            logger.error(
                "Failed to get or create user",
                email=email,
                error=str(e),
                exc_info=True,
            )
            raise DatabaseError(f"Failed to get or create user: {str(e)}") from e

    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by user ID.

        Args:
            user_id: User identifier

        Returns:
            User dictionary or None if not found

        Raises:
            DatabaseError: If operation fails
        """
        try:
            async with self._get_db_manager().get_session() as session:
                stmt = select(UserRecord).where(UserRecord.user_id == user_id)
                result = await session.execute(stmt)
                user = result.scalar_one_or_none()

                if not user:
                    return None

                return {
                    "user_id": user.user_id,
                    "email": user.email,
                    "name": user.name,
                    "google_id": user.google_id,
                    "picture": user.picture,
                    "verified_email": user.verified_email,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "updated_at": user.updated_at.isoformat() if user.updated_at else None,
                }

        except Exception as e:
            logger.error("Failed to get user by ID", user_id=user_id, error=str(e))
            raise DatabaseError(f"Failed to get user: {str(e)}") from e

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email.

        Args:
            email: User email

        Returns:
            User dictionary or None if not found

        Raises:
            DatabaseError: If operation fails
        """
        try:
            async with self._get_db_manager().get_session() as session:
                stmt = select(UserRecord).where(UserRecord.email == email)
                result = await session.execute(stmt)
                user = result.scalar_one_or_none()

                if not user:
                    return None

                return {
                    "user_id": user.user_id,
                    "email": user.email,
                    "name": user.name,
                    "google_id": user.google_id,
                    "picture": user.picture,
                    "verified_email": user.verified_email,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "updated_at": user.updated_at.isoformat() if user.updated_at else None,
                }

        except Exception as e:
            logger.error("Failed to get user by email", email=email, error=str(e))
            raise DatabaseError(f"Failed to get user: {str(e)}") from e

    async def get_files_by_folder(self, folder_id: str) -> List[Dict[str, Any]]:
        """Get all files in a folder.

        Args:
            folder_id: Folder ID

        Returns:
            List of file dictionaries

        Raises:
            DatabaseError: If query fails
        """
        try:
            async with self._get_db_manager().get_session() as session:
                stmt = select(FileRecord).where(FileRecord.folder_id == folder_id)
                result = await session.execute(stmt)
                files = result.scalars().all()

                return [
                    {
                        "file_id": file.file_id,
                        "file_name": file.file_name,
                        "mime_type": file.mime_type,
                        "size": file.size,
                        "folder_id": file.folder_id,
                        "created_at": file.created_at.isoformat() if file.created_at else None,
                        "modified_at": file.modified_at.isoformat() if file.modified_at else None,
                    }
                    for file in files
                ]

        except Exception as e:
            logger.error(
                "Failed to get files by folder",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            raise DatabaseError(f"Failed to get files: {str(e)}") from e

    async def get_file_count_by_folder(self, folder_id: str) -> int:
        """Get file count for a folder.

        Args:
            folder_id: Folder ID

        Returns:
            Number of files in folder

        Raises:
            DatabaseError: If query fails
        """
        try:
            async with self._get_db_manager().get_session() as session:
                result = await session.execute(
                    select(func.count(FileRecord.file_id)).where(
                        FileRecord.folder_id == folder_id
                    )
                )
                count = result.scalar() or 0
                return count
        except Exception as e:
            logger.error(
                "Failed to get file count",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            raise DatabaseError(f"Failed to get file count: {str(e)}") from e

    async def close(self) -> None:
        """Close database connections.

        Note: This is a no-op when using dependency injection.
        Connections are managed by the session manager.
        """
        pass
