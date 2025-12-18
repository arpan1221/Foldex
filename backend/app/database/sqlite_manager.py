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
    FolderRecord,
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
                for chunk in chunks:
                    # Check if chunk already exists
                    stmt = select(ChunkRecord).where(ChunkRecord.chunk_id == chunk.chunk_id)
                    result = await session.execute(stmt)
                    existing = result.scalar_one_or_none()
                    
                    if existing:
                        # Update existing chunk
                        existing.content = chunk.content
                        existing.metadata = chunk.metadata
                        existing.embedding = None
                        logger.debug("Updating existing chunk", chunk_id=chunk.chunk_id)
                    else:
                        # Create new chunk
                        chunk_record = ChunkRecord(
                            chunk_id=chunk.chunk_id,
                            file_id=chunk.file_id,
                            content=chunk.content,
                            metadata=chunk.metadata,
                            embedding=None,  # Embeddings stored in vector DB
                            created_at=datetime.utcnow(),
                        )
                        session.add(chunk_record)

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
        folder_id: Optional[str] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Store chat message.

        Args:
            conversation_id: Conversation ID
            role: Message role (user/assistant)
            content: Message content
            user_id: User ID
            folder_id: Optional folder ID
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
                    # Create new conversation if it doesn't exist
                    conversation = ConversationRecord(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        folder_id=folder_id,
                        title=content[:50] + ("..." if len(content) > 50 else "") if role == "user" else "New Chat",
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    )
                    session.add(conversation)
                elif conversation.title == "New Chat" and role == "user":
                    # Update title if it's still the default and this is the first user message
                    conversation.title = content[:50] + ("..." if len(content) > 50 else "")

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

                # Update conversation timestamp and folder_id if provided
                conversation.updated_at = datetime.utcnow()
                if folder_id and not conversation.folder_id:
                    conversation.folder_id = folder_id

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

    async def get_folder_conversations(
        self, user_id: str, folder_id: str
    ) -> List[Dict[str, Any]]:
        """Get all conversations for a specific folder and user.

        Args:
            user_id: User ID
            folder_id: Folder ID

        Returns:
            List of conversation dictionaries
        """
        try:
            async with self._get_db_manager().get_session() as session:
                stmt = (
                    select(ConversationRecord)
                    .where(
                        ConversationRecord.user_id == user_id,
                        ConversationRecord.folder_id == folder_id,
                    )
                    .order_by(ConversationRecord.updated_at.desc())
                )
                result = await session.execute(stmt)
                conversations = result.scalars().all()

                return [
                    {
                        "conversation_id": conv.conversation_id,
                        "folder_id": conv.folder_id,
                        "title": conv.title or "Untitled Chat",
                        "created_at": conv.created_at,
                        "updated_at": conv.updated_at,
                    }
                    for conv in conversations
                ]
        except Exception as e:
            logger.error(
                "Failed to get folder conversations",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            raise DatabaseError(f"Failed to get folder conversations: {str(e)}") from e

    async def create_conversation(
        self, user_id: str, folder_id: str, title: str = "New Chat"
    ) -> str:
        """Create a new conversation session.

        Args:
            user_id: User ID
            folder_id: Folder ID
            title: Initial title

        Returns:
            New conversation ID
        """
        try:
            async with self._get_db_manager().get_session() as session:
                conversation_id = str(uuid.uuid4())
                conversation = ConversationRecord(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    folder_id=folder_id,
                    title=title,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                session.add(conversation)
                await session.commit()
                return conversation_id
        except Exception as e:
            logger.error("Failed to create conversation", error=str(e))
            raise DatabaseError(f"Failed to create conversation: {str(e)}") from e

    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation.

        Args:
            conversation_id: Conversation ID
            user_id: User ID (for verification)

        Returns:
            True if deleted
        """
        try:
            async with self._get_db_manager().get_session() as session:
                stmt = select(ConversationRecord).where(
                    ConversationRecord.conversation_id == conversation_id,
                    ConversationRecord.user_id == user_id,
                )
                result = await session.execute(stmt)
                conversation = result.scalar_one_or_none()

                if not conversation:
                    return False

                await session.delete(conversation)
                await session.commit()
                return True
        except Exception as e:
            logger.error("Failed to delete conversation", error=str(e))
            raise DatabaseError(f"Failed to delete conversation: {str(e)}") from e

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
        google_access_token: Optional[str] = None,
        google_refresh_token: Optional[str] = None,
        google_token_expiry: Optional[datetime] = None,
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """Get or create user record.

        Args:
            google_id: Google user ID
            email: User email
            name: User name
            picture: User profile picture URL
            verified_email: Email verification status
            session: Optional database session (creates new if not provided)

        Returns:
            User dictionary

        Raises:
            DatabaseError: If operation fails
        """
        try:
            # Use provided session or create a new one
            if session is not None:
                # Use provided session (don't commit, let caller handle it)
                return await self._get_or_create_user_with_session(
                    session, google_id, email, name, picture, verified_email,
                    google_access_token, google_refresh_token, google_token_expiry
                )
            
            # Create new session
            async with self._get_db_manager().get_session() as session:
                result = await self._get_or_create_user_with_session(
                    session, google_id, email, name, picture, verified_email,
                    google_access_token, google_refresh_token, google_token_expiry
                )
                await session.commit()
                return result

        except Exception as e:
            logger.error(
                "Failed to get or create user",
                email=email,
                error=str(e),
                exc_info=True,
            )
            raise DatabaseError(f"Failed to get or create user: {str(e)}") from e

    async def _get_or_create_user_with_session(
        self,
        session: AsyncSession,
        google_id: str,
        email: str,
        name: Optional[str] = None,
        picture: Optional[str] = None,
        verified_email: bool = False,
        google_access_token: Optional[str] = None,
        google_refresh_token: Optional[str] = None,
        google_token_expiry: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Internal method to get or create user with provided session.
        
        Args:
            session: Database session
            google_id: Google user ID
            email: User email
            name: User name
            picture: User profile picture URL
            verified_email: Email verification status
            
        Returns:
            User dictionary
        """
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
            if google_access_token:
                user.google_access_token = google_access_token
            if google_refresh_token:
                user.google_refresh_token = google_refresh_token
            if google_token_expiry:
                user.google_token_expiry = google_token_expiry
            user.updated_at = datetime.utcnow()
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
                google_access_token=google_access_token,
                google_refresh_token=google_refresh_token,
                google_token_expiry=google_token_expiry,
            )
            session.add(user)
            await session.flush()  # Flush to get the ID without committing
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
                    "google_access_token": user.google_access_token,
                    "google_refresh_token": user.google_refresh_token,
                    "google_token_expiry": user.google_token_expiry.isoformat() if user.google_token_expiry else None,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "updated_at": user.updated_at.isoformat() if user.updated_at else None,
                }

        except Exception as e:
            logger.error("Failed to get user by ID", user_id=user_id, error=str(e))
            raise DatabaseError(f"Failed to get user: {str(e)}") from e

    async def update_user(
        self,
        user_id: str,
        google_access_token: Optional[str] = None,
        google_refresh_token: Optional[str] = None,
        google_token_expiry: Optional[datetime] = None,
    ) -> None:
        """Update user's Google OAuth tokens.

        Args:
            user_id: User identifier
            google_access_token: Optional new access token
            google_refresh_token: Optional new refresh token
            google_token_expiry: Optional new token expiry

        Raises:
            DatabaseError: If update fails
        """
        try:
            async with self._get_db_manager().get_session() as session:
                stmt = select(UserRecord).where(UserRecord.user_id == user_id)
                result = await session.execute(stmt)
                user = result.scalar_one_or_none()

                if not user:
                    raise DatabaseError(f"User not found: {user_id}")

                if google_access_token is not None:
                    user.google_access_token = google_access_token
                if google_refresh_token is not None:
                    user.google_refresh_token = google_refresh_token
                if google_token_expiry is not None:
                    user.google_token_expiry = google_token_expiry
                
                user.updated_at = datetime.utcnow()
                await session.commit()

                logger.info("User tokens updated", user_id=user_id)
        except Exception as e:
            logger.error("Failed to update user", user_id=user_id, error=str(e), exc_info=True)
            raise DatabaseError(f"Failed to update user: {str(e)}") from e

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

    async def store_folder(
        self,
        folder_id: str,
        user_id: str,
        folder_name: str,
        folder_url: Optional[str] = None,
        file_count: int = 0,
        status: str = "processing",
        parent_folder_id: Optional[str] = None,
        root_folder_id: Optional[str] = None,
    ) -> None:
        """Store or update folder record.

        Args:
            folder_id: Folder ID
            user_id: User ID
            folder_name: Folder name
            folder_url: Optional folder URL
            file_count: Number of files in folder
            status: Processing status

        Raises:
            DatabaseError: If storage fails
        """
        try:
            async with self._get_db_manager().get_session() as session:
                # Check if folder exists
                stmt = select(FolderRecord).where(FolderRecord.folder_id == folder_id)
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    # Update existing folder
                    existing.folder_name = folder_name
                    if folder_url:
                        existing.folder_url = folder_url
                    existing.file_count = file_count
                    existing.status = status
                    if parent_folder_id is not None:
                        existing.parent_folder_id = parent_folder_id
                    if root_folder_id is not None:
                        existing.root_folder_id = root_folder_id
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new folder
                    folder = FolderRecord(
                        folder_id=folder_id,
                        user_id=user_id,
                        folder_name=folder_name,
                        folder_url=folder_url,
                        file_count=file_count,
                        status=status,
                        parent_folder_id=parent_folder_id,
                        root_folder_id=root_folder_id or folder_id,
                    )
                    session.add(folder)

                await session.commit()
                logger.info("Folder stored", folder_id=folder_id, user_id=user_id, folder_name=folder_name)
        except Exception as e:
            logger.error("Failed to store folder", folder_id=folder_id, error=str(e), exc_info=True)
            raise DatabaseError(f"Failed to store folder: {str(e)}") from e

    async def get_user_folders(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all root folders for a user (folders without parents).

        Args:
            user_id: User ID

        Returns:
            List of root folder dictionaries

        Raises:
            DatabaseError: If query fails
        """
        try:
            async with self._get_db_manager().get_session() as session:
                # Only get root folders (where parent_folder_id is NULL AND folder_id == root_folder_id)
                stmt = (
                    select(FolderRecord)
                    .where(
                        FolderRecord.user_id == user_id,
                        FolderRecord.parent_folder_id.is_(None),  # Must be NULL
                        FolderRecord.folder_id == FolderRecord.root_folder_id  # Must be root
                    )
                    .order_by(FolderRecord.updated_at.desc())
                )
                result = await session.execute(stmt)
                folders = result.scalars().all()

                return [
                    {
                        "folder_id": folder.folder_id,
                        "folder_name": folder.folder_name,
                        "folder_url": folder.folder_url,
                        "file_count": folder.file_count,
                        "status": folder.status,
                        "parent_folder_id": folder.parent_folder_id,
                        "root_folder_id": folder.root_folder_id,
                        "created_at": folder.created_at.isoformat() if folder.created_at else None,
                        "updated_at": folder.updated_at.isoformat() if folder.updated_at else None,
                    }
                    for folder in folders
                ]
        except Exception as e:
            logger.error("Failed to get user folders", user_id=user_id, error=str(e), exc_info=True)
            raise DatabaseError(f"Failed to get user folders: {str(e)}") from e

    async def update_folder_status(
        self,
        folder_id: str,
        status: str,
    ) -> None:
        """Update folder processing status.

        Args:
            folder_id: Folder ID
            status: New processing status

        Raises:
            DatabaseError: If update fails
        """
        try:
            async with self._get_db_manager().get_session() as session:
                stmt = select(FolderRecord).where(FolderRecord.folder_id == folder_id)
                result = await session.execute(stmt)
                folder = result.scalar_one_or_none()

                if folder:
                    folder.status = status
                    folder.updated_at = datetime.utcnow()
                    await session.commit()
                    logger.debug("Folder status updated", folder_id=folder_id, status=status)
                else:
                    logger.warning("Folder not found for status update", folder_id=folder_id)

        except Exception as e:
            logger.error("Failed to update folder status", folder_id=folder_id, error=str(e), exc_info=True)
            raise DatabaseError(f"Failed to update folder status: {str(e)}") from e

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

    async def delete_folder(self, folder_id: str, user_id: str) -> bool:
        """Delete a folder and all associated data.

        Args:
            folder_id: Folder ID
            user_id: User ID (for verification)

        Returns:
            True if deleted, False if not found

        Raises:
            DatabaseError: If deletion fails
        """
        try:
            async with self._get_db_manager().get_session() as session:
                # Verify folder belongs to user
                stmt = select(FolderRecord).where(
                    FolderRecord.folder_id == folder_id,
                    FolderRecord.user_id == user_id
                )
                result = await session.execute(stmt)
                folder = result.scalar_one_or_none()

                if not folder:
                    logger.warning("Folder not found or access denied", folder_id=folder_id, user_id=user_id)
                    return False

                # Get all file IDs for this folder (for cache cleanup)
                file_stmt = select(FileRecord.file_id).where(FileRecord.folder_id == folder_id)
                file_result = await session.execute(file_stmt)
                file_ids = [row[0] for row in file_result.all()]

                # Delete folder record (cascade will delete files and chunks due to relationships)
                await session.delete(folder)
                await session.commit()

                logger.info(
                    "Folder deleted",
                    folder_id=folder_id,
                    user_id=user_id,
                    file_count=len(file_ids)
                )
                return True
        except Exception as e:
            logger.error("Failed to delete folder", folder_id=folder_id, error=str(e), exc_info=True)
            raise DatabaseError(f"Failed to delete folder: {str(e)}") from e

    async def close(self) -> None:
        """Close database connections.

        Note: This is a no-op when using dependency injection.
        Connections are managed by the session manager.
        """
        pass
