"""WebSocket endpoints for real-time processing updates with authentication."""

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status
from fastapi.exceptions import WebSocketException
from typing import Dict, Set, Optional
import json
import structlog
from datetime import datetime

from app.core.auth import verify_token
from app.core.exceptions import AuthenticationError

logger = structlog.get_logger(__name__)
router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections with authentication and status tracking."""

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_metadata: Dict[WebSocket, Dict[str, any]] = {}

    async def connect(
        self, websocket: WebSocket, folder_id: str, user_id: Optional[str] = None
    ):
        """Connect a WebSocket for a folder.

        Args:
            websocket: WebSocket connection
            folder_id: Folder ID to subscribe to
            user_id: Optional user ID for tracking
        """
        await websocket.accept()
        
        if folder_id not in self.active_connections:
            self.active_connections[folder_id] = set()
        
        self.active_connections[folder_id].add(websocket)
        self.connection_metadata[websocket] = {
            "folder_id": folder_id,
            "user_id": user_id,
            "connected_at": datetime.utcnow().isoformat(),
        }

        logger.info(
            "WebSocket connected",
            folder_id=folder_id,
            user_id=user_id,
            total_connections=len(self.active_connections[folder_id]),
        )

    def disconnect(self, websocket: WebSocket, folder_id: str):
        """Disconnect a WebSocket.

        Args:
            websocket: WebSocket connection
            folder_id: Folder ID
        """
        if folder_id in self.active_connections:
            self.active_connections[folder_id].discard(websocket)
            if not self.active_connections[folder_id]:
                del self.active_connections[folder_id]

        if websocket in self.connection_metadata:
            metadata = self.connection_metadata.pop(websocket)
            logger.info(
                "WebSocket disconnected",
                folder_id=folder_id,
                user_id=metadata.get("user_id"),
            )

    async def send_message(self, folder_id: str, message: dict):
        """Send message to all connections for a folder.

        Args:
            folder_id: Folder ID
            message: Message to send (will be enriched with timestamp)
        """
        if folder_id not in self.active_connections:
            return

        # Enrich message with timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()

        disconnected = set()
        connection_count = len(self.active_connections[folder_id])

        for connection in self.active_connections[folder_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(
                    "Failed to send WebSocket message",
                    folder_id=folder_id,
                    error=str(e),
                )
                disconnected.add(connection)

        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn, folder_id)

        logger.debug(
            "WebSocket message sent",
            folder_id=folder_id,
            message_type=message.get("type"),
            connections=connection_count,
            disconnected=len(disconnected),
        )

    def get_connection_count(self, folder_id: str) -> int:
        """Get number of active connections for a folder.

        Args:
            folder_id: Folder ID

        Returns:
            Number of active connections
        """
        return len(self.active_connections.get(folder_id, set()))

    def get_all_folder_ids(self) -> Set[str]:
        """Get all folder IDs with active connections.

        Returns:
            Set of folder IDs
        """
        return set(self.active_connections.keys())


manager = ConnectionManager()


async def authenticate_websocket(
    websocket: WebSocket, token: Optional[str] = None
) -> Optional[str]:
    """Authenticate WebSocket connection.

    Args:
        websocket: WebSocket connection
        token: Optional authentication token from query params

    Returns:
        User ID if authenticated, None otherwise

    Raises:
        WebSocketException: If authentication fails
    """
    if not token:
        # Try to get token from query params
        query_params = dict(websocket.query_params)
        token = query_params.get("token")

    if not token:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Authentication token required",
        )

    try:
        payload = verify_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Invalid token payload",
            )
        return user_id
    except AuthenticationError as e:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason=f"Authentication failed: {e.message}",
        )


@router.websocket("/{folder_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    folder_id: str,
    token: Optional[str] = Query(None, description="JWT authentication token"),
):
    """WebSocket endpoint for real-time folder processing updates.

    Args:
        websocket: WebSocket connection
        folder_id: Folder ID to monitor
        token: Optional JWT token for authentication

    Raises:
        WebSocketException: If authentication fails
    """
    user_id = None
    
    try:
        # Authenticate connection
        user_id = await authenticate_websocket(websocket, token)
        
        # Connect to manager
        await manager.connect(websocket, folder_id, user_id)

        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "folder_id": folder_id,
            "message": "Connected to processing updates",
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Create tasks for keepalive and message handling
        async def send_keepalive():
            """Send periodic keepalive pings to prevent connection timeout."""
            try:
                while True:
                    await asyncio.sleep(15)  # Send ping every 15 seconds
                    try:
                        await websocket.send_json({
                            "type": "ping",
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                    except Exception as e:
                        logger.debug(f"Keepalive ping failed: {e}")
                        break
            except asyncio.CancelledError:
                pass

        async def handle_messages():
            """Handle incoming messages from client."""
            try:
                while True:
                    data = await websocket.receive_text()

                    try:
                        message = json.loads(data)
                        message_type = message.get("type")

                        if message_type == "ping":
                            # Respond to client ping
                            await websocket.send_json({
                                "type": "pong",
                                "timestamp": datetime.utcnow().isoformat(),
                            })
                        elif message_type == "get_status":
                            # Send current status (if available)
                            await websocket.send_json({
                                "type": "status",
                                "folder_id": folder_id,
                                "message": "Status request received",
                                "timestamp": datetime.utcnow().isoformat(),
                            })
                        else:
                            # Echo unknown messages
                            await websocket.send_json({
                                "type": "message_received",
                                "original_type": message_type,
                                "timestamp": datetime.utcnow().isoformat(),
                            })
                    except json.JSONDecodeError:
                        # Non-JSON message, just echo
                        await websocket.send_json({
                            "type": "echo",
                            "data": data,
                            "timestamp": datetime.utcnow().isoformat(),
                        })
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.warning(
                    "WebSocket message handling error",
                    folder_id=folder_id,
                    error=str(e),
                )

        # Run both tasks concurrently
        keepalive_task = asyncio.create_task(send_keepalive())
        message_task = asyncio.create_task(handle_messages())

        try:
            # Wait for either task to complete (disconnection or error)
            done, pending = await asyncio.wait(
                {keepalive_task, message_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        except Exception as e:
            logger.error(f"WebSocket task error: {e}")
            keepalive_task.cancel()
            message_task.cancel()

    except WebSocketException:
        # Authentication or policy violation
        raise
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", folder_id=folder_id, user_id=user_id)
    except Exception as e:
        logger.error(
            "WebSocket connection error",
            folder_id=folder_id,
            user_id=user_id,
            error=str(e),
            exc_info=True,
        )
    finally:
        manager.disconnect(websocket, folder_id)
