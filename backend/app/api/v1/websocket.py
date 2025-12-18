"""WebSocket endpoints for real-time updates."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json

router = APIRouter()

# Store active connections
active_connections: Dict[str, Set[WebSocket]] = {}


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, folder_id: str):
        """Connect a WebSocket for a folder.

        Args:
            websocket: WebSocket connection
            folder_id: Folder ID to subscribe to
        """
        await websocket.accept()
        if folder_id not in self.active_connections:
            self.active_connections[folder_id] = set()
        self.active_connections[folder_id].add(websocket)

    def disconnect(self, websocket: WebSocket, folder_id: str):
        """Disconnect a WebSocket.

        Args:
            websocket: WebSocket connection
            folder_id: Folder ID
        """
        if folder_id in self.active_connections:
            self.active_connections[folder_id].discard(websocket)

    async def send_message(self, folder_id: str, message: dict):
        """Send message to all connections for a folder.

        Args:
            folder_id: Folder ID
            message: Message to send
        """
        if folder_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[folder_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.add(connection)

            # Remove disconnected connections
            for conn in disconnected:
                self.active_connections[folder_id].discard(conn)


manager = ConnectionManager()


@router.websocket("/{folder_id}")
async def websocket_endpoint(websocket: WebSocket, folder_id: str):
    """WebSocket endpoint for real-time folder processing updates.

    Args:
        websocket: WebSocket connection
        folder_id: Folder ID to monitor
    """
    await manager.connect(websocket, folder_id)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back or process message
            await websocket.send_json({"type": "pong", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket, folder_id)

