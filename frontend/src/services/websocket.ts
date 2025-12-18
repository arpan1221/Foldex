interface WebSocketMessage {
  type: string;
  message?: string;
  progress?: number;
  file_name?: string;
  folder_id?: string;
  [key: string]: any;
}

type MessageHandler = (message: WebSocketMessage) => void;

class WebSocketService {
  private connections: Map<string, WebSocket> = new Map();
  private handlers: Map<string, MessageHandler[]> = new Map();

  connect(folderId: string, handler: MessageHandler): void {
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
    const ws = new WebSocket(`${wsUrl}/ws/${folderId}`);

    ws.onopen = () => {
      console.log(`WebSocket connected for folder ${folderId}`);
      this.connections.set(folderId, ws);
    };

    ws.onmessage = (event) => {
      const message: WebSocketMessage = JSON.parse(event.data);
      const handlers = this.handlers.get(folderId) || [];
      handlers.forEach((h) => h(message));
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log(`WebSocket disconnected for folder ${folderId}`);
      this.connections.delete(folderId);
      this.handlers.delete(folderId);
    };

    // Register handler
    if (!this.handlers.has(folderId)) {
      this.handlers.set(folderId, []);
    }
    this.handlers.get(folderId)!.push(handler);
  }

  disconnect(folderId: string): void {
    const ws = this.connections.get(folderId);
    if (ws) {
      ws.close();
      this.connections.delete(folderId);
      this.handlers.delete(folderId);
    }
  }
}

export const websocketService = new WebSocketService();

