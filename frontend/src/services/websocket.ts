/**
 * WebSocket service for real-time processing updates.
 * 
 * Handles connection management, reconnection logic, and message routing.
 */

interface WebSocketMessage {
  type: string;
  message?: string;
  progress?: number;
  file_name?: string;
  file_id?: string;
  folder_id?: string;
  files_processed?: number;
  total_files?: number;
  failed_files?: number;
  error?: string;
  timestamp?: string;
  [key: string]: any;
}

type MessageHandler = (message: WebSocketMessage) => void;

interface ConnectionOptions {
  autoReconnect?: boolean;
  maxReconnectAttempts?: number;
  reconnectDelay?: number;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
}

/**
 * WebSocketService
 * 
 * Manages WebSocket connections for real-time processing updates.
 * Includes automatic reconnection, error handling, and message routing.
 */
class WebSocketService {
  private connections: Map<string, WebSocket> = new Map();
  private handlers: Map<string, MessageHandler[]> = new Map();
  private reconnectTimers: Map<string, NodeJS.Timeout> = new Map();
  private reconnectAttempts: Map<string, number> = new Map();
  private connectionOptions: Map<string, ConnectionOptions> = new Map();

  /**
   * Connect to WebSocket for a folder.
   * 
   * @param folderId - Folder ID to monitor
   * @param handler - Message handler callback
   * @param options - Connection options
   */
  connect(
    folderId: string,
    handler: MessageHandler,
    options: ConnectionOptions = {}
  ): void {
    const {
      autoReconnect = true,
      maxReconnectAttempts = 5,
      reconnectDelay = 3000,
      onConnect,
      onDisconnect,
      onError,
    } = options;

    // Store options
    this.connectionOptions.set(folderId, {
      autoReconnect,
      maxReconnectAttempts,
      reconnectDelay,
      onConnect,
      onDisconnect,
      onError,
    });

    // Register handler
    if (!this.handlers.has(folderId)) {
      this.handlers.set(folderId, []);
    }
    this.handlers.get(folderId)!.push(handler);

    // Connect if not already connected
    if (!this.connections.has(folderId)) {
      this._connect(folderId);
    }
  }

  /**
   * Internal connection method.
   */
  private _connect(folderId: string): void {
    const options = this.connectionOptions.get(folderId) || {};
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
    
    // Get auth token
    const token = localStorage.getItem('access_token');
    const url = token
      ? `${wsUrl}/ws/${folderId}?token=${encodeURIComponent(token)}`
      : `${wsUrl}/ws/${folderId}`;

    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        console.log(`WebSocket connected for folder ${folderId}`);
        this.connections.set(folderId, ws);
        this.reconnectAttempts.set(folderId, 0);
        
        // Clear any reconnect timer
        const timer = this.reconnectTimers.get(folderId);
        if (timer) {
          clearTimeout(timer);
          this.reconnectTimers.delete(folderId);
        }

        // Send ping to keep connection alive
        this._startKeepAlive(folderId);

        // Call onConnect callback
        if (options.onConnect) {
          options.onConnect();
        }
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          
          // Handle pong responses
          if (message.type === 'pong') {
            return;
          }

          // Route message to handlers
          const handlers = this.handlers.get(folderId) || [];
          handlers.forEach((h) => {
            try {
              h(message);
            } catch (error) {
              console.error('Error in WebSocket message handler:', error);
            }
          });
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        
        if (options.onError) {
          options.onError(error);
        }
      };

      ws.onclose = (event) => {
        console.log(`WebSocket disconnected for folder ${folderId}`, {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean,
        });

        this.connections.delete(folderId);
        this._stopKeepAlive(folderId);

        // Call onDisconnect callback
        if (options.onDisconnect) {
          options.onDisconnect();
        }

        // Attempt reconnection if enabled
        const opts = this.connectionOptions.get(folderId);
        if (opts?.autoReconnect && event.code !== 1000) {
          // Don't reconnect if clean close
          this._attemptReconnect(folderId);
        }
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      if (options.onError) {
        options.onError(error as Event);
      }
    }
  }

  /**
   * Attempt to reconnect WebSocket.
   */
  private _attemptReconnect(folderId: string): void {
    const options = this.connectionOptions.get(folderId);
    if (!options) return;

    const attempts = this.reconnectAttempts.get(folderId) || 0;
    const maxAttempts = options.maxReconnectAttempts || 5;

    if (attempts >= maxAttempts) {
      console.error(
        `Max reconnection attempts reached for folder ${folderId}`
      );
      return;
    }

    const delay = options.reconnectDelay || 3000;
    const nextAttempt = attempts + 1;

    console.log(
      `Attempting to reconnect WebSocket for folder ${folderId} (attempt ${nextAttempt}/${maxAttempts})`
    );

    this.reconnectAttempts.set(folderId, nextAttempt);

    const timer = setTimeout(() => {
      this._connect(folderId);
    }, delay);

    this.reconnectTimers.set(folderId, timer);
  }

  /**
   * Start keep-alive ping mechanism.
   */
  private _startKeepAlive(folderId: string): void {
    const ws = this.connections.get(folderId);
    if (!ws) return;

    const keepAliveInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify({ type: 'ping' }));
        } catch (error) {
          console.error('Failed to send ping:', error);
          clearInterval(keepAliveInterval);
        }
      } else {
        clearInterval(keepAliveInterval);
      }
    }, 30000); // Ping every 30 seconds

    // Store interval ID for cleanup
    (ws as any).__keepAliveInterval = keepAliveInterval;
  }

  /**
   * Stop keep-alive mechanism.
   */
  private _stopKeepAlive(folderId: string): void {
    const ws = this.connections.get(folderId);
    if (ws && (ws as any).__keepAliveInterval) {
      clearInterval((ws as any).__keepAliveInterval);
    }
  }

  /**
   * Disconnect WebSocket for a folder.
   */
  disconnect(folderId: string): void {
    const ws = this.connections.get(folderId);
    if (ws) {
      this._stopKeepAlive(folderId);
      ws.close(1000, 'Client disconnect'); // Clean close
      this.connections.delete(folderId);
    }

    // Clear reconnect timer
    const timer = this.reconnectTimers.get(folderId);
    if (timer) {
      clearTimeout(timer);
      this.reconnectTimers.delete(folderId);
    }

    // Clear handlers
    this.handlers.delete(folderId);
    this.connectionOptions.delete(folderId);
    this.reconnectAttempts.delete(folderId);

    console.log(`WebSocket disconnected for folder ${folderId}`);
  }

  /**
   * Send message to server (if connection is open).
   */
  send(folderId: string, message: any): boolean {
    const ws = this.connections.get(folderId);
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify(message));
        return true;
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        return false;
      }
    }
    return false;
  }

  /**
   * Check if WebSocket is connected for a folder.
   */
  isConnected(folderId: string): boolean {
    const ws = this.connections.get(folderId);
    return ws !== undefined && ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get connection state for a folder.
   */
  getConnectionState(folderId: string): number {
    const ws = this.connections.get(folderId);
    return ws ? ws.readyState : WebSocket.CLOSED;
  }
}

export const websocketService = new WebSocketService();
