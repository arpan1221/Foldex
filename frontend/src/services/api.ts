/**
 * Centralized API service with comprehensive error handling and type safety.
 * 
 * Provides typed methods for all backend API endpoints with proper
 * error handling, request/response transformation, and retry logic.
 */

import axios, { AxiosError, AxiosRequestConfig, AxiosResponse } from 'axios';
import {
  TokenResponse,
  RefreshTokenResponse,
  UserResponse,
  ProcessFolderRequest,
  ProcessFolderResponse,
  FolderStatusResponse,
  FolderMetadata,
  FileMetadata,
  TreeNode,
  ChatRequest,
  ChatResponse,
  KnowledgeGraphResponse,
  HealthCheck,
  APIError,
  Citation,
  ChatMessage,
  Conversation,
  FolderSummary,
} from './types';

const API_BASE_URL = (import.meta.env?.VITE_API_BASE_URL as string) || 'http://localhost:8000';

/**
 * Custom error class for API errors.
 */
export class APIException extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public errorType?: string,
    public details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'APIException';
  }
}

/**
 * Create axios client with interceptors.
 */
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 90000, // 90 seconds - increased to handle Google OAuth flow
});

// Request interceptor: Add auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor: Handle errors and token refresh
apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError<APIError>) => {
    const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };

    // Handle 401 errors (unauthorized)
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      // Try to refresh token
      const refreshToken = localStorage.getItem('refresh_token');
      if (refreshToken) {
        try {
          const authService = new AuthService();
          const response = await authService.refreshToken(refreshToken);
          
          localStorage.setItem('access_token', response.access_token);
          if (response.refresh_token) {
            localStorage.setItem('refresh_token', response.refresh_token);
          }

          // Retry original request
          if (originalRequest.headers) {
            originalRequest.headers.Authorization = `Bearer ${response.access_token}`;
          }
          return apiClient(originalRequest);
        } catch (refreshError) {
          // Refresh failed, clear tokens and redirect
          localStorage.removeItem('access_token');
          localStorage.removeItem('refresh_token');
          if (window.location.pathname !== '/') {
            window.location.href = '/';
          }
        }
      } else {
        // No refresh token, redirect to login
        localStorage.removeItem('access_token');
        if (window.location.pathname !== '/') {
          window.location.href = '/';
        }
      }
    }

    // Transform error to APIException
    const apiError = error.response?.data;
    throw new APIException(
      apiError?.error || apiError?.detail || error.message || 'An error occurred',
      error.response?.status,
      apiError?.error_type,
      apiError?.details
    );
  }
);

/**
 * Helper function to handle API responses.
 */
function handleResponse<T>(response: AxiosResponse<T>): T {
  return response.data;
}

/**
 * AuthService - Authentication API endpoints.
 */
class AuthService {
  /**
   * Exchange Google OAuth2 token for JWT tokens.
   * Includes retry logic with exponential backoff for resilience.
   */
  async exchangeToken(googleToken: string): Promise<TokenResponse> {
    const maxRetries = 2;
    let lastError: Error | null = null;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const response = await apiClient.post<TokenResponse>('/api/v1/auth/token', {
          google_token: googleToken,
        });
        return handleResponse(response);
      } catch (error) {
        lastError = error instanceof APIException ? error : new APIException('Failed to exchange token');
        
        // Don't retry on 401 (authentication errors) or 400 (bad request)
        if (error instanceof APIException && 
            (error.statusCode === 401 || error.statusCode === 400)) {
          throw error;
        }
        
        // Retry with exponential backoff
        if (attempt < maxRetries) {
          const delay = Math.pow(2, attempt) * 1000; // 1s, 2s, 4s
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }
        
        throw lastError;
      }
    }
    
    throw lastError || new APIException('Failed to exchange token');
  }

  /**
   * Refresh access token using refresh token.
   */
  async refreshToken(refreshToken: string): Promise<RefreshTokenResponse> {
    try {
      const response = await apiClient.post<RefreshTokenResponse>('/api/v1/auth/refresh', {
        refresh_token: refreshToken,
      });
      return handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to refresh token', undefined, undefined, { originalError: error });
    }
  }

  /**
   * Get current authenticated user information.
   */
  async getCurrentUser(): Promise<UserResponse> {
    try {
      const response = await apiClient.get<UserResponse>('/api/v1/auth/me');
      return handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to get user information', undefined, undefined, { originalError: error });
    }
  }

  /**
   * Logout (client-side token discard).
   */
  async logout(): Promise<void> {
    try {
      await apiClient.post('/api/v1/auth/logout');
    } catch (error) {
      // Logout is best-effort, don't throw on error
      console.warn('Logout request failed:', error);
    } finally {
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
    }
  }
}

/**
 * FolderService - Folder processing API endpoints.
 */
class FolderService {
  /**
   * Process a Google Drive folder for indexing.
   */
  async processFolder(request: ProcessFolderRequest): Promise<ProcessFolderResponse> {
    try {
      const response = await apiClient.post<ProcessFolderResponse>(
        '/api/v1/folders/process',
        request
      );
      return handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to process folder', undefined, undefined, { originalError: error });
    }
  }

  /**
   * Get processing status for a folder.
   */
  async getFolderStatus(folderId: string): Promise<FolderStatusResponse> {
    try {
      const response = await apiClient.get<FolderStatusResponse>(
        `/api/v1/folders/${folderId}/status`
      );
      return handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to get folder status', undefined, undefined, { originalError: error });
    }
  }

  /**
   * Get list of files in a processed folder.
   */
  async getFolderFiles(folderId: string): Promise<FileMetadata[]> {
    try {
      const response = await apiClient.get<FileMetadata[]>(
        `/api/v1/folders/${folderId}/files`
      );
      return handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to get folder files', undefined, undefined, { originalError: error });
    }
  }

  /**
   * Get tree structure representing the exact Google Drive folder hierarchy.
   */
  async getFolderTree(folderId: string): Promise<TreeNode> {
    try {
      const response = await apiClient.get<TreeNode>(
        `/api/v1/folders/${folderId}/tree`
      );
      return handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to get folder tree', undefined, undefined, { originalError: error });
    }
  }

  /**
   * Get metadata for a processed folder.
   */
  async getFolderMetadata(folderId: string): Promise<FolderMetadata> {
    try {
      const response = await apiClient.get<FolderMetadata>(
        `/api/v1/folders/${folderId}/metadata`
      );
      return handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to get folder metadata', undefined, undefined, { originalError: error });
    }
  }

  /**
   * Delete a folder and all associated data.
   */
  async deleteFolder(folderId: string): Promise<void> {
    try {
      const response = await apiClient.delete(
        `/api/v1/folders/${folderId}`
      );
      handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to delete folder', undefined, undefined, { originalError: error });
    }
  }

  /**
   * Get list of folders processed by the current user.
   */
  async getUserFolders(): Promise<FolderMetadata[]> {
    try {
      const response = await apiClient.get<FolderMetadata[]>('/api/v1/folders');
      return handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to get user folders', undefined, undefined, { originalError: error });
    }
  }

  /**
   * Get knowledge graph for a folder.
   */
  async getKnowledgeGraph(folderId: string): Promise<KnowledgeGraphResponse> {
    try {
      const response = await apiClient.get<KnowledgeGraphResponse>(
        `/api/v1/folders/${folderId}/graph`
      );
      return handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to load knowledge graph', undefined, undefined, { originalError: error });
    }
  }

  /**
   * Get folder summary and learning status.
   */
  async getFolderSummary(folderId: string): Promise<FolderSummary> {
    try {
      const response = await apiClient.get<FolderSummary>(
        `/api/v1/folders/${folderId}/summary`
      );
      return handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to get folder summary', undefined, undefined, { originalError: error });
    }
  }

  /**
   * Trigger folder summary regeneration.
   */
  async regenerateFolderSummary(folderId: string): Promise<{ message: string; status: string }> {
    try {
      const response = await apiClient.post<{ message: string; status: string }>(
        `/api/v1/folders/${folderId}/summary/regenerate`
      );
      return handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to regenerate folder summary', undefined, undefined, { originalError: error });
    }
  }
}

/**
 * ChatService - Chat and query API endpoints.
 */
class ChatService {
  /**
   * Get all conversations for a specific folder.
   */
  async getFolderConversations(folderId: string): Promise<Conversation[]> {
    try {
      const response = await apiClient.get<Conversation[]>(`/api/v1/chat/folders/${folderId}/conversations`);
      return handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to get folder conversations', undefined, undefined, { originalError: error });
    }
  }

  /**
   * Create a new conversation for a folder.
   */
  async createConversation(folderId: string, title?: string): Promise<Conversation> {
    try {
      const response = await apiClient.post<Conversation>(`/api/v1/chat/folders/${folderId}/conversations`, {
        title: title || 'New Chat'
      });
      return handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to create conversation', undefined, undefined, { originalError: error });
    }
  }

  /**
   * Get all messages for a specific conversation.
   */
  async getConversationMessages(conversationId: string): Promise<ChatMessage[]> {
    try {
      const response = await apiClient.get<ChatMessage[]>(`/api/v1/chat/conversations/${conversationId}/messages`);
      return handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to get conversation messages', undefined, undefined, { originalError: error });
    }
  }

  /**
   * Delete a conversation.
   */
  async deleteConversation(conversationId: string): Promise<void> {
    try {
      const response = await apiClient.delete(`/api/v1/chat/conversations/${conversationId}`);
      handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to delete conversation', undefined, undefined, { originalError: error });
    }
  }

  /**
   * Send a chat query and get AI response.
   */
  async query(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await apiClient.post<ChatResponse>('/api/v1/chat/query', request);
      return handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Failed to send chat query', undefined, undefined, { originalError: error });
    }
  }

  /**
   * Send a chat query and stream the AI response.
   *
   * @param request Chat request
   * @param onToken Callback for each token chunk
   * @param onCitations Callback when citations are received
   * @param onDone Callback when streaming is complete
   * @param onError Callback for errors
   * @param onStatus Callback for status updates (e.g., "Retrieving context...")
   */
  async queryStream(
    request: ChatRequest,
    onToken: (token: string) => void,
    onCitations: (citations: Citation[]) => void,
    onDone: (conversationId: string, debugData?: any) => void,
    onError: (error: Error) => void,
    onStatus?: (message: string) => void
  ): Promise<void> {
    try {
      const token = localStorage.getItem('access_token');
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const response = await fetch(`${API_BASE_URL}/api/v1/chat/query/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new APIException(`Streaming request failed: ${response.statusText}`, response.status);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new APIException('Response body is not readable');
      }

      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === 'status') {
                // Status update (e.g., "Retrieving context...", "Generating response...")
                if (onStatus) {
                  onStatus(data.message);
                }
              } else if (data.type === 'token') {
                onToken(data.content);
              } else if (data.type === 'citations') {
                onCitations(data.citations || []);
              } else if (data.type === 'done') {
                onDone(data.conversation_id, data.debug);
                return;
              } else if (data.type === 'error') {
                throw new APIException(data.content || 'Streaming error');
              }
            } catch (parseError) {
              console.error('Failed to parse SSE data:', parseError);
            }
          }
        }
      }
    } catch (error) {
      if (error instanceof APIException) {
        onError(error);
      } else {
        onError(new APIException('Failed to stream chat query', undefined, undefined, { originalError: error }));
      }
    }
  }
}

/**
 * SystemService - System and health check endpoints.
 */
class SystemService {
  /**
   * Check API health status.
   */
  async healthCheck(): Promise<HealthCheck> {
    try {
      const response = await apiClient.get<HealthCheck>('/health');
      return handleResponse(response);
    } catch (error) {
      if (error instanceof APIException) {
        throw error;
      }
      throw new APIException('Health check failed', undefined, undefined, { originalError: error });
    }
  }
}

// Export service instances
export const authService = new AuthService();
export const folderService = new FolderService();
export const chatService = new ChatService();
export const systemService = new SystemService();

// APIException is already exported above, no need to re-export

// Export axios instance for advanced usage
export { apiClient };
