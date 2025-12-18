import axios, { AxiosError } from 'axios';
import { KnowledgeGraphNode, KnowledgeGraphEdge, Relationship, FolderMetadata } from './types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 errors (unauthorized) - redirect to login
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      // Clear token and redirect to login
      localStorage.removeItem('access_token');
      if (window.location.pathname !== '/') {
        window.location.href = '/';
      }
    }
    return Promise.reject(error);
  }
);

export interface ChatMessage {
  message_id: string;
  conversation_id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  citations?: Citation[];
}

export interface Citation {
  file_id: string;
  file_name: string;
  chunk_id: string;
  page_number?: number;
  timestamp?: number;
  confidence: number;
}

export interface ChatResponse {
  response: string;
  citations: Citation[];
  conversation_id: string;
}

class APIService {
  async processFolder(folderId: string): Promise<void> {
    const response = await apiClient.post('/api/v1/folders/process', {
      folder_id: folderId,
    });
    return response.data;
  }

  async getFolderStatus(folderId: string): Promise<any> {
    const response = await apiClient.get(`/api/v1/folders/${folderId}/status`);
    return response.data;
  }

  async getKnowledgeGraph(folderId: string): Promise<{
    nodes: KnowledgeGraphNode[];
    edges: KnowledgeGraphEdge[];
    relationships: Relationship[];
  }> {
    try {
      const response = await apiClient.get(`/api/v1/folders/${folderId}/graph`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(
          error.response?.data?.error || 'Failed to load knowledge graph'
        );
      }
      throw error;
    }
  }

  async getUserFolders(): Promise<FolderMetadata[]> {
    try {
      const response = await apiClient.get('/api/v1/folders');
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(
          error.response?.data?.error || 'Failed to load folders'
        );
      }
      throw error;
    }
  }
}

class ChatService {
  async query(
    query: string,
    folderId: string,
    conversationId?: string
  ): Promise<ChatResponse> {
    const response = await apiClient.post('/api/v1/chat/query', {
      query,
      folder_id: folderId,
      conversation_id: conversationId,
    });
    return response.data;
  }
}

class AuthService {
  async exchangeToken(googleToken: string): Promise<{ access_token: string }> {
    try {
      const response = await apiClient.post('/api/v1/auth/token', {
        access_token: googleToken,
      });
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(
          error.response?.data?.error || 'Failed to exchange token'
        );
      }
      throw error;
    }
  }

  async getCurrentUser(): Promise<{
    user_id: string;
    sub: string;
    email: string;
    name?: string;
    picture?: string;
    google_id?: string;
  }> {
    try {
      const response = await apiClient.get('/api/v1/auth/me');
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(
          error.response?.data?.error || 'Failed to get user information'
        );
      }
      throw error;
    }
  }
}

export const folderService = new APIService();
export const chatService = new ChatService();
export const authService = new AuthService();

