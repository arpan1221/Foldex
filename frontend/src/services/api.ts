import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

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
    const response = await apiClient.post('/api/v1/auth/token', {
      access_token: googleToken,
    });
    return response.data;
  }

  async getCurrentUser(): Promise<any> {
    const response = await apiClient.get('/api/v1/auth/me');
    return response.data;
  }
}

export const folderService = new APIService();
export const chatService = new ChatService();
export const authService = new AuthService();

