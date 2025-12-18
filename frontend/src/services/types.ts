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

export interface FileMetadata {
  file_id: string;
  file_name: string;
  mime_type: string;
  size: number;
}

export interface ProcessingStatus {
  type: string;
  message?: string;
  progress?: number;
  file_name?: string;
}

