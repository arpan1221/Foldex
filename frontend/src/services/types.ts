/**
 * TypeScript type definitions for Foldex API communication.
 * 
 * These types match the backend Pydantic models and ensure type safety
 * across the frontend application.
 */

// ============================================================================
// Chat & Messaging Types
// ============================================================================

export interface ChatMessage {
  message_id: string;
  conversation_id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date | string;
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

export interface ChatRequest {
  query: string;
  folder_id?: string;
  conversation_id?: string;
}

export interface ChatResponse {
  response: string;
  citations: Citation[];
  conversation_id: string;
}

export interface Conversation {
  conversation_id: string;
  user_id: string;
  folder_id?: string;
  created_at: Date | string;
  updated_at: Date | string;
  messages?: ChatMessage[];
}

// ============================================================================
// File & Folder Types
// ============================================================================

export interface FileMetadata {
  file_id: string;
  file_name: string;
  mime_type: string;
  size: number;
  folder_id?: string;
  created_at?: Date | string;
  modified_at?: Date | string;
}

export interface FolderMetadata {
  folder_id: string;
  folder_name: string;
  file_count: number;
  total_size: number;
  created_at?: Date | string;
  modified_at?: Date | string;
}

export interface ProcessFolderRequest {
  folder_id: string;
}

export interface ProcessFolderResponse {
  folder_id: string;
  status: 'processing' | 'completed' | 'failed';
  message: string;
  files_processed?: number;
  total_files?: number;
}

export interface ProcessingStatus {
  type: 'processing_started' | 'file_processed' | 'processing_complete' | 'processing_error';
  message?: string;
  progress?: number; // 0.0 to 1.0
  file_name?: string;
  file_id?: string;
  folder_id?: string;
  files_processed?: number;
  total_files?: number;
  error?: string;
}

// ============================================================================
// Authentication Types
// ============================================================================

export interface User {
  user_id: string;
  email: string;
  name?: string;
  google_id?: string;
}

export interface TokenRequest {
  access_token: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: 'bearer';
}

// ============================================================================
// Knowledge Graph Types
// ============================================================================

export interface Relationship {
  source: string;
  target: string;
  type: 'entity_overlap' | 'temporal' | 'cross_reference' | 'topical_similarity' | 'implementation_gap';
  confidence: number;
}

export interface KnowledgeGraphNode {
  node_id: string;
  label: string;
  type: 'document' | 'entity' | 'chunk';
  metadata?: Record<string, unknown>;
}

export interface KnowledgeGraphEdge {
  source: string;
  target: string;
  relationship_type: string;
  confidence: number;
}

// ============================================================================
// API Response Types
// ============================================================================

export interface APIError {
  error: string;
  error_type: string;
  details?: Record<string, unknown>;
}

export interface HealthCheck {
  status: 'healthy' | 'unhealthy';
  version: string;
  environment?: string;
}

// ============================================================================
// WebSocket Message Types
// ============================================================================

export interface WebSocketMessage {
  type: string;
  [key: string]: unknown;
}

export interface ProcessingUpdateMessage extends WebSocketMessage {
  type: 'processing_started' | 'file_processed' | 'processing_complete' | 'processing_error';
  folder_id: string;
  message?: string;
  progress?: number;
  file_name?: string;
  file_id?: string;
  files_processed?: number;
  total_files?: number;
  error?: string;
}

