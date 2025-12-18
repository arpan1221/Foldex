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
  title: string;
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
  web_view_link?: string;
  web_content_link?: string;
  is_folder?: boolean;
}

export interface TreeNode {
  id: string;
  name: string;
  is_folder: boolean;
  mime_type: string;
  size: number;
  created_at?: Date | string;
  modified_at?: Date | string;
  web_view_link?: string;
  web_content_link?: string;
  children: TreeNode[];
}

export interface FolderMetadata {
  folder_id: string;
  folder_name: string;
  file_count: number;
  folder_count: number;
  total_size?: number;
  status?: string;
  created_at?: Date | string;
  modified_at?: Date | string;
  updated_at?: Date | string;
  web_view_link?: string;
}

export interface ProcessFolderRequest {
  folder_id: string;
  folder_url?: string;
}

export interface ProcessFolderResponse {
  folder_id: string;
  status: 'processing' | 'completed' | 'failed';
  message: string;
  files_detected?: number;
}

export interface FolderStatusResponse {
  folder_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  files_processed: number;
  total_files: number;
  progress: number; // 0.0 to 1.0
  error?: string;
}

export interface ProcessingStatus {
  type: 'processing_started' | 'files_detected' | 'file_processing' | 'file_processed' | 'file_error' | 'building_graph' | 'graph_complete' | 'processing_complete' | 'processing_error' | 'connected' | 'pong' | 'status' | 'error' | 'folder_structure' | 'folder_discovered' | 'folder_completed';
  message?: string;
  progress?: number; // 0.0 to 1.0
  file_name?: string;
  file_id?: string;
  folder_id?: string;
  files_processed?: number;
  total_files?: number;
  failed_files?: number;
  file_index?: number;
  error?: string;
  timestamp?: string;
  folder_name?: string;
  total_subfolders?: number;
  subfolders?: any[];
  folders?: any[];
  file_path?: string;
  parent_folder_id?: string;
  [key: string]: any;
}

// ============================================================================
// Authentication Types
// ============================================================================

export interface User {
  user_id: string;
  email: string;
  name?: string;
  picture?: string;
  verified_email?: boolean;
}

export interface TokenRequest {
  google_token: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token?: string;
  token_type: 'bearer';
  expires_in: number;
}

export interface RefreshTokenRequest {
  refresh_token: string;
}

export interface RefreshTokenResponse {
  access_token: string;
  token_type: 'bearer';
  refresh_token?: string;
}

export interface RefreshTokenResponse {
  access_token: string;
  token_type: 'bearer';
  expires_in: number;
}

export interface UserResponse {
  user_id: string;
  email: string;
  name?: string;
  picture?: string;
  verified_email: boolean;
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

export interface KnowledgeGraphResponse {
  nodes: KnowledgeGraphNode[];
  edges: KnowledgeGraphEdge[];
  relationships: Relationship[];
}

// ============================================================================
// API Response Types
// ============================================================================

export interface APIError {
  error: string;
  error_type?: string;
  detail?: string;
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
  type: 'processing_started' | 'file_processed' | 'processing_complete' | 'processing_error' | 'files_detected' | 'file_processing' | 'file_error' | 'building_graph' | 'graph_complete';
  folder_id: string;
  message?: string;
  progress?: number;
  file_name?: string;
  file_id?: string;
  files_processed?: number;
  total_files?: number;
  failed_files?: number;
  error?: string;
  timestamp?: string;
}

// ============================================================================
// API Request/Response Helpers
// ============================================================================

export interface APIResponse<T = any> {
  data: T;
  status: number;
  statusText: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}
