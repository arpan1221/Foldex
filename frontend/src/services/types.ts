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
  // Basic citation info
  citation_number?: number;  // Inline citation number [1], [2], etc.
  file_id: string;
  file_name: string;
  chunk_id: string;
  mime_type?: string;

  // Location metadata
  page_number?: number;
  page_numbers?: number[];  // Multiple pages from same file
  page_display?: string;  // Formatted page display (e.g., "p.1, p.3")
  chunk_index?: number;

  // Audio/video metadata
  start_time?: number;
  end_time?: number;
  timestamp?: number;

  // Granular citation data
  claim_text?: string;  // The claim from the response
  exact_quote?: string;  // Exact quote from source
  quote_confidence?: number;  // Confidence score 0.0-1.0

  // Character-level positions for highlighting
  char_start?: number;
  char_end?: number;

  // Sentence/paragraph level
  sentence_index?: number;
  paragraph_index?: number;

  // Context snippets
  context_before?: string;
  context_after?: string;

  // UI-formatted data
  highlight?: {
    start: number;
    end: number;
    text: string;
  };

  context?: {
    before: string;
    quote: string;
    after: string;
  };

  location?: {
    page?: number;
    paragraph?: number;
    sentence?: number;
    timestamp?: string;
  };

  // Legacy fields
  confidence?: number;
  google_drive_url?: string;  // Clickable link to source file
  content_preview?: string;  // Preview of the cited content (short, for lists)
  chunk_content?: string;  // Full chunk content for hover tooltip

  // Full metadata
  metadata?: Record<string, any>;
}

export interface ChatRequest {
  query: string;
  folder_id?: string;
  conversation_id?: string;
  file_id?: string;  // Optional file ID for file-specific chat
  debug?: boolean;
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

export interface FolderSummary {
  folder_id: string;
  folder_name: string | null;
  summary: string | null;
  learning_status: 'learning_pending' | 'learning_in_progress' | 'learning_complete' | 'learning_failed' | null;
  insights: {
    total_files: number;
    unique_file_types: number;
    top_themes: string[];
    key_relationships: number;
  } | null;
  file_type_distribution: Record<string, number> | null;
  entity_summary: {
    top_entities: Array<{ entity: string; count: number }>;
    top_themes: Array<{ theme: string; count: number }>;
  } | null;
  relationship_summary: Array<{
    source_file: string;
    target_file: string;
    relationship_type: string;
    confidence: number;
  }> | null;
  capabilities: string[] | null;
  graph_statistics: {
    node_count: number;
    edge_count: number;
    relationship_types: number;
  } | null;
  learning_completed_at: Date | string | null;
}

export interface ProcessingStatus {
  type: 'processing_started' | 'files_detected' | 'file_processing' | 'file_processed' | 'file_error' | 'building_graph' | 'graph_complete' | 'learning_started' | 'generating_summary' | 'summary_progress' | 'summary_complete' | 'summary_error' | 'processing_complete' | 'processing_error' | 'connected' | 'pong' | 'status' | 'error' | 'folder_structure' | 'folder_discovered' | 'folder_completed';
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
  summary_data?: {
    total_files: number;
    unique_file_types: number;
    top_themes: string[];
  };
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
  type: 'processing_started' | 'file_processed' | 'processing_complete' | 'processing_error' | 'files_detected' | 'file_processing' | 'file_error' | 'learning_started' | 'summary_progress' | 'summary_complete' | 'summary_error' | 'building_graph' | 'graph_complete';
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
