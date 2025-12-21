/**
 * Centralized API state management hook.
 * 
 * Provides loading states, error handling, and optimistic updates
 * for API operations across the application.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  authService,
  folderService,
  chatService,
  APIException,
} from '../services/api';
import {
  TokenResponse,
  UserResponse,
  ProcessFolderRequest,
  ChatMessage,
  Citation,
} from '../services/types';

/**
 * Generic API state interface.
 */
interface APIState<T = any> {
  data: T | null;
  isLoading: boolean;
  error: APIException | null;
  lastUpdated: Date | null;
}

// APIOperationResult interface removed - not used

/**
 * useAPI Hook
 * 
 * Centralized API state management with loading states,
 * error handling, and retry logic.
 * 
 * @example
 * ```tsx
 * const { data, isLoading, error, execute } = useAPI(() => 
 *   folderService.getUserFolders()
 * );
 * 
 * useEffect(() => {
 *   execute();
 * }, []);
 * ```
 */
export function useAPI<T>(
  apiCall: () => Promise<T>,
  options: {
    immediate?: boolean;
    onSuccess?: (data: T) => void;
    onError?: (error: APIException) => void;
    retryCount?: number;
    retryDelay?: number;
  } = {}
) {
  const {
    immediate = false,
    onSuccess,
    onError,
    retryCount = 0,
    retryDelay = 1000,
  } = options;

  const [state, setState] = useState<APIState<T>>({
    data: null,
    isLoading: immediate,
    error: null,
    lastUpdated: null,
  });

  const retryCountRef = useRef(0);
  const abortControllerRef = useRef<AbortController | null>(null);

  const execute = useCallback(
    async (optimisticUpdate?: T): Promise<T | null> => {
      // Cancel previous request if still pending
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      abortControllerRef.current = new AbortController();
      retryCountRef.current = 0;

      // Set optimistic update if provided
      if (optimisticUpdate !== undefined) {
        setState((prev) => ({
          ...prev,
          data: optimisticUpdate,
          isLoading: true,
          error: null,
        }));
      } else {
        setState((prev) => ({
          ...prev,
          isLoading: true,
          error: null,
        }));
      }

      const attemptRequest = async (attempt: number): Promise<T | null> => {
        try {
          const result = await apiCall();
          
          setState({
            data: result,
            isLoading: false,
            error: null,
            lastUpdated: new Date(),
          });

          if (onSuccess) {
            onSuccess(result);
          }

          retryCountRef.current = 0;
          return result;
        } catch (error) {
          const apiError = error instanceof APIException ? error : new APIException(
            error instanceof Error ? error.message : 'Unknown error'
          );

          // Check if we should retry
          if (attempt < retryCount && !abortControllerRef.current?.signal.aborted) {
            retryCountRef.current = attempt + 1;
            await new Promise((resolve) => setTimeout(resolve, retryDelay * attempt));
            return attemptRequest(attempt + 1);
          }

          setState((prev) => ({
            ...prev,
            isLoading: false,
            error: apiError,
          }));

          if (onError) {
            onError(apiError);
          }

          throw apiError;
        }
      };

      try {
        return await attemptRequest(0);
      } catch (error) {
        return null;
      }
    },
    [apiCall, onSuccess, onError, retryCount, retryDelay]
  );

  const reset = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setState({
      data: null,
      isLoading: false,
      error: null,
      lastUpdated: null,
    });
    retryCountRef.current = 0;
  }, []);

  // Execute immediately if requested
  if (immediate && !state.data && !state.isLoading && !state.error) {
    execute();
  }

  return {
    ...state,
    execute,
    reset,
    retry: () => execute(),
  };
}

/**
 * useAuth Hook
 * 
 * Authentication API operations with state management.
 */
export function useAuth() {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<APIException | null>(null);

  const login = useCallback(async (googleToken: string): Promise<TokenResponse> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await authService.exchangeToken(googleToken);
      
      localStorage.setItem('access_token', response.access_token);
      if (response.refresh_token) {
        localStorage.setItem('refresh_token', response.refresh_token);
      }

      // Fetch user info
      const userInfo = await authService.getCurrentUser();
      setUser(userInfo);

      return response;
    } catch (err) {
      const apiError = err instanceof APIException ? err : new APIException('Login failed');
      setError(apiError);
      throw apiError;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const logout = useCallback(async (): Promise<void> => {
    setIsLoading(true);
    try {
      await authService.logout();
    } catch (err) {
      // Logout is best-effort
      console.warn('Logout error:', err);
    } finally {
      setUser(null);
      setIsLoading(false);
    }
  }, []);

  const refreshUser = useCallback(async (): Promise<void> => {
    setIsLoading(true);
    setError(null);

    try {
      const userInfo = await authService.getCurrentUser();
      setUser(userInfo);
    } catch (err) {
      const apiError = err instanceof APIException ? err : new APIException('Failed to refresh user');
      setError(apiError);
      // Clear user on auth failure
      if (apiError.statusCode === 401) {
        setUser(null);
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    user,
    isLoading,
    error,
    login,
    logout,
    refreshUser,
  };
}

/**
 * useFolders Hook
 * 
 * Folder API operations with state management.
 */
export function useFolders() {
  const getUserFolders = useCallback(
    () => folderService.getUserFolders(),
    []
  );

  const getFolderMetadata = useCallback(
    (folderId: string) => folderService.getFolderMetadata(folderId),
    []
  );

  const getFolderFiles = useCallback(
    (folderId: string) => folderService.getFolderFiles(folderId),
    []
  );

  const getFolderStatus = useCallback(
    (folderId: string) => folderService.getFolderStatus(folderId),
    []
  );

  const processFolder = useCallback(
    (request: ProcessFolderRequest) => folderService.processFolder(request),
    []
  );

  const getKnowledgeGraph = useCallback(
    (folderId: string) => folderService.getKnowledgeGraph(folderId),
    []
  );

  return {
    getUserFolders,
    getFolderMetadata,
    getFolderFiles,
    getFolderStatus,
    processFolder,
    getKnowledgeGraph,
  };
}

/**
 * useChat Hook
 * 
 * Chat API operations with optimistic updates.
 */
export function useChat(folderId: string, initialConversationId: string | null = null) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(initialConversationId);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<APIException | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  // Sync conversationId with initialConversationId when it changes
  useEffect(() => {
    setConversationId(initialConversationId);
    if (initialConversationId) {
      loadMessages(initialConversationId);
    } else {
      setMessages([]);
    }
  }, [initialConversationId]);

  const loadMessages = useCallback(async (convId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const history = await chatService.getConversationMessages(convId);
      setMessages(history);
    } catch (err) {
      const apiError = err instanceof APIException ? err : new APIException('Failed to load chat history');
      setError(apiError);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const sendMessage = useCallback(
    async (content: string, useStreaming: boolean = true): Promise<void> => {
      setIsLoading(true);
      setError(null);
      setStatusMessage(null);

      // Optimistic update: Add user message immediately
      const userMessage: ChatMessage = {
        message_id: `user_${Date.now()}`,
        conversation_id: conversationId || 'new',
        role: 'user',
        content,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMessage]);

      try {
        if (useStreaming) {
          // Streaming mode
          const assistantMessageId = `assistant_${Date.now()}`;
          let assistantContent = '';
          let assistantCitations: Citation[] = [];
          let finalConversationId = conversationId || 'new';
          let assistantMessageCreated = false;

          // Batch streaming updates for better performance
          let updateScheduled = false;
          const scheduleUpdate = () => {
            if (updateScheduled) return;
            updateScheduled = true;
            requestAnimationFrame(() => {
              updateScheduled = false;
              if (!assistantMessageCreated) {
                // First token - create message immediately
                assistantMessageCreated = true;
                const assistantMessage: ChatMessage = {
                  message_id: assistantMessageId,
                  conversation_id: finalConversationId,
                  role: 'assistant',
                  content: assistantContent,
                  timestamp: new Date(),
                  citations: [],
                };
                setMessages((prev) => [...prev, assistantMessage]);
              } else {
                // Subsequent tokens - update existing message
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.message_id === assistantMessageId
                      ? { ...msg, content: assistantContent }
                      : msg
                  )
                );
              }
            });
          };

          // Stream response
          await chatService.queryStream(
            {
              query: content,
              folder_id: folderId,
              conversation_id: conversationId || undefined,
            },
            (token: string) => {
              // Accumulate tokens
              assistantContent += token;
              // Schedule batched update (first token will create message immediately)
              scheduleUpdate();
              // Clear status message once we start receiving tokens
              if (statusMessage) {
                setStatusMessage(null);
              }
            },
            (citations: Citation[]) => {
              // Update citations (progressive citations) - show immediately
              assistantCitations = citations;
              console.log('Citations received:', citations.length, 'citations');
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.message_id === assistantMessageId
                    ? { ...msg, citations: [...assistantCitations] } // Create new array to trigger re-render
                    : msg
                )
              );
            },
            (convId: string) => {
              // Update conversation ID
              finalConversationId = convId;
              if (!conversationId) {
                setConversationId(convId);
              }
              setStatusMessage(''); // Clear status when done
              setIsLoading(false);
            },
            (err: Error) => {
              const apiError = err instanceof APIException ? err : new APIException('Failed to stream message');
              setError(apiError);
              setStatusMessage(''); // Clear status on error
              setIsLoading(false);

              // Remove optimistic messages on error (only remove assistant message if it was created)
              setMessages((prev) =>
                prev.filter((msg) =>
                  msg.message_id !== userMessage.message_id &&
                  (assistantMessageCreated ? msg.message_id !== assistantMessageId : true)
                )
              );

              throw apiError;
            },
            (status: string) => {
              // Update status message (e.g., "Retrieving context...", "Generating response...")
              setStatusMessage(status);
            }
          );
        } else {
          // Non-streaming mode (fallback)
          const response = await chatService.query({
            query: content,
            folder_id: folderId,
            conversation_id: conversationId || undefined,
          });

          // Update conversation ID if new
          if (!conversationId && response.conversation_id) {
            setConversationId(response.conversation_id);
          }

          // Add assistant response
          const assistantMessage: ChatMessage = {
            message_id: `assistant_${Date.now()}`,
            conversation_id: response.conversation_id,
            role: 'assistant',
            content: response.response,
            timestamp: new Date(),
            citations: response.citations,
          };

          setMessages((prev) => [...prev, assistantMessage]);
          setIsLoading(false);
        }
      } catch (err) {
        const apiError = err instanceof APIException ? err : new APIException('Failed to send message');
        setError(apiError);
        setIsLoading(false);
        
        // Remove optimistic user message on error
        setMessages((prev) => prev.filter((msg) => msg.message_id !== userMessage.message_id));
        
        throw apiError;
      }
    },
    [folderId, conversationId]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setConversationId(null);
    setError(null);
  }, []);

  return {
    messages,
    conversationId,
    isLoading,
    error,
    statusMessage,
    sendMessage,
    clearMessages,
  };
}

// Re-export types for convenience
export type { ChatMessage };

