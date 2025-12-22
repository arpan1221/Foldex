import { useState, useCallback, useEffect, useRef } from 'react';
import { folderService, APIException } from '../services/api';
import { websocketService } from '../services/websocket';
import { ProcessingStatus, FileMetadata, ProcessFolderRequest } from '../services/types';
import { eventSystem } from '../utils/eventSystem';

/**
 * useFolderProcessor Hook
 * 
 * Manages folder processing state with WebSocket integration for real-time updates.
 * Handles processing initiation, progress tracking, and error management.
 * 
 * @returns {UseFolderProcessorReturn} Processing state and methods
 * 
 * @example
 * ```tsx
 * const { processFolder, isProcessing, status, files, error } = useFolderProcessor();
 * 
 * await processFolder('folder_id_123');
 * ```
 */
interface UseFolderProcessorReturn {
  processFolder: (folderId: string) => Promise<void>;
  isProcessing: boolean;
  status: ProcessingStatus | null;
  files: FileMetadata[];
  error: Error | null;
  reset: () => void;
}

export const useFolderProcessor = (): UseFolderProcessorReturn => {
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [status, setStatus] = useState<ProcessingStatus | null>(null);
  const [files, setFiles] = useState<FileMetadata[]>([]);
  const [error, setError] = useState<APIException | null>(null);
  const currentFolderIdRef = useRef<string | null>(null);
  const wsHandlerRef = useRef<((message: any) => void) | null>(null);
  // Keep a stable reference to the last status to prevent disappearing
  const lastStatusRef = useRef<ProcessingStatus | null>(null);

  /**
   * Reset processing state
   */
  const reset = useCallback(() => {
    setIsProcessing(false);
    // Don't clear status immediately - let it persist for a bit
    // Only clear if explicitly resetting (e.g., new folder)
    setStatus(null);
    lastStatusRef.current = null;
    setFiles([]);
    setError(null);
    
    // Disconnect WebSocket if connected
    if (currentFolderIdRef.current) {
      websocketService.disconnect(currentFolderIdRef.current);
      currentFolderIdRef.current = null;
    }
  }, []);

  /**
   * Process a folder
   */
  const processFolder = useCallback(async (folderId: string) => {
    // Reset previous state
    reset();

    setIsProcessing(true);
    setError(null);
    setStatus({
      type: 'processing_started',
      message: 'Initializing folder processing...',
      progress: 0,
    });
    currentFolderIdRef.current = folderId;

    try {
      // Start processing via API
      const request: ProcessFolderRequest = { folder_id: folderId };
      await folderService.processFolder(request);

      // Set up WebSocket connection for real-time updates
      const handler = (message: any) => {
        console.log('WebSocket message received:', message);

        // Update status based on message type
        // Pass through ALL message data, not just specific fields
        const newStatus: ProcessingStatus = {
          type: message.type || 'processing_started',
          message: message.message,
          progress: message.progress,
          file_name: message.file_name,
          file_id: message.file_id,
          folder_id: message.folder_id,
          files_processed: message.files_processed,
          total_files: message.total_files,
          error: message.error,
          summary_data: message.summary_data,
          // Pass through additional fields for folder structure
          ...message,
        };

        // Handle completion or error
        if (message.type === 'processing_complete') {
          // Update status immediately with completion
          const completedStatus: ProcessingStatus = {
            ...newStatus,
            type: 'processing_complete', // Explicitly set type
            progress: 1.0,
            message: message.message || 'Processing completed successfully',
            folder_id: message.folder_id || newStatus.folder_id,
            files_processed: message.files_processed,
            total_files: message.total_files,
          };
          
          // Update state immediately
          setStatus(completedStatus);
          lastStatusRef.current = completedStatus; // Persist status

          // IMPORTANT: Set isProcessing to false so navigation can happen immediately
          // File chunking is complete - users can now chat with files
          // Learning phase (summarization) happens in background and doesn't block navigation
          setIsProcessing(false);

          // Emit robust event for navigation and other components
          // Use setTimeout to ensure state is updated before emitting event
          setTimeout(() => {
            if (message.folder_id) {
              console.log('📤 Emitting processing_complete event:', {
                folder_id: message.folder_id,
                files_processed: message.files_processed,
                total_files: message.total_files,
              });
              eventSystem.emit('processing_complete', {
                folder_id: message.folder_id,
                files_processed: message.files_processed,
                total_files: message.total_files,
              });
            }
          }, 100);

          console.log('Processing completed - chunking done, ready for chat', {
            folder_id: message.folder_id,
            files_processed: message.files_processed,
            total_files: message.total_files,
            status: completedStatus,
          });
        } else if (message.type === 'summary_complete' || message.type === 'summary_error') {
          // Learning/summarization phase is complete
          // IMPORTANT: Set isProcessing to false so navigation can happen
          // Knowledge graph building happens in background and doesn't block
          setIsProcessing(false);
          
          // Update status with final learning status
          setStatus(newStatus);
          lastStatusRef.current = newStatus; // Persist status
          
          // Dispatch custom event for summary_complete so sidebar and other components can update
          if (message.type === 'summary_complete' && message.folder_id) {
            const detail = {
              folder_id: message.folder_id,
              summary_data: message.summary_data,
            };
            // Use robust event system
            eventSystem.emit('summary_complete', detail);
            // Also dispatch window event for backward compatibility
            window.dispatchEvent(new CustomEvent('summary_complete', { detail }));
          }
          
          if (message.type === 'summary_error' && message.folder_id) {
            window.dispatchEvent(new CustomEvent('summary_error', {
              detail: {
                folder_id: message.folder_id,
                error: message.error,
              }
            }));
          }
          
          console.log('Folder learning completed', {
            type: message.type,
            summary_data: message.summary_data,
          });
        } else if (message.type === 'learning_started' || message.type === 'summary_progress') {
          // Learning/summarization progress messages should NOT affect isProcessing state
          // These are background tasks that don't block file-specific chatting
          // Just update status for UI display, but keep isProcessing as-is
          setStatus(newStatus);
          
          console.log('Learning/summarization status update (background)', {
            type: message.type,
          });
        } else if (message.type === 'building_graph' || message.type === 'graph_complete' || message.type === 'graph_error') {
          // Knowledge graph messages should NOT affect isProcessing state
          // These are background tasks that don't block navigation
          // Just update status for UI display, but keep isProcessing as-is
          setStatus(newStatus);
          
          // Dispatch custom events for graph_complete so sidebar can update
          if (message.type === 'graph_complete' && message.folder_id) {
            const detail = {
              folder_id: message.folder_id,
              graph_stats: message.graph_stats,
            };
            // Use robust event system
            eventSystem.emit('graph_complete', detail);
            // Also dispatch window event for backward compatibility
            window.dispatchEvent(new CustomEvent('graph_complete', { detail }));
          }
          
          if (message.type === 'graph_error' && message.folder_id) {
            window.dispatchEvent(new CustomEvent('graph_error', {
              detail: {
                folder_id: message.folder_id,
                error: message.error,
              }
            }));
          }
          
          console.log('Knowledge graph status update (background)', {
            type: message.type,
          });
        } else if (message.type === 'processing_error') {
          setIsProcessing(false);
          setError(new Error(message.error || 'Processing failed'));
          
          // Disconnect WebSocket on error
          if (currentFolderIdRef.current) {
            websocketService.disconnect(currentFolderIdRef.current);
            currentFolderIdRef.current = null;
          }
        } else if (message.type === 'file_processed') {
          // Update progress for file processing
          if (message.files_processed !== undefined && message.total_files !== undefined) {
            const progress = message.files_processed / message.total_files;
            setStatus({
              ...newStatus,
              progress,
            });
          }
        } else {
          // For all other messages (including learning messages), update status
          // This ensures learning_started, summary_progress, etc. are displayed
          setStatus(newStatus);
          lastStatusRef.current = newStatus; // Persist status
        }

        // Update files list if provided
        if (message.files && Array.isArray(message.files)) {
          setFiles(message.files);
        }
      };

      wsHandlerRef.current = handler;
      websocketService.connect(folderId, handler);

      // Update status to show connection established
      setStatus({
        type: 'processing_started',
        message: 'Connected. Starting to process files...',
        progress: 0.1,
      });
    } catch (err) {
      const apiError = err instanceof APIException 
        ? err 
        : new APIException(err instanceof Error ? err.message : 'Failed to process folder');
      setError(apiError);
      setIsProcessing(false);
      setStatus({
        type: 'processing_error',
        message: 'Failed to start processing',
        error: apiError.message,
      });

      // Clean up WebSocket on error
      if (currentFolderIdRef.current) {
        websocketService.disconnect(currentFolderIdRef.current);
        currentFolderIdRef.current = null;
      }
    }
  }, [reset]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (currentFolderIdRef.current) {
        websocketService.disconnect(currentFolderIdRef.current);
      }
    };
  }, []);

  // Return status, but fall back to lastStatusRef if status is null
  // This prevents the progress bar from disappearing during state transitions
  const stableStatus = status || lastStatusRef.current;

  return {
    processFolder,
    isProcessing,
    status: stableStatus,
    files,
    error,
    reset,
  };
};
