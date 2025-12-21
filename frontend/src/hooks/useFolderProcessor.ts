import { useState, useCallback, useEffect, useRef } from 'react';
import { folderService, APIException } from '../services/api';
import { websocketService } from '../services/websocket';
import { ProcessingStatus, FileMetadata, ProcessFolderRequest } from '../services/types';

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

  /**
   * Reset processing state
   */
  const reset = useCallback(() => {
    setIsProcessing(false);
    setStatus(null);
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
          setStatus({
            ...newStatus,
            progress: 1.0,
            message: message.message || 'Processing completed successfully',
          });

          // IMPORTANT: Set isProcessing to false so navigation can happen immediately
          // File chunking is complete - users can now chat with files
          // Learning phase (summarization) happens in background and doesn't block navigation
          setIsProcessing(false);

          console.log('Processing completed - chunking done, ready for chat', {
            files_processed: message.files_processed,
            total_files: message.total_files,
          });
        } else if (message.type === 'summary_complete' || message.type === 'summary_error') {
          // Learning/summarization phase is complete
          // IMPORTANT: Set isProcessing to false so navigation can happen
          // Knowledge graph building happens in background and doesn't block
          setIsProcessing(false);
          
          // Update status with final learning status
          setStatus(newStatus);
          
          // Dispatch custom event for summary_complete so sidebar and other components can update
          if (message.type === 'summary_complete' && message.folder_id) {
            window.dispatchEvent(new CustomEvent('summary_complete', {
              detail: {
                folder_id: message.folder_id,
                summary_data: message.summary_data,
              }
            }));
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
            window.dispatchEvent(new CustomEvent('graph_complete', {
              detail: {
                folder_id: message.folder_id,
                graph_stats: message.graph_stats,
              }
            }));
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

  return {
    processFolder,
    isProcessing,
    status,
    files,
    error,
    reset,
  };
};
