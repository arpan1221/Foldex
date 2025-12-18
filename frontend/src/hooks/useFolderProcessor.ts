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
          // Pass through additional fields for folder structure
          ...message,
        };

        setStatus(newStatus);

        // Update files list if provided
        if (message.files && Array.isArray(message.files)) {
          setFiles(message.files);
        }

        // Handle completion or error
        if (message.type === 'processing_complete') {
          // Update status immediately with completion
          setStatus({
            ...newStatus,
            progress: 1.0,
            message: message.message || 'Processing completed successfully',
          });

          // Set isProcessing to false to stop the infinite loop
          setIsProcessing(false);

          // Keep WebSocket connected to allow UI to show completion state
          // It will be disconnected when navigating away or on component unmount

          console.log('Processing completed', {
            files_processed: message.files_processed,
            total_files: message.total_files,
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
