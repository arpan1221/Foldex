import { useState, useCallback } from 'react';
import { folderService } from '../services/api';
import { websocketService } from '../services/websocket';

interface ProcessingStatus {
  type: string;
  message?: string;
  progress?: number;
  file_name?: string;
}

interface UseFolderProcessorReturn {
  processFolder: (folderId: string) => Promise<void>;
  isProcessing: boolean;
  status: ProcessingStatus | null;
  error: Error | null;
}

export const useFolderProcessor = (): UseFolderProcessorReturn => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState<ProcessingStatus | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const processFolder = useCallback(async (folderId: string) => {
    setIsProcessing(true);
    setError(null);
    setStatus({ type: 'processing_started', message: 'Starting folder processing...' });

    try {
      // Start processing
      await folderService.processFolder(folderId);

      // Connect to WebSocket for updates
      websocketService.connect(folderId, (message) => {
        setStatus({
          type: message.type,
          message: message.message,
          progress: message.progress,
          file_name: message.file_name,
        });

        if (message.type === 'processing_complete' || message.type === 'processing_error') {
          setIsProcessing(false);
        }
      });
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to process folder'));
      setIsProcessing(false);
    }
  }, []);

  return { processFolder, isProcessing, status, error };
};

