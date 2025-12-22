import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useFolderProcessor } from '../../hooks/useFolderProcessor';
import { chatService, folderService, APIException } from '../../services/api';
import FolderUploadInterface from './FolderUploadInterface';
import FileOverview from './FileOverview';
import { isValidGoogleDriveUrl, extractFolderId } from '../../utils/validators';
import { eventSystem } from '../../utils/eventSystem';

/**
 * FolderInput Component
 * 
 * Google Drive folder URL input with validation and processing initiation.
 * Follows Figma wireframe design with dark gradient background.
 */
const FolderInput: React.FC = () => {
  const navigate = useNavigate();
  const [folderUrl, setFolderUrl] = useState('');
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [processedFolderId, setProcessedFolderId] = useState<string | null>(null);
  const { processFolder, isProcessing, status, error, files } = useFolderProcessor();
  const navigationInitiatedRef = useRef<boolean>(false);
  const navigationTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Validate URL on change
  useEffect(() => {
    if (folderUrl.trim() && !isValidating) {
      const isValid = isValidGoogleDriveUrl(folderUrl);
      if (!isValid && folderUrl.length > 20) {
        setValidationError('Please enter a valid Google Drive folder URL');
      } else {
        setValidationError(null);
      }
    } else if (!folderUrl.trim()) {
      setValidationError(null);
    }
  }, [folderUrl, isValidating]);

  // Show message when chunking completes - file-specific chat is now available
  const [showChatAvailable, setShowChatAvailable] = useState(false);
  
  // Robust auto-navigation using event system
  useEffect(() => {
    const targetFolderId = processedFolderId || status?.folder_id;

    // Debug logging
    console.log('🔍 Auto-navigation check:', {
      targetFolderId,
      navigationInitiated: navigationInitiatedRef.current,
      statusType: status?.type,
      isProcessing,
      processedFolderId,
      statusFolderId: status?.folder_id,
    });

    if (!targetFolderId) {
      return;
    }

    // Only prevent navigation if it's already initiated AND we're still on the same folder
    if (navigationInitiatedRef.current) {
      console.log('⏸️ Navigation already initiated, skipping');
      return;
    }

    const performNavigation = async (folderId: string) => {
      // Double-check to prevent race conditions
      if (navigationInitiatedRef.current) {
        console.log('⚠️ Navigation already initiated (double-check), skipping');
        return;
      }

      console.log('✅ Navigation triggered for folder:', folderId);
      navigationInitiatedRef.current = true;
      setShowChatAvailable(true);

      // Clear any existing timer
      if (navigationTimerRef.current) {
        clearTimeout(navigationTimerRef.current);
        navigationTimerRef.current = null;
      }

      // Navigate after a short delay to show the success message
      navigationTimerRef.current = setTimeout(async () => {
        try {
          console.log('📝 Creating initial conversation for folder:', folderId);
          const newConv = await chatService.createConversation(folderId, 'Initial Chat');
          console.log('🚀 Navigating to chat:', `/chat/${folderId}/${newConv.conversation_id}`);
          navigate(`/chat/${folderId}/${newConv.conversation_id}`);
        } catch (err) {
          console.error('Failed to create conversation:', err);
          // Navigate to folder without conversation ID
          console.log('🚀 Navigating to chat (fallback):', `/chat/${folderId}`);
          navigate(`/chat/${folderId}`);
        } finally {
          navigationTimerRef.current = null;
        }
      }, 1500);
    };

    // Method 1: Listen to status change (direct from hook) - most immediate
    if (status?.type === 'processing_complete' && status?.folder_id === targetFolderId) {
      console.log('📊 Processing complete status detected (Method 1):', targetFolderId);
      if (!processedFolderId && status.folder_id) {
        setProcessedFolderId(status.folder_id);
      }
      // Small delay to ensure state is updated
      setTimeout(() => performNavigation(targetFolderId), 100);
      return; // Exit early if Method 1 triggers
    }

    // Method 2: Listen to event system (more robust, cross-component)
    const unsubscribe = eventSystem.on('processing_complete', (detail) => {
      console.log('📡 Processing complete event received (Method 2):', detail.folder_id);
      // Match any target folder ID or the detail folder ID
      const matchesFolder = detail.folder_id === targetFolderId || 
                           detail.folder_id === processedFolderId ||
                           detail.folder_id === status?.folder_id;
      
      if (matchesFolder && !navigationInitiatedRef.current) {
        console.log('✅ Event matches folder, triggering navigation:', detail.folder_id);
        if (!processedFolderId && detail.folder_id) {
          setProcessedFolderId(detail.folder_id);
        }
        performNavigation(detail.folder_id);
      }
    });

    // Method 3: Polling fallback (safety net in case events are missed)
    // Improved: Check folder status, not just existence
    let stopPolling: (() => void) | null = null;
    if (targetFolderId && !navigationInitiatedRef.current && !isProcessing) {
      console.log('🔄 Starting polling fallback for folder:', targetFolderId);
      stopPolling = eventSystem.startPolling(
        targetFolderId,
        'processing_complete',
        async () => {
          // Better check: verify folder status is completed
          try {
            const folders = await folderService.getUserFolders();
            const folder = folders.find((f) => f.folder_id === targetFolderId);
            // Folder exists AND has file_count > 0 means processing likely complete
            // OR check if folder status is "completed"
            const isComplete = folder && (
              (folder.file_count !== undefined && folder.file_count > 0) ||
              folder.status === 'completed' ||
              folder.status === 'learning_complete'
            );
            
            if (isComplete) {
              console.log('✅ Polling detected folder is complete:', {
                folder_id: targetFolderId,
                file_count: folder?.file_count,
                status: folder?.status,
              });
            }
            
            return !!isComplete;
          } catch (err) {
            console.debug('Polling check error:', err);
            return false;
          }
        },
        {
          interval: 2000,
          maxAttempts: 30, // 60 seconds total
          onComplete: (detail) => {
            console.log('🔄 Polling detected completion (Method 3) for folder:', detail.folder_id);
            if (!navigationInitiatedRef.current) {
              const folderId = detail.folder_id || targetFolderId;
              if (!processedFolderId && folderId) {
                setProcessedFolderId(folderId);
              }
              performNavigation(folderId);
            }
          },
        }
      );
    }

    return () => {
      unsubscribe();
      if (stopPolling) {
        stopPolling();
      }
      // Don't clear timer on cleanup - let navigation complete
    };
  }, [status?.type, status?.folder_id, processedFolderId, isProcessing, navigate]);

  // Reset navigation flag when a new folder is processed or component unmounts
  useEffect(() => {
    // Reset when processing starts for a NEW folder
    if (status?.type === 'processing_started') {
      const currentFolderId = status?.folder_id;
      // Only reset if it's a different folder
      if (currentFolderId && currentFolderId !== processedFolderId) {
        console.log('🔄 Resetting navigation for new folder:', currentFolderId);
        navigationInitiatedRef.current = false;
        setShowChatAvailable(false);
        if (navigationTimerRef.current) {
          clearTimeout(navigationTimerRef.current);
          navigationTimerRef.current = null;
        }
      }
    }
    
    // Cleanup on unmount
    return () => {
      if (navigationTimerRef.current) {
        clearTimeout(navigationTimerRef.current);
        navigationTimerRef.current = null;
      }
    };
  }, [status?.type, status?.folder_id, processedFolderId]);


  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setValidationError(null);

    if (!folderUrl.trim()) {
      setValidationError('Please enter a folder URL');
      return;
    }

    setIsValidating(true);

    try {
      const folderId = extractFolderId(folderUrl);
      
      if (!folderId) {
        setValidationError('Invalid Google Drive folder URL. Please check the link and try again.');
        setIsValidating(false);
        return;
      }

      // Start processing and store folder ID for auto-navigation
      setProcessedFolderId(folderId);
      await processFolder(folderId);
    } catch (err) {
      const errorMessage = err instanceof APIException 
        ? err.message 
        : err instanceof Error 
        ? err.message 
        : 'Failed to process folder';
      setValidationError(errorMessage);
      setIsValidating(false);
    }
  };

  const handlePaste = async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text && isValidGoogleDriveUrl(text)) {
        setFolderUrl(text);
        setValidationError(null);
      }
    } catch (err) {
      console.error('Failed to read clipboard:', err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950 flex items-center justify-center p-4 sm:p-6">
      <div className="max-w-2xl w-full space-y-8 animate-fade-in">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center">
            <div className="w-16 h-16 bg-gray-700 rounded-2xl flex items-center justify-center shadow-lg">
              <svg
                className="w-9 h-9 text-gray-300"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.5}
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3.75 9.776c.112-.017.227-.026.344-.026h15.812c.117 0 .232.009.344.026m-16.5 0a2.25 2.25 0 00-1.883 2.542l.857 6a2.25 2.25 0 002.227 1.932H19.05a2.25 2.25 0 002.227-1.932l.857-6a2.25 2.25 0 00-1.883-2.542m-16.5 0V6A2.25 2.25 0 016 3.75h3.879a1.5 1.5 0 011.06.44l2.122 2.12a1.5 1.5 0 001.06.44H18A2.25 2.25 0 0120.25 9v.776"
                />
              </svg>
            </div>
          </div>
          <h1 className="text-3xl font-bold text-gray-100">
            Paste your Google Drive folder link
          </h1>
          <p className="text-gray-400">
            Copy any folder link from Google Drive to start chatting with your files
          </p>
        </div>

        {/* Input Form Card */}
        <div className="bg-gray-800/50 backdrop-blur-sm border-2 border-gray-700 rounded-lg p-6 sm:p-8 shadow-xl">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* URL Input */}
            <div className="space-y-2">
              <label
                htmlFor="folder-url"
                className="text-sm font-medium text-gray-300 block"
              >
                Folder Link
              </label>
              <div className="relative">
                <div className="absolute left-3 top-1/2 transform -translate-y-1/2">
                  <svg
                    className="w-5 h-5 text-gray-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
                    />
                  </svg>
                </div>
                <input
                  id="folder-url"
                  type="text"
                  value={folderUrl}
                  onChange={(e) => setFolderUrl(e.target.value)}
                  placeholder="https://drive.google.com/drive/folders/..."
                  className={`
                    w-full pl-10 pr-12 py-3
                    bg-gray-900 border-2 rounded-lg
                    text-gray-100 placeholder:text-gray-500
                    focus:outline-none focus:ring-2 focus:ring-gray-600 focus:border-transparent
                    transition-all
                    ${validationError ? 'border-red-500' : 'border-gray-600'}
                    ${isProcessing || isValidating ? 'opacity-50 cursor-not-allowed' : ''}
                  `}
                  disabled={isProcessing || isValidating}
                />
                <button
                  type="button"
                  onClick={handlePaste}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-sm text-gray-300 hover:text-gray-200 transition-colors"
                  disabled={isProcessing || isValidating}
                >
                  Paste
                </button>
              </div>

              {/* Validation Error */}
              {validationError && (
                <div className="flex items-center gap-2 text-sm text-red-400">
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <span>{validationError}</span>
                </div>
              )}

              {/* Success Indicator */}
              {folderUrl && !validationError && isValidGoogleDriveUrl(folderUrl) && (
                <div className="flex items-center gap-2 text-sm text-green-400">
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                  <span>Valid Google Drive folder URL</span>
                </div>
              )}
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isProcessing || isValidating || !!validationError || !folderUrl.trim()}
              className={`
                w-full py-3 px-6 rounded-lg font-medium text-base
                bg-gray-700 hover:bg-gray-600
                text-white
                transition-all duration-200
                disabled:opacity-50 disabled:cursor-not-allowed
                disabled:hover:bg-gray-700
                flex items-center justify-center gap-2
                shadow-lg hover:shadow-xl
              `}
            >
              {isProcessing || isValidating ? (
                <>
                  <svg
                    className="animate-spin h-5 w-5"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <span>Start Chatting</span>
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M13 7l5 5m0 0l-5 5m5-5H6"
                    />
                  </svg>
                </>
              )}
            </button>
          </form>

          {/* Example URL */}
          <div className="mt-6 p-4 bg-gray-800/50 rounded-lg border border-gray-700">
            <p className="text-sm text-gray-400 mb-2">
              <span className="font-medium text-gray-200">Example:</span>
            </p>
            <p className="text-sm text-gray-500 font-mono break-all">
              https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i0j...
            </p>
          </div>
        </div>

        {/* Processing Status - Floating Google Drive-like Interface */}
        {/* Only show while processing, hide once navigation is initiated */}
        {(isProcessing || status) && status && !navigationInitiatedRef.current && (
          <FolderUploadInterface status={status} error={error} />
        )}

        {/* Chat Available Message */}
        {showChatAvailable && !isProcessing && (
          <div className="bg-green-900/30 border border-green-700 rounded-lg p-4 animate-fade-in">
            <div className="flex items-center gap-3">
              <svg
                className="w-6 h-6 text-green-400 flex-shrink-0"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <div className="flex-1">
                <p className="text-green-300 font-medium">
                  File-specific chat is now available!
                </p>
                <p className="text-green-400/80 text-sm mt-1">
                  You can now chat with individual files. To ask general questions about the folder, use the "Summarize folder contents" button in the sidebar.
                </p>
              </div>
              <button
                onClick={() => setShowChatAvailable(false)}
                className="text-green-400 hover:text-green-300 transition-colors"
                aria-label="Dismiss"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
        )}

        {/* File Overview */}
        {files && files.length > 0 && (
          <FileOverview files={files} />
        )}
      </div>
    </div>
  );
};

export default FolderInput;
