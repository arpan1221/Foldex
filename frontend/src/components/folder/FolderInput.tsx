import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useFolderProcessor } from '../../hooks/useFolderProcessor';
import ProcessingStatus from './ProcessingStatus';
import FileOverview from './FileOverview';
import { isValidGoogleDriveUrl, extractFolderId } from '../../utils/validators';

/**
 * FolderInput Component
 * 
 * Google Drive folder URL input with validation and processing initiation.
 * Follows Figma wireframe design with dark gradient background.
 */
const FolderInput: React.FC = () => {
  const [folderUrl, setFolderUrl] = useState('');
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const { processFolder, isProcessing, status, error, files } = useFolderProcessor();
  const navigate = useNavigate();

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

      // Start processing
      await processFolder(folderId);
      
      // Navigate to chat after processing starts
      // (Processing will continue in background)
      setTimeout(() => {
        navigate(`/chat/${folderId}`);
      }, 1000);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to process folder';
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
            <div className="w-16 h-16 bg-gradient-to-br from-foldex-primary-500 to-foldex-accent-500 rounded-2xl flex items-center justify-center shadow-lg">
              <svg
                className="w-9 h-9 text-white"
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
                    focus:outline-none focus:ring-2 focus:ring-foldex-primary-500 focus:border-transparent
                    transition-all
                    ${validationError ? 'border-red-500' : 'border-gray-600'}
                    ${isProcessing || isValidating ? 'opacity-50 cursor-not-allowed' : ''}
                  `}
                  disabled={isProcessing || isValidating}
                />
                <button
                  type="button"
                  onClick={handlePaste}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-sm text-foldex-primary-400 hover:text-foldex-primary-300 transition-colors"
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
                bg-gradient-to-r from-foldex-primary-600 to-foldex-accent-600
                hover:from-foldex-primary-700 hover:to-foldex-accent-700
                text-white
                transition-all duration-200
                disabled:opacity-50 disabled:cursor-not-allowed
                disabled:hover:from-foldex-primary-600 disabled:hover:to-foldex-accent-600
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
          <div className="mt-6 p-4 bg-foldex-primary-950/30 rounded-lg border border-foldex-primary-800/50">
            <p className="text-sm text-gray-400 mb-2">
              <span className="font-medium text-gray-200">Example:</span>
            </p>
            <p className="text-sm text-gray-500 font-mono break-all">
              https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i0j...
            </p>
          </div>
        </div>

        {/* Processing Status */}
        {isProcessing && status && (
          <ProcessingStatus status={status} error={error} />
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
