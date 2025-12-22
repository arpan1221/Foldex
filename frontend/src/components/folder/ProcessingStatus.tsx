import React from 'react';
import { ProcessingStatus as ProcessingStatusType } from '../../services/types';
import IngestionProgress from './IngestionProgress';
import LearningProgress from './LearningProgress';

interface ProcessingStatusProps {
  status: ProcessingStatusType;
  error?: Error | null;
  onRetryFile?: (fileId: string) => void;
}

/**
 * ProcessingStatus Component
 * 
 * Real-time processing progress display with WebSocket updates.
 * Shows progress bars, file-by-file processing, and completion status.
 * Includes file type distribution and detailed ingestion progress.
 */
const ProcessingStatus: React.FC<ProcessingStatusProps> = ({ status, error, onRetryFile }) => {
  const progressPercentage = status.progress
    ? Math.round(status.progress * 100)
    : 0;

  // Determine status icon and color - using neutral dark theme
  const getStatusConfig = () => {
    switch (status.type) {
      case 'processing_started':
      case 'file_processed':
      case 'file_processing':
      case 'files_detected':
      case 'building_graph':
        return {
          icon: 'spinner',
          color: 'text-gray-300',
          bgColor: 'bg-gray-800/50',
          borderColor: 'border-gray-700',
        };
      case 'processing_complete':
      case 'graph_complete':
      case 'summary_complete':
        return {
          icon: 'complete',
          color: 'text-green-400',
          bgColor: 'bg-gray-800/50',
          borderColor: 'border-green-500/30',
        };
      case 'processing_error':
      case 'file_error':
        return {
          icon: 'error',
          color: 'text-red-400',
          bgColor: 'bg-gray-800/50',
          borderColor: 'border-red-600',
        };
      default:
        return {
          icon: 'spinner',
          color: 'text-gray-300',
          bgColor: 'bg-gray-800/50',
          borderColor: 'border-gray-700',
        };
    }
  };

  const config = getStatusConfig();

  // Don't render if no status
  // But be more lenient - if status exists, always render (even if some fields are missing)
  if (!status || !status.type) {
    return null;
  }

  return (
    <div
      className={`
        ${config.bgColor} ${config.borderColor}
        border-2 rounded-lg p-6 shadow-xl
        animate-fade-in
      `}
    >
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        {status.type === 'processing_started' || status.type === 'file_processed' ? (
          <div className="relative">
            <svg
              className={`animate-spin h-6 w-6 ${config.color}`}
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
          </div>
        ) : (status.type === 'processing_complete' || status.type === 'summary_complete') ? (
          <svg
            className={`h-6 w-6 ${config.color}`}
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
        ) : (
          <svg
            className={`h-6 w-6 ${config.color}`}
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
        )}

        <div className="flex-1">
          <h3 className={`text-lg font-semibold ${config.color}`}>
            {status.type === 'processing_started' && 'Processing Started'}
            {status.type === 'files_detected' && 'Files Detected'}
            {status.type === 'file_processing' && 'Processing Files'}
            {status.type === 'file_processed' && 'Processing Files'}
            {status.type === 'building_graph' && 'Building Knowledge Graph'}
            {status.type === 'graph_complete' && 'Graph Complete'}
            {status.type === 'processing_complete' && 'Processing Complete'}
            {status.type === 'summary_complete' && 'Folder Summarized'}
            {status.type === 'processing_error' && 'Processing Error'}
            {status.type === 'file_error' && 'File Processing Error'}
            {!['processing_started', 'files_detected', 'file_processing', 'file_processed', 'building_graph', 'graph_complete', 'processing_complete', 'summary_complete', 'processing_error', 'file_error', 'learning_started', 'generating_summary', 'summary_progress', 'summary_error'].includes(status.type) && 'Processing'}
          </h3>
          {status.message && (
            <p className="text-sm text-gray-400 mt-1">{status.message}</p>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      {(status.progress !== undefined || status.files_processed !== undefined) && (
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-300">
              {status.files_processed !== undefined && status.total_files !== undefined
                ? `File ${status.files_processed} of ${status.total_files}`
                : 'Progress'}
            </span>
            <span className="text-sm font-medium text-gray-200">
              {progressPercentage}%
            </span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
            <div
              className={`
                h-full rounded-full transition-all duration-300 ease-out
                ${
                  status.type === 'processing_complete' || status.type === 'graph_complete'
                    ? 'bg-gray-400'
                    : status.type === 'processing_error' || status.type === 'file_error'
                    ? 'bg-red-500'
                    : 'bg-gray-500'
                }
              `}
              style={{
                width: `${progressPercentage}%`,
              }}
            />
          </div>
        </div>
      )}

      {/* Current File */}
      {status.file_name && (
        <div className="flex items-center gap-2 p-3 bg-gray-900/50 rounded-lg border border-gray-700">
          <svg
            className="w-4 h-4 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <span className="text-sm text-gray-400 truncate">
            Processing: <span className="font-medium text-gray-300">{status.file_name}</span>
          </span>
        </div>
      )}

      {/* Error Display */}
      {(error || status.type === 'processing_error' || status.type === 'file_error') && (
        <div className="mt-4 p-3 bg-gray-900/50 border border-red-600 rounded-lg">
          <div className="flex items-start gap-2">
            <svg
              className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5"
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
            <div className="flex-1">
              <p className="text-sm font-medium text-red-400">Error</p>
              <p className="text-sm text-gray-400 mt-1">
                {error?.message || status.error || 'An error occurred during processing'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Ingestion Progress Details */}
      {(status.type === 'file_processing' || 
        status.type === 'file_processed' || 
        status.type === 'file_error' ||
        status.type === 'files_detected') && (
        <div className="mt-4">
          <IngestionProgress status={status} onRetry={onRetryFile} />
        </div>
      )}

      {/* Learning Progress */}
      {(status.type === 'learning_started' ||
        status.type === 'generating_summary' ||
        status.type === 'summary_progress' ||
        status.type === 'summary_complete' ||
        status.type === 'summary_error') && (
        <div className="mt-4">
          <LearningProgress status={status} />
        </div>
      )}

      {/* Success/Completion Message for Processing */}
      {(status.type === 'processing_complete' || status.type === 'graph_complete') && 
       !['learning_started', 'generating_summary', 'summary_progress', 'summary_complete', 'summary_error'].includes(status.type) && (
        <div className="mt-4 p-3 bg-gray-900/50 border border-gray-700 rounded-lg">
          {status.failed_files && status.failed_files > 0 ? (
            <div className="flex items-center gap-2">
              <svg
                className="w-5 h-5 text-yellow-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
              <p className="text-sm text-yellow-400">
                Processing complete: {status.files_processed || 0}/{status.total_files || 0} files processed, {status.failed_files} failed. You can still chat with the successfully processed files.
              </p>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <svg
                className="w-5 h-5 text-green-400"
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
              <p className="text-sm text-green-400">
                All files processed successfully! Learning folder...
              </p>
            </div>
          )}
        </div>
      )}

      {/* Success Message for Summary Completion */}
      {status.type === 'summary_complete' && (
        <div className="mt-4 p-3 bg-gray-900/50 border border-green-500/30 rounded-lg">
          <div className="flex items-center gap-2">
            <svg
              className="w-5 h-5 text-green-400"
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
            <p className="text-sm text-green-400 font-medium">
              Folder summarized! Check the Folder Knowledge tab below.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProcessingStatus;
