import React from 'react';
import { APIException } from '../../services/api';

interface ErrorDisplayProps {
  error: Error | APIException | null;
  title?: string;
  onRetry?: () => void;
  onDismiss?: () => void;
  className?: string;
}

/**
 * ErrorDisplay Component
 * 
 * Displays error messages with retry and dismiss options.
 * Handles both regular errors and API exceptions.
 */
const ErrorDisplay: React.FC<ErrorDisplayProps> = ({
  error,
  title = 'Error',
  onRetry,
  onDismiss,
  className = '',
}) => {
  if (!error) {
    return null;
  }

  const errorMessage = error instanceof APIException 
    ? error.message 
    : error.message || 'An unexpected error occurred';

  const statusCode = error instanceof APIException ? error.statusCode : undefined;
  const errorType = error instanceof APIException ? error.errorType : undefined;

  return (
    <div
      className={`
        bg-red-950/30 border border-red-800/50 rounded-lg p-4
        ${className}
      `}
    >
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0">
          <svg
            className="w-5 h-5 text-red-400"
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
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-medium text-red-400 mb-1">{title}</h3>
          <p className="text-sm text-red-300">{errorMessage}</p>
          {statusCode && (
            <p className="text-xs text-red-400/70 mt-1">
              Status: {statusCode} {errorType && `(${errorType})`}
            </p>
          )}
        </div>
        {(onRetry || onDismiss) && (
          <div className="flex-shrink-0 flex gap-2">
            {onRetry && (
              <button
                onClick={onRetry}
                className="text-xs text-red-300 hover:text-red-200 underline"
              >
                Retry
              </button>
            )}
            {onDismiss && (
              <button
                onClick={onDismiss}
                className="text-xs text-red-300 hover:text-red-200"
                aria-label="Dismiss error"
              >
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
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ErrorDisplay;

