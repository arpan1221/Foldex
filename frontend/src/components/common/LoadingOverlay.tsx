import React from 'react';
import LoadingSpinner from './LoadingSpinner';

interface LoadingOverlayProps {
  isLoading: boolean;
  message?: string;
  fullScreen?: boolean;
}

/**
 * LoadingOverlay Component
 * 
 * Full-screen or container overlay with loading spinner.
 */
const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  isLoading,
  message,
  fullScreen = false,
}) => {
  if (!isLoading) {
    return null;
  }

  const containerClass = fullScreen
    ? 'fixed inset-0 z-50 bg-gray-900/80 backdrop-blur-sm'
    : 'absolute inset-0 z-10 bg-gray-900/50 backdrop-blur-sm';

  return (
    <div className={`${containerClass} flex items-center justify-center`}>
      <div className="text-center">
        <LoadingSpinner size="lg" />
        {message && (
          <p className="mt-4 text-gray-300 text-sm">{message}</p>
        )}
      </div>
    </div>
  );
};

export default LoadingOverlay;

