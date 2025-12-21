import React from 'react';
import { ProcessingStatus } from '../../services/types';

interface LearningProgressProps {
  status: ProcessingStatus;
}

/**
 * LearningProgress Component
 * 
 * Displays folder learning/summarization progress with visual feedback.
 * Shows progress bar, learning stage text, and summary statistics when complete.
 */
const LearningProgress: React.FC<LearningProgressProps> = ({ status }) => {
  const getLearningStageText = () => {
    switch (status.type) {
      case 'learning_started':
        return 'Starting to learn your folder...';
      case 'generating_summary':
        return 'Analyzing folder contents...';
      case 'summary_progress':
        return status.message || 'Generating summary...';
      case 'summary_complete':
        return 'Folder learning complete!';
      case 'summary_error':
        return 'Learning failed';
      default:
        return 'Learning folder...';
    }
  };

  const getProgressPercentage = (): number => {
    if (status.type === 'summary_complete') return 100;
    if (status.progress !== undefined) return Math.round(status.progress * 100);
    
    // Default progress based on stage
    switch (status.type) {
      case 'learning_started':
        return 0;
      case 'generating_summary':
        return 20;
      case 'summary_progress':
        return status.progress ? Math.round(status.progress * 100) : 50;
      default:
        return 0;
    }
  };

  const progressPercentage = getProgressPercentage();
  const isComplete = status.type === 'summary_complete';
  const isError = status.type === 'summary_error';

  return (
    <div className="bg-gray-800/50 border-2 border-gray-700 rounded-lg p-6 shadow-xl animate-fade-in">
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        {!isComplete && !isError && (
          <div className="relative">
            <svg
              className="animate-spin h-6 w-6 text-gray-300"
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
        )}
        {isComplete && (
          <svg
            className="h-6 w-6 text-green-400"
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
        )}
        {isError && (
          <svg
            className="h-6 w-6 text-red-400"
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
          <h3 className={`text-lg font-semibold ${isComplete ? 'text-green-400' : isError ? 'text-red-400' : 'text-gray-300'}`}>
            {isComplete ? 'Learning Complete' : isError ? 'Learning Failed' : 'Learning Your Folder'}
          </h3>
          <p className="text-sm text-gray-400 mt-1">
            {getLearningStageText()}
          </p>
        </div>
      </div>

      {/* Progress Bar */}
      {!isError && (
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-300">Progress</span>
            <span className="text-sm font-medium text-gray-200">
              {progressPercentage}%
            </span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-300 ease-out ${
                isComplete ? 'bg-green-500' : 'bg-gray-500'
              }`}
              style={{ width: `${progressPercentage}%` }}
            />
          </div>
        </div>
      )}

      {/* Error Display */}
      {isError && status.error && (
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
                {status.error}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Summary Statistics (when complete) */}
      {isComplete && status.summary_data && (
        <div className="mt-4 p-3 bg-gray-900/50 border border-gray-700 rounded-lg">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wide">Files Analyzed</p>
              <p className="text-lg font-semibold text-gray-200">
                {status.summary_data.total_files || 0}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wide">File Types</p>
              <p className="text-lg font-semibold text-gray-200">
                {status.summary_data.unique_file_types || 0}
              </p>
            </div>
            {status.summary_data.top_themes && status.summary_data.top_themes.length > 0 && (
              <div className="col-span-2">
                <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Key Themes</p>
                <div className="flex flex-wrap gap-2">
                  {status.summary_data.top_themes.slice(0, 5).map((theme: string, idx: number) => (
                    <span
                      key={idx}
                      className="px-2 py-1 text-xs bg-gray-700 text-gray-300 rounded"
                    >
                      {theme}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default LearningProgress;

