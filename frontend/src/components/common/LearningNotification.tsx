import React from 'react';

interface LearningNotificationProps {
  isVisible: boolean;
}

/**
 * LearningNotification Component
 * 
 * Small notification bar in bottom-right corner showing that Foldex
 * is learning relationships between files for answering general questions.
 * This appears during folder summarization (background task).
 */
const LearningNotification: React.FC<LearningNotificationProps> = ({ isVisible }) => {
  if (!isVisible) return null;

  return (
    <div className="fixed bottom-6 right-6 z-50 animate-slide-up">
      <div className="bg-gray-800/95 backdrop-blur-sm border border-gray-700 rounded-lg shadow-xl px-4 py-3 flex items-center gap-3 min-w-[320px] max-w-[400px]">
        {/* Spinner */}
        <div className="flex-shrink-0">
          <svg
            className="animate-spin h-5 w-5 text-blue-400"
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

        {/* Message */}
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-200">
            Foldex is learning relationships between files
          </p>
          <p className="text-xs text-gray-400 mt-0.5">
            This enables general questions across your folder
          </p>
        </div>
      </div>
    </div>
  );
};

export default LearningNotification;
