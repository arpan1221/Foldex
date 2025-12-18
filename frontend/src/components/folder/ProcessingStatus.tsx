import React from 'react';

interface ProcessingStatusProps {
  status: {
    type: string;
    message?: string;
    progress?: number;
    file_name?: string;
  };
}

const ProcessingStatus: React.FC<ProcessingStatusProps> = ({ status }) => {
  return (
    <div className="mt-4 p-4 bg-blue-50 rounded-lg">
      <div className="flex items-center gap-2">
        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
        <span className="text-blue-700">{status.message || 'Processing...'}</span>
      </div>
      {status.progress !== undefined && (
        <div className="mt-2">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all"
              style={{ width: `${status.progress * 100}%` }}
            ></div>
          </div>
          <p className="text-sm text-gray-600 mt-1">
            {Math.round(status.progress * 100)}% complete
          </p>
        </div>
      )}
      {status.file_name && (
        <p className="text-sm text-gray-600 mt-2">Processing: {status.file_name}</p>
      )}
    </div>
  );
};

export default ProcessingStatus;

