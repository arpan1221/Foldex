import React from 'react';

interface Citation {
  file_id: string;
  file_name: string;
  chunk_id: string;
  page_number?: number;
  timestamp?: number;
  confidence: number;
}

interface CitationDisplayProps {
  citations: Citation[];
}

const CitationDisplay: React.FC<CitationDisplayProps> = ({ citations }) => {
  return (
    <div className="mt-2 space-y-1">
      <p className="text-sm text-gray-600 font-semibold">Sources:</p>
      {citations.map((citation, index) => (
        <div
          key={index}
          className="text-sm text-gray-500 bg-gray-50 p-2 rounded"
        >
          <span className="font-medium">{citation.file_name}</span>
          {citation.page_number && (
            <span className="ml-2">Page {citation.page_number}</span>
          )}
          {citation.timestamp && (
            <span className="ml-2">
              {Math.floor(citation.timestamp / 60)}:{(citation.timestamp % 60).toFixed(0).padStart(2, '0')}
            </span>
          )}
        </div>
      ))}
    </div>
  );
};

export default CitationDisplay;

