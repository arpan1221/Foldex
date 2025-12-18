import React, { useState } from 'react';
import { Citation } from '../../services/types';
import { formatTimestamp } from '../../utils/formatters';

interface CitationDisplayProps {
  citations: Citation[];
}

/**
 * CitationDisplay Component
 * 
 * Displays source citations with file references, page numbers, and timestamps.
 * Includes hover effects and clickable file links following Figma design.
 */
const CitationDisplay: React.FC<CitationDisplayProps> = ({ citations }) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  if (!citations || citations.length === 0) {
    return null;
  }

  const getFileIcon = (mimeType?: string) => {
    // Determine icon based on file type
    if (!mimeType) {
      return (
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      );
    }

    if (mimeType.includes('pdf')) {
      return (
        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
          <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
        </svg>
      );
    }

    return (
      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    );
  };

  const handleCitationClick = (citation: Citation) => {
    // TODO: Navigate to file or show file preview
    console.log('Citation clicked:', citation);
  };

  return (
    <div className="flex flex-wrap items-center gap-2 ml-1">
      <span className="text-xs text-gray-500 font-medium">Sources:</span>
      {citations.map((citation, index) => (
        <button
          key={index}
          onClick={() => handleCitationClick(citation)}
          onMouseEnter={() => setHoveredIndex(index)}
          onMouseLeave={() => setHoveredIndex(null)}
          className={`
            inline-flex items-center gap-1.5
            px-2.5 py-1
            text-xs
            rounded-md
            border
            transition-all duration-200
            ${
              hoveredIndex === index
                ? 'bg-foldex-primary-950/50 border-foldex-primary-700/50 text-foldex-primary-300'
                : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:border-gray-600'
            }
          `}
        >
          {getFileIcon()}
          <span className="truncate max-w-[200px]">{citation.file_name}</span>
          {citation.page_number && (
            <span className="text-gray-500">• p.{citation.page_number}</span>
          )}
          {citation.timestamp && (
            <span className="text-gray-500">
              • {formatTimestamp(citation.timestamp)}
            </span>
          )}
          {citation.confidence && citation.confidence < 0.8 && (
            <span className="text-yellow-400" title="Lower confidence">
              ⚠
            </span>
          )}
        </button>
      ))}
    </div>
  );
};

export default CitationDisplay;
