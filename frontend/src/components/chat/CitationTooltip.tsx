import React from 'react';
import { Citation } from '../../services/types';

interface CitationTooltipProps {
  citation: Citation;
  children: React.ReactNode;
}

/**
 * CitationTooltip Component
 * 
 * Displays a hoverable tooltip showing the exact chunk content
 * that was used for response generation.
 */
const CitationTooltip: React.FC<CitationTooltipProps> = ({ citation, children }) => {
  const [isHovered, setIsHovered] = React.useState(false);
  const tooltipRef = React.useRef<HTMLDivElement>(null);

  // Get the content to display in tooltip
  const tooltipContent = citation.chunk_content || 
                        citation.exact_quote || 
                        citation.content_preview || 
                        'No content available';

  // Format location info
  const locationInfo = citation.page_display || 
                       (citation.page_number ? `p.${citation.page_number}` : '') ||
                       (citation.start_time !== undefined ? `${Math.floor(citation.start_time / 60)}:${Math.floor(citation.start_time % 60).toString().padStart(2, '0')}` : '');

  return (
    <span 
      className="relative inline-block"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {children}
      
      {/* Tooltip */}
      {isHovered && tooltipContent && (
        <div
          ref={tooltipRef}
          className="absolute z-50 bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-96 max-w-[90vw] p-4 bg-gray-900 border border-gray-700 rounded-lg shadow-xl pointer-events-none"
          style={{ 
            maxHeight: '400px',
            overflowY: 'auto'
          }}
        >
          {/* Header */}
          <div className="flex items-start justify-between gap-2 mb-2 pb-2 border-b border-gray-800">
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-gray-200 truncate">
                {citation.file_name}
              </div>
              {locationInfo && (
                <div className="text-xs text-gray-500 mt-0.5">
                  {locationInfo}
                </div>
              )}
            </div>
            {citation.citation_number && (
              <span className="flex items-center justify-center w-5 h-5 rounded-full bg-foldex-primary-900/50 text-foldex-primary-300 text-xs font-bold flex-shrink-0">
                {citation.citation_number}
              </span>
            )}
          </div>

          {/* Content */}
          <div className="space-y-2">
            <div className="text-xs font-medium text-gray-400 uppercase tracking-wide">
              Source Content:
            </div>
            <div className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap break-words">
              {tooltipContent}
            </div>
          </div>

          {/* Footer hint */}
          <div className="mt-3 pt-2 border-t border-gray-800 text-[10px] text-gray-500 text-center">
            {citation.google_drive_url ? 'Click citation to open in Google Drive' : 'Hover to view source'}
          </div>
        </div>
      )}
    </span>
  );
};

export default CitationTooltip;

