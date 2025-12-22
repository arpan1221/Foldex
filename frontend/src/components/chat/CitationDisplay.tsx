import React, { useState, useMemo } from 'react';
import { Citation } from '../../services/types';

interface CitationDisplayProps {
  citations: Citation[];
}

/**
 * CitationDisplay Component
 *
 * Displays granular source citations with:
 * - Exact quotes from source documents
 * - Confidence scoring
 * - Contextual snippets
 * - Precise location information (page, paragraph, sentence)
 * - Interactive tooltips and detail panels
 */
const CitationDisplay: React.FC<CitationDisplayProps> = ({ citations }) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  if (!citations || citations.length === 0) {
    return null;
  }

  // Deduplicate and group citations by file type
  const groupedCitations = useMemo(() => {
    // First, deduplicate citations by file_id + page_number (or chunk_id)
    const citationMap = new Map<string, Citation>();
    
    citations.forEach((citation) => {
      // Create unique key: file_id + page_number + chunk_id (if available)
      const key = `${citation.file_id || citation.file_name || ''}_${citation.page_number || ''}_${citation.chunk_id || ''}`;
      
      // Keep the first occurrence (or prefer one with more metadata)
      if (!citationMap.has(key)) {
        citationMap.set(key, citation);
      } else {
        // If we already have this citation, keep the one with more complete data
        const existing = citationMap.get(key)!;
        const existingScore = (existing.exact_quote ? 1 : 0) + (existing.content_preview ? 1 : 0) + (existing.google_drive_url ? 1 : 0);
        const currentScore = (citation.exact_quote ? 1 : 0) + (citation.content_preview ? 1 : 0) + (citation.google_drive_url ? 1 : 0);
        if (currentScore > existingScore) {
          citationMap.set(key, citation);
        }
      }
    });
    
    // Group deduplicated citations by file type
    const groups: Record<string, Citation[]> = {};
    
    Array.from(citationMap.values()).forEach((citation) => {
      // Determine file type from mime_type or metadata
      const fileType = citation.metadata?.file_type || 
                     (citation.mime_type?.includes('audio') ? 'audio' :
                      citation.mime_type?.includes('video') ? 'audio' :
                      citation.mime_type?.includes('pdf') ? 'unstructured_native' :
                      citation.mime_type?.includes('text') && (
                        citation.mime_type.includes('javascript') || 
                        citation.mime_type.includes('python') ||
                        citation.mime_type.includes('java')
                      ) ? 'code' : 'unstructured_native');
      
      if (!groups[fileType]) {
        groups[fileType] = [];
      }
      groups[fileType].push(citation);
    });
    
    return groups;
  }, [citations]);

  const formatTimestamp = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleCitationClick = (_citation: Citation, index: number) => {
    // Toggle expanded detail panel
    if (expandedIndex === index) {
      setExpandedIndex(null);
    } else {
      setExpandedIndex(index);
    }
  };

  const openInDrive = (url: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (url) {
      window.open(url, '_blank', 'noopener,noreferrer');
    }
  };

  const formatLocation = (citation: Citation): string => {
    const parts = [];

    if (citation.location) {
      if (citation.location.page) parts.push(`p.${citation.location.page}`);
      if (citation.location.paragraph !== undefined) parts.push(`¶${citation.location.paragraph + 1}`);
      if (citation.location.timestamp) parts.push(citation.location.timestamp);
    } else {
      if (citation.page_number) parts.push(`p.${citation.page_number}`);
    }

    return parts.length > 0 ? parts.join(', ') : '';
  };


  return (
    <div className="space-y-3 ml-1">
      <div className="text-xs text-gray-500 font-medium mb-2">Sources:</div>
      
      {/* Group citations by file type */}
      {Object.entries(groupedCitations).map(([fileType, typeCitations]) => (
        <div key={fileType} className="space-y-2">
          {/* File type header */}
          <div className="text-xs text-gray-600 font-medium capitalize">
            {fileType === 'unstructured_native' ? 'Documents' :
             fileType === 'audio' ? 'Audio' :
             fileType === 'code' ? 'Code Files' :
             fileType === 'notebook' ? 'Notebooks' : fileType}
          </div>
          
          {/* Citations for this file type */}
          <div className="flex flex-wrap items-center gap-2">
            {typeCitations.map((citation, index) => {
          const location = formatLocation(citation);
          const hasGranularData = citation.exact_quote || citation.context;

          return (
            <div key={index} className="relative">
              {/* Citation Badge */}
              <button
                onClick={() => handleCitationClick(citation, index)}
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
                    expandedIndex === index
                      ? 'bg-foldex-primary-900/70 border-foldex-primary-600 text-foldex-primary-200'
                      : hoveredIndex === index
                      ? 'bg-foldex-primary-950/50 border-foldex-primary-700/50 text-foldex-primary-300'
                      : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:border-gray-600'
                  }
                  cursor-pointer
                `}
              >
                {/* Citation number badge */}
                {citation.citation_number && (
                  <span className="flex items-center justify-center w-4 h-4 rounded-full bg-foldex-primary-900/50 text-foldex-primary-300 text-[10px] font-bold">
                    {citation.citation_number}
                  </span>
                )}

                {/* Use FileBadge for file display - as div to avoid button nesting */}
                <div
                  onClick={(e) => {
                    e.stopPropagation();
                    if (citation.google_drive_url) {
                      window.open(citation.google_drive_url, '_blank', 'noopener,noreferrer');
                    }
                  }}
                  className={`
                    inline-flex items-center gap-2
                    px-2 py-0.5
                    text-xs
                    ${citation.google_drive_url ? 'cursor-pointer hover:opacity-80' : ''}
                  `}
                >
                  <span className="text-base">
                    {citation.mime_type?.includes('pdf') ? '📄' :
                     citation.mime_type?.includes('word') || citation.mime_type?.includes('document') ? '📝' :
                     citation.mime_type?.includes('audio') ? '🎵' :
                     citation.mime_type?.includes('image') ? '🖼️' : '📄'}
                  </span>
                  <span className="truncate max-w-[150px]">{citation.file_name}</span>
                  {citation.google_drive_url && (
                    <svg 
                      className="w-3 h-3 text-gray-500 flex-shrink-0" 
                      fill="none" 
                      stroke="currentColor" 
                      viewBox="0 0 24 24"
                    >
                      <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        strokeWidth={2} 
                        d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" 
                      />
                    </svg>
                  )}
                </div>

                {/* Show location-specific info */}
                {citation.page_number && (
                  <span className="text-xs text-gray-500">
                    Page {citation.page_number}
                  </span>
                )}
                
                {/* Show timestamp only for audio files */}
                {(citation.start_time !== undefined || citation.end_time !== undefined) && 
                 (citation.mime_type?.includes('audio') || citation.mime_type?.includes('video')) && (
                  <span
                    onClick={(e) => {
                      e.stopPropagation();
                      // Could trigger audio playback at timestamp
                      const start = citation.start_time || 0;
                      const end = citation.end_time || start + 10;
                      console.log(`Play audio from ${formatTimestamp(start)} to ${formatTimestamp(end)}`);
                    }}
                    className="text-xs text-foldex-primary-400 hover:text-foldex-primary-300 hover:underline cursor-pointer"
                    title={`Jump to ${formatTimestamp(citation.start_time || 0)}`}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        e.stopPropagation();
                        const start = citation.start_time || 0;
                        const end = citation.end_time || start + 10;
                        console.log(`Play audio from ${formatTimestamp(start)} to ${formatTimestamp(end)}`);
                      }
                    }}
                  >
                    {formatTimestamp(citation.start_time || 0)}
                    {citation.end_time && citation.end_time !== citation.start_time && (
                      <>-{formatTimestamp(citation.end_time)}</>
                    )}
                  </span>
                )}

                {/* Expand icon */}
                {hasGranularData && (
                  <svg
                    className={`w-3 h-3 transition-transform ${expandedIndex === index ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                )}
              </button>

              {/* Hover Tooltip */}
              {hoveredIndex === index && hasGranularData && expandedIndex !== index && (
                <div className="absolute z-50 bottom-full left-0 mb-2 min-w-fit max-w-md p-3 bg-gray-900 border border-gray-700 rounded-lg shadow-xl">
                  <div className="space-y-2">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-medium text-gray-300 break-words">
                          {citation.file_name}
                        </div>
                        {location && (
                          <div className="text-[10px] text-gray-500 mt-0.5">
                            {location}
                          </div>
                        )}
                      </div>
                    </div>

                    {citation.exact_quote && (
                      <div className="text-xs text-gray-400 italic border-l-2 border-foldex-primary-700 pl-2">
                        "{citation.exact_quote.substring(0, 150)}
                        {citation.exact_quote.length > 150 ? '...' : ''}"
                      </div>
                    )}

                    <div className="text-[10px] text-gray-500 pt-1 border-t border-gray-800">
                      Click for full context
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
          </div>
        </div>
      ))}

      {/* Expanded Detail Panel */}
      {expandedIndex !== null && citations[expandedIndex] && (
        <div className="bg-gray-900/50 border border-gray-700 rounded-lg p-4 space-y-3 animate-in fade-in slide-in-from-top-2 duration-200">
          {(() => {
            const citation = citations[expandedIndex];

            return (
              <>
                {/* Header */}
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 space-y-1">
                    <div className="flex items-center gap-2">
                      {citation.citation_number && (
                        <span className="flex items-center justify-center w-5 h-5 rounded-full bg-foldex-primary-900/50 text-foldex-primary-300 text-xs font-bold">
                          {citation.citation_number}
                        </span>
                      )}
                      <h4 className="text-sm font-medium text-gray-200">
                        {citation.file_name}
                      </h4>
                    </div>

                    <div className="flex items-center gap-3 text-xs text-gray-400">
                      {formatLocation(citation) && (
                        <span>{formatLocation(citation)}</span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {citation.google_drive_url && (
                      <button
                        onClick={(e) => openInDrive(citation.google_drive_url!, e)}
                        className="px-2 py-1 text-xs bg-foldex-primary-900/50 hover:bg-foldex-primary-900/70 text-foldex-primary-300 rounded border border-foldex-primary-700/50 transition-colors"
                      >
                        Open in Drive
                      </button>
                    )}
                    <button
                      onClick={() => setExpandedIndex(null)}
                      className="p-1 hover:bg-gray-800 rounded transition-colors"
                    >
                      <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                </div>

                {/* Context Display */}
                {citation.context ? (
                  <div className="space-y-2">
                    <div className="text-xs font-medium text-gray-400">Context:</div>
                    <div className="text-sm leading-relaxed">
                      {citation.context.before && (
                        <span className="text-gray-500">
                          ...{citation.context.before}
                        </span>
                      )}
                      <span className="text-gray-200 bg-foldex-primary-900/30 px-1 py-0.5 rounded">
                        {citation.context.quote}
                      </span>
                      {citation.context.after && (
                        <span className="text-gray-500">
                          {citation.context.after}...
                        </span>
                      )}
                    </div>
                  </div>
                ) : citation.exact_quote ? (
                  <div className="space-y-2">
                    <div className="text-xs font-medium text-gray-400">Exact Quote:</div>
                    <div className="text-sm text-gray-200 bg-foldex-primary-900/30 px-3 py-2 rounded border-l-2 border-foldex-primary-700 italic">
                      "{citation.exact_quote}"
                    </div>
                  </div>
                ) : citation.content_preview ? (
                  <div className="space-y-2">
                    <div className="text-xs font-medium text-gray-400">Preview:</div>
                    <div className="text-sm text-gray-400">
                      {citation.content_preview}
                    </div>
                  </div>
                ) : null}

                {/* Claim Text */}
                {citation.claim_text && (
                  <div className="space-y-1 pt-2 border-t border-gray-800">
                    <div className="text-xs font-medium text-gray-400">Supports claim:</div>
                    <div className="text-xs text-gray-300 italic">
                      "{citation.claim_text}"
                    </div>
                  </div>
                )}
              </>
            );
          })()}
        </div>
      )}
    </div>
  );
};

export default CitationDisplay;
