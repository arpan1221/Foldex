import React, { useState, useEffect } from 'react';
import { FolderSummary } from '../../services/types';
import { folderService } from '../../services/api';
import { eventSystem } from '../../utils/eventSystem';

interface FolderSummaryDisplayProps {
  folderId: string;
}

/**
 * FolderSummaryDisplay Component
 * 
 * Displays learned folder context in a collapsible panel.
 * Shows folder overview, file type distribution, capabilities, themes, and graph statistics.
 */
const FolderSummaryDisplay: React.FC<FolderSummaryDisplayProps> = ({ folderId }) => {
  const [summary, setSummary] = useState<FolderSummary | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const expectingSummaryRef = React.useRef(false); // Track if we're expecting a summary soon
  const pollingIntervalRef = React.useRef<ReturnType<typeof setInterval> | null>(null);

  // Define loadSummary outside useEffect so we can call it from event handlers
  const loadSummary = React.useCallback(async (autoExpand: boolean = false) => {
    if (!folderId) return;

    setIsLoading(true);
    setError(null);
    try {
      const summaryData = await folderService.getFolderSummary(folderId);
      setSummary(summaryData);
      // Auto-expand if summary just became available
      if (autoExpand && summaryData.learning_status === 'learning_complete') {
        console.log('Auto-expanding folder knowledge panel after summary completion');
        setIsExpanded(true);
        expectingSummaryRef.current = false; // Summary arrived, stop expecting
        // Stop polling once we have a complete summary
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
      }
      // Also check if summary is now complete (even if not auto-expanding)
      if (summaryData.learning_status === 'learning_complete') {
        expectingSummaryRef.current = false;
        // Stop polling once we have a complete summary
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load folder summary');
      console.error('Error loading folder summary:', err);
      // Don't clear expectingSummaryRef on error - keep polling
    } finally {
      setIsLoading(false);
    }
  }, [folderId]);

  useEffect(() => {
    loadSummary();

    // Start polling to check for summary availability
    // This ensures the component refreshes even if WebSocket events are missed
    const startPolling = () => {
      // Clear any existing polling interval
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
      
      console.log('🔄 Starting polling for folder summary:', folderId);
      pollingIntervalRef.current = setInterval(async () => {
        try {
          const summaryData = await folderService.getFolderSummary(folderId);
          // If summary just became available, update and stop polling
          if (summaryData.learning_status === 'learning_complete') {
            console.log('✅ Polling detected summary is complete:', folderId);
            setSummary(summaryData);
            setIsExpanded(true);
            expectingSummaryRef.current = false;
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current);
              pollingIntervalRef.current = null;
            }
          }
        } catch (err) {
          // Ignore polling errors, will retry on next interval
          console.debug('Polling check error:', err);
        }
      }, 2000); // Poll every 2 seconds
    };

    // Listen for summary_complete event to reload when summary becomes available
    // Use robust event system for reliability
    const unsubscribe = eventSystem.on('summary_complete', (detail) => {
      if (detail.folder_id === folderId) {
        console.log('✅ Summary complete event received, reloading and expanding summary for folder:', folderId);
        expectingSummaryRef.current = true; // Mark that we're expecting a summary
        startPolling(); // Start polling as fallback
        
        // Also try loading with progressive retries
        const attemptLoad = (attempt: number = 0) => {
          setTimeout(async () => {
            try {
              await loadSummary(true); // Auto-expand when summary completes
            } catch (err) {
              // Retry if we haven't exhausted attempts
              if (attempt < 4) {
                attemptLoad(attempt + 1);
              }
            }
          }, 1000 + (attempt * 500)); // Progressive delay: 1s, 1.5s, 2s, 2.5s
        };
        attemptLoad(0);
      }
    });

    // Also listen to window events for backward compatibility
    const handleSummaryComplete = (event: Event) => {
      const customEvent = event as CustomEvent;
      if (customEvent.detail?.folder_id === folderId) {
        console.log('✅ Summary complete window event received, reloading and expanding summary for folder:', folderId);
        expectingSummaryRef.current = true;
        startPolling();
        
        // Same retry logic as above
        const attemptLoad = (attempt: number = 0) => {
          setTimeout(async () => {
            try {
              await loadSummary(true);
            } catch (err) {
              if (attempt < 4) {
                attemptLoad(attempt + 1);
              }
            }
          }, 1000 + (attempt * 500));
        };
        attemptLoad(0);
      }
    };

    window.addEventListener('summary_complete', handleSummaryComplete);

    // Start polling initially - it will stop itself once a complete summary is found
    // This ensures we catch the summary even if WebSocket events are missed
    startPolling();

    return () => {
      unsubscribe();
      window.removeEventListener('summary_complete', handleSummaryComplete);
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [folderId, loadSummary]);

  // Only render if learning is complete (i.e., summarization has been done)
  // This ensures the Folder Knowledge tab only appears after user clicks "Summarize folder contents"
  // However, if we're expecting a summary (after summary_complete event) or loading, show a loading state
  if (!summary || summary.learning_status !== 'learning_complete') {
    // Show loading state if we're expecting a summary or actively loading
    if (isLoading || expectingSummaryRef.current) {
      return (
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg mb-6 p-4">
          <div className="flex items-center gap-2 text-gray-400">
            <div className="inline-block animate-spin rounded-full h-4 w-4 border-b-2 border-gray-400"></div>
            <span className="text-sm">Loading folder knowledge...</span>
          </div>
        </div>
      );
    }
    return null;
  }

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-lg mb-6">
      {/* Header - Always Visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-gray-800/70 transition-colors"
      >
        <div className="flex items-center gap-3">
          <svg
            className={`w-5 h-5 text-gray-400 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5l7 7-7 7"
            />
          </svg>
          <div className="flex items-center gap-2">
            <svg
              className="w-5 h-5 text-gray-300"
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
            <h3 className="text-sm font-semibold text-gray-300">Folder Knowledge</h3>
          </div>
        </div>
        {summary.learning_status === 'learning_complete' && (
          <span className="px-2 py-1 text-xs bg-green-500/20 text-green-400 rounded">
            Learned
          </span>
        )}
      </button>

      {/* Content - Collapsible */}
      {isExpanded && (
        <div className="px-4 pt-4 pb-6 space-y-4 border-t border-gray-700 max-h-[60vh] overflow-y-auto custom-scrollbar">
          {/* Loading State */}
          {isLoading && (
            <div className="py-4 text-center text-gray-400">
              <div className="inline-block animate-spin rounded-full h-5 w-5 border-b-2 border-gray-400"></div>
              <p className="mt-2 text-sm">Loading folder knowledge...</p>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="py-4 p-3 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-sm">
              {error}
            </div>
          )}

          {/* Summary Content */}
          {summary && !isLoading && !error && (
            <>
              {/* Folder Overview */}
              {summary.summary && (
                <div>
                  <h4 className="text-sm font-semibold text-gray-300 mb-2">Overview</h4>
                  <p className="text-sm text-gray-400 leading-relaxed">{summary.summary}</p>
                </div>
              )}

              {/* File Type Distribution */}
              {summary.file_type_distribution && Object.keys(summary.file_type_distribution).length > 0 && (
                <div>
                  <h4 className="text-sm font-semibold text-gray-300 mb-2">File Types</h4>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(summary.file_type_distribution).map(([type, count]) => (
                      <span
                        key={type}
                        className="px-3 py-1 text-xs bg-gray-700 text-gray-300 rounded-full"
                      >
                        {type}: {count}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Insights */}
              {summary.insights && (
                <div>
                  <h4 className="text-sm font-semibold text-gray-300 mb-2">Insights</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-500">Total Files</p>
                      <p className="text-gray-200 font-semibold">{summary.insights.total_files}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">File Types</p>
                      <p className="text-gray-200 font-semibold">{summary.insights.unique_file_types}</p>
                    </div>
                    {summary.insights.key_relationships > 0 && (
                      <div className="col-span-2">
                        <p className="text-gray-500">Key Relationships</p>
                        <p className="text-gray-200 font-semibold">{summary.insights.key_relationships}</p>
                      </div>
                    )}
                  </div>
                  {summary.insights.top_themes && summary.insights.top_themes.length > 0 && (
                    <div className="mt-3">
                      <p className="text-gray-500 text-sm mb-2">Main Themes</p>
                      <div className="flex flex-wrap gap-2">
                        {summary.insights.top_themes.map((theme, idx) => (
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
              )}

              {/* Capabilities */}
              {summary.capabilities && summary.capabilities.length > 0 && (
                <div>
                  <h4 className="text-sm font-semibold text-gray-300 mb-2">What You Can Ask</h4>
                  <ul className="space-y-1">
                    {summary.capabilities.map((capability, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-sm text-gray-400">
                        <svg
                          className="w-4 h-4 text-gray-500 mt-0.5 flex-shrink-0"
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
                        <span>{capability}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Learning Completed At */}
              {summary.learning_completed_at && (
                <div className="pt-2 border-t border-gray-700">
                  <p className="text-xs text-gray-500">
                    Learned on {new Date(summary.learning_completed_at).toLocaleString()}
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default FolderSummaryDisplay;

