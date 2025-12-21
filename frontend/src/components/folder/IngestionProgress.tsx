import React, { useState, useMemo } from 'react';
import { ProcessingStatus } from '../../services/types';
import FileBadge from '../common/FileBadge';

interface FileProcessingStatus {
  file_id: string;
  file_name: string;
  file_type?: string;
  mime_type?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'skipped';
  error?: string;
  progress?: number;
}

interface IngestionProgressProps {
  status: ProcessingStatus | null;
  onRetry?: (fileId: string) => void;
}

/**
 * IngestionProgress Component
 * 
 * Displays per-file processing status during folder ingestion.
 * Shows file type distribution and allows retry for failed files.
 */
const IngestionProgress: React.FC<IngestionProgressProps> = ({ status, onRetry }) => {
  const [expandedSection, setExpandedSection] = useState<'all' | 'failed' | 'byType' | null>(null);
  const [fileStatuses, setFileStatuses] = useState<Map<string, FileProcessingStatus>>(new Map());

  // Update file statuses from processing status
  React.useEffect(() => {
    if (!status) return;

    if (status.type === 'file_processing' && status.file_id && status.file_name) {
      setFileStatuses((prev) => {
        const updated = new Map(prev);
        updated.set(status.file_id!, {
          file_id: status.file_id,
          file_name: status.file_name!,
          file_type: status.metadata?.file_type,
          mime_type: status.metadata?.mime_type,
          status: 'processing',
          progress: 0,
        });
        return updated;
      });
    }

    if (status.type === 'file_processed' && status.file_id) {
      setFileStatuses((prev) => {
        const updated = new Map(prev);
        const existing = updated.get(status.file_id!);
        if (existing) {
          updated.set(status.file_id!, {
            ...existing,
            status: 'completed',
            progress: 1,
          });
        }
        return updated;
      });
    }

    if (status.type === 'file_error' && status.file_id) {
      setFileStatuses((prev) => {
        const updated = new Map(prev);
        const existing = updated.get(status.file_id!);
        if (existing) {
          updated.set(status.file_id!, {
            ...existing,
            status: 'failed',
            error: status.error,
          });
        }
        return updated;
      });
    }
  }, [status]);

  // Calculate file type distribution
  const fileTypeDistribution = useMemo(() => {
    const distribution: Record<string, { total: number; completed: number; failed: number; processing: number }> = {};
    
    fileStatuses.forEach((fileStatus) => {
      const type = fileStatus.file_type || fileStatus.mime_type || 'unknown';
      if (!distribution[type]) {
        distribution[type] = { total: 0, completed: 0, failed: 0, processing: 0 };
      }
      distribution[type].total++;
      if (fileStatus.status === 'completed') distribution[type].completed++;
      if (fileStatus.status === 'failed') distribution[type].failed++;
      if (fileStatus.status === 'processing') distribution[type].processing++;
    });
    
    return distribution;
  }, [fileStatuses]);

  // Get failed files
  const failedFiles = useMemo(() => {
    return Array.from(fileStatuses.values()).filter((f) => f.status === 'failed');
  }, [fileStatuses]);

  // Get processing files
  const processingFiles = useMemo(() => {
    return Array.from(fileStatuses.values()).filter((f) => f.status === 'processing');
  }, [fileStatuses]);

  if (!status || status.type === 'processing_complete') {
    return null;
  }

  const totalFiles = status.total_files || 0;
  const filesProcessed = status.files_processed || 0;
  const progress = totalFiles > 0 ? filesProcessed / totalFiles : 0;

  return (
    <div className="bg-gray-900/50 border border-gray-700 rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <h3 className="text-sm font-medium text-gray-200 mb-1">Processing Files</h3>
          <div className="flex items-center gap-2">
            <div className="flex-1 bg-gray-800 rounded-full h-2 overflow-hidden">
              <div
                className="h-full bg-foldex-primary-500 transition-all duration-300"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
            <span className="text-xs text-gray-400 whitespace-nowrap">
              {filesProcessed} / {totalFiles}
            </span>
          </div>
        </div>
      </div>

      {/* File Type Distribution */}
      {Object.keys(fileTypeDistribution).length > 0 && (
        <div className="space-y-2">
          <button
            onClick={() => setExpandedSection(expandedSection === 'byType' ? null : 'byType')}
            className="w-full flex items-center justify-between text-xs text-gray-400 hover:text-gray-300"
          >
            <span>File Type Distribution</span>
            <svg
              className={`w-4 h-4 transition-transform ${expandedSection === 'byType' ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          {expandedSection === 'byType' && (
            <div className="space-y-2 pl-2 border-l-2 border-gray-800">
              {Object.entries(fileTypeDistribution).map(([type, stats]) => (
                <div key={type} className="flex items-center justify-between text-xs">
                  <span className="text-gray-400 capitalize">{type.replace('_', ' ')}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-gray-500">
                      {stats.completed}/{stats.total}
                    </span>
                    {stats.failed > 0 && (
                      <span className="text-red-400">{stats.failed} failed</span>
                    )}
                    {stats.processing > 0 && (
                      <span className="text-yellow-400">{stats.processing} processing</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Currently Processing */}
      {processingFiles.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-gray-400 font-medium">Currently Processing:</div>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {processingFiles.map((file) => (
              <div
                key={file.file_id}
                className="flex items-center gap-2 text-xs text-gray-400 bg-gray-800/30 rounded px-2 py-1"
              >
                <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
                <FileBadge
                  fileName={file.file_name}
                  fileType={file.file_type as any}
                  mimeType={file.mime_type}
                  className="!text-xs !px-2 !py-0.5"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Failed Files */}
      {failedFiles.length > 0 && (
        <div className="space-y-2">
          <button
            onClick={() => setExpandedSection(expandedSection === 'failed' ? null : 'failed')}
            className="w-full flex items-center justify-between text-xs text-red-400 hover:text-red-300"
          >
            <span>Failed Files ({failedFiles.length})</span>
            <svg
              className={`w-4 h-4 transition-transform ${expandedSection === 'failed' ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          {expandedSection === 'failed' && (
            <div className="space-y-2 pl-2 border-l-2 border-red-900/50">
              {failedFiles.map((file) => (
                <div
                  key={file.file_id}
                  className="bg-gray-800/30 rounded p-2 space-y-1"
                >
                  <div className="flex items-center justify-between">
                    <FileBadge
                      fileName={file.file_name}
                      fileType={file.file_type as any}
                      mimeType={file.mime_type}
                      className="!text-xs !px-2 !py-0.5"
                    />
                    {onRetry && (
                      <button
                        onClick={() => onRetry(file.file_id)}
                        className="text-xs text-foldex-primary-400 hover:text-foldex-primary-300 hover:underline"
                      >
                        Retry
                      </button>
                    )}
                  </div>
                  {file.error && (
                    <div className="text-xs text-red-400 mt-1">{file.error}</div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* All Files (expandable) */}
      {fileStatuses.size > 0 && (
        <div className="space-y-2">
          <button
            onClick={() => setExpandedSection(expandedSection === 'all' ? null : 'all')}
            className="w-full flex items-center justify-between text-xs text-gray-400 hover:text-gray-300"
          >
            <span>All Files ({fileStatuses.size})</span>
            <svg
              className={`w-4 h-4 transition-transform ${expandedSection === 'all' ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          {expandedSection === 'all' && (
            <div className="space-y-1 max-h-64 overflow-y-auto pl-2 border-l-2 border-gray-800">
              {Array.from(fileStatuses.values()).map((file) => (
                <div
                  key={file.file_id}
                  className={`flex items-center gap-2 text-xs rounded px-2 py-1 ${
                    file.status === 'completed' ? 'bg-green-900/20 text-green-400' :
                    file.status === 'failed' ? 'bg-red-900/20 text-red-400' :
                    file.status === 'processing' ? 'bg-yellow-900/20 text-yellow-400' :
                    'bg-gray-800/30 text-gray-400'
                  }`}
                >
                  <div className={`w-2 h-2 rounded-full ${
                    file.status === 'completed' ? 'bg-green-400' :
                    file.status === 'failed' ? 'bg-red-400' :
                    file.status === 'processing' ? 'bg-yellow-400 animate-pulse' :
                    'bg-gray-500'
                  }`} />
                  <FileBadge
                    fileName={file.file_name}
                    fileType={file.file_type as any}
                    mimeType={file.mime_type}
                    className="!text-xs !px-2 !py-0.5"
                  />
                  <span className="ml-auto text-gray-500 capitalize">{file.status}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default IngestionProgress;

