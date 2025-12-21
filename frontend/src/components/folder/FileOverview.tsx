import React, { useMemo } from 'react';
import { FileMetadata } from '../../services/types';
import { formatFileSize } from '../../utils/formatters';
import FileBadge from '../common/FileBadge';

/**
 * File type icon mapping
 */
const getFileIcon = (mimeType: string): JSX.Element => {
  const iconClass = "w-5 h-5";
  
  if (mimeType.includes('pdf')) {
    return (
      <svg className={iconClass} fill="currentColor" viewBox="0 0 24 24">
        <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
      </svg>
    );
  }
  
  if (mimeType.includes('word') || mimeType.includes('document')) {
    return (
      <svg className={iconClass} fill="currentColor" viewBox="0 0 24 24">
        <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
      </svg>
    );
  }
  
  if (mimeType.includes('sheet') || mimeType.includes('excel')) {
    return (
      <svg className={iconClass} fill="currentColor" viewBox="0 0 24 24">
        <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
      </svg>
    );
  }
  
  if (mimeType.includes('text') || mimeType.includes('plain')) {
    return (
      <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    );
  }
  
  // Default file icon
  return (
    <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  );
};

/**
 * Get file type badge color
 */
const getFileTypeColor = (mimeType: string): string => {
  if (mimeType.includes('pdf')) return 'bg-red-500/20 text-red-400 border-red-500/30';
  if (mimeType.includes('word') || mimeType.includes('document')) return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
  if (mimeType.includes('sheet') || mimeType.includes('excel')) return 'bg-green-500/20 text-green-400 border-green-500/30';
  if (mimeType.includes('text') || mimeType.includes('plain')) return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
  return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
};

/**
 * Get file extension from MIME type or filename
 */
const getFileExtension = (mimeType: string, fileName: string): string => {
  if (mimeType) {
    const parts = mimeType.split('/');
    if (parts.length > 1) {
      const type = parts[1];
      if (type.includes('pdf')) return 'PDF';
      if (type.includes('word') || type.includes('document')) return 'DOC';
      if (type.includes('sheet') || type.includes('excel')) return 'XLS';
      if (type.includes('text') || type.includes('plain')) return 'TXT';
    }
  }
  
  // Fallback to file extension
  const ext = fileName.split('.').pop()?.toUpperCase();
  return ext || 'FILE';
};

interface FileOverviewProps {
  files: FileMetadata[];
  isProcessing?: boolean;
}

/**
 * FileOverview Component
 * 
 * Displays detected files in a folder with file type icons,
 * sizes, and processing status. Follows Figma design patterns.
 */
const FileOverview: React.FC<FileOverviewProps> = ({ files, isProcessing = false }) => {
  if (!files || files.length === 0) {
    return null;
  }

  const totalSize = files.reduce((sum, file) => sum + file.size, 0);

  // Calculate file type distribution
  const fileTypeDistribution = useMemo(() => {
    const distribution: Record<string, number> = {};
    
    files.forEach((file) => {
      const type = getFileTypeFromMime(file.mime_type);
      distribution[type] = (distribution[type] || 0) + 1;
    });
    
    return distribution;
  }, [files]);

  const getFileTypeFromMime = (mimeType: string): string => {
    if (mimeType.includes('pdf')) return 'PDF';
    if (mimeType.includes('word') || mimeType.includes('document')) return 'Document';
    if (mimeType.includes('sheet') || mimeType.includes('excel')) return 'Spreadsheet';
    if (mimeType.includes('presentation') || mimeType.includes('powerpoint')) return 'Presentation';
    if (mimeType.includes('audio')) return 'Audio';
    if (mimeType.includes('video')) return 'Video';
    if (mimeType.includes('image')) return 'Image';
    if (mimeType.includes('text') && (
      mimeType.includes('javascript') || 
      mimeType.includes('python') || 
      mimeType.includes('java')
    )) return 'Code';
    return 'Other';
  };

  // Estimate processing time (rough: 1MB ≈ 2 seconds)
  const estimatedProcessingTime = useMemo(() => {
    const totalMB = totalSize / (1024 * 1024);
    const seconds = Math.ceil(totalMB * 2);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
  }, [totalSize]);

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm border-2 border-gray-700 rounded-lg p-6 shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <svg
            className="w-5 h-5 text-foldex-primary-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M3.75 9.776c.112-.017.227-.026.344-.026h15.812c.117 0 .232.009.344.026m-16.5 0a2.25 2.25 0 00-1.883 2.542l.857 6a2.25 2.25 0 002.227 1.932H19.05a2.25 2.25 0 002.227-1.932l.857-6a2.25 2.25 0 00-1.883-2.542m-16.5 0V6A2.25 2.25 0 016 3.75h3.879a1.5 1.5 0 011.06.44l2.122 2.12a1.5 1.5 0 001.06.44H18A2.25 2.25 0 0120.25 9v.776"
            />
          </svg>
          <h2 className="text-xl font-semibold text-gray-100">
            Files in Folder
          </h2>
        </div>
        <div className="text-sm text-gray-400">
          {files.length} {files.length === 1 ? 'file' : 'files'} • {formatFileSize(totalSize)}
        </div>
      </div>

      {/* File List */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {files.map((file, index) => (
          <div
            key={file.file_id || index}
            className={`
              flex items-center justify-between
              p-3 rounded-lg
              border transition-all
              ${isProcessing ? 'bg-gray-700/30 border-gray-600' : 'bg-gray-700/50 border-gray-600 hover:bg-gray-700/70'}
            `}
          >
            <div className="flex items-center gap-3 flex-1 min-w-0">
              {/* File Icon */}
              <div className={`flex-shrink-0 p-2 rounded ${getFileTypeColor(file.mime_type)}`}>
                {getFileIcon(file.mime_type)}
              </div>

              {/* File Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-medium text-gray-200 truncate">
                    {file.file_name}
                  </p>
                  <span className={`text-xs px-2 py-0.5 rounded ${getFileTypeColor(file.mime_type)}`}>
                    {getFileExtension(file.mime_type, file.file_name)}
                  </span>
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  {formatFileSize(file.size)}
                </p>
              </div>
            </div>

            {/* Processing Indicator */}
            {isProcessing && (
              <div className="flex-shrink-0 ml-3">
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-foldex-primary-500 border-t-transparent"></div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* File Type Distribution */}
      {Object.keys(fileTypeDistribution).length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-700">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Files by Type</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {Object.entries(fileTypeDistribution).map(([type, count]) => (
              <div key={type} className="flex items-center justify-between text-xs bg-gray-800/30 rounded px-2 py-1">
                <span className="text-gray-400">{type}</span>
                <span className="text-gray-200 font-medium">{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Summary */}
      <div className="mt-4 pt-4 border-t border-gray-700 space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-400">Total size</span>
          <span className="text-gray-200 font-medium">{formatFileSize(totalSize)}</span>
        </div>
        {isProcessing && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Est. processing time</span>
            <span className="text-gray-200 font-medium">{estimatedProcessingTime}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileOverview;
