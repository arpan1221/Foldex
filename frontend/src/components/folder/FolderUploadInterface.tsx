import React, { useState, useEffect, useMemo } from 'react';
import { ProcessingStatus as ProcessingStatusType } from '../../services/types';

interface FileItem {
  file_id: string;
  file_name: string;
  file_path?: string;
  parent_folder_id?: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress?: number;
  error?: string;
}

interface FolderItem {
  folder_id: string;
  folder_name: string;
  folder_path?: string;
  parent_folder_id?: string;
  file_count: number;
  subfolder_count: number;
  status: 'pending' | 'processing' | 'completed';
}

interface FolderUploadInterfaceProps {
  status: ProcessingStatusType;
  error?: Error | null;
}

/**
 * FolderUploadInterface Component
 * 
 * Google Drive-style floating upload manager with expandable/collapsible interface
 * showing hierarchical folder structure and real-time file processing progress.
 */
const FolderUploadInterface: React.FC<FolderUploadInterfaceProps> = ({ status, error }) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [isMinimized, setIsMinimized] = useState(false);
  const [files, setFiles] = useState<Map<string, FileItem>>(new Map());
  const [folders, setFolders] = useState<Map<string, FolderItem>>(new Map());
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  // Track backend-provided counts for accurate progress
  const [backendTotalFiles, setBackendTotalFiles] = useState<number>(0);
  const [backendProcessedFiles, setBackendProcessedFiles] = useState<number>(0);
  const [backendFailedFiles, setBackendFailedFiles] = useState<number>(0);
  // Track summarization status
  const [isSummarizing, setIsSummarizing] = useState<boolean>(false);
  const [summarizationProgress, setSummarizationProgress] = useState<number>(0);
  const [summarizationMessage, setSummarizationMessage] = useState<string>('');

  // Update files and folders based on WebSocket messages
  useEffect(() => {
    if (!status) return;

    const updateFromStatus = () => {
      console.log('FolderUploadInterface received status:', status);

      // Update backend-provided totals when available
      if (status.total_files !== undefined) {
        setBackendTotalFiles(status.total_files);
      }
      if (status.files_processed !== undefined) {
        setBackendProcessedFiles(status.files_processed);
      }
      if (status.failed_files !== undefined) {
        setBackendFailedFiles(status.failed_files);
      }

      // Handle file processing updates
      if (status.type === 'file_processing' && status.file_id) {
        setFiles(prev => {
          const updated = new Map(prev);
          updated.set(status.file_id!, {
            file_id: status.file_id!,
            file_name: status.file_name || 'Unknown File',
            file_path: status.file_path,
            parent_folder_id: status.parent_folder_id,
            status: 'processing',
            progress: 0,
          });
          return updated;
        });
      }

      // Handle file processed updates
      if (status.type === 'file_processed' && status.file_id) {
        setFiles(prev => {
          const updated = new Map(prev);
          const existing = updated.get(status.file_id!);
          updated.set(status.file_id!, {
            ...existing,
            file_id: status.file_id!,
            file_name: status.file_name || existing?.file_name || 'Unknown File',
            file_path: status.file_path || existing?.file_path,
            parent_folder_id: status.parent_folder_id || existing?.parent_folder_id,
            status: 'completed',
            progress: 100,
          });
          return updated;
        });
      }

      // Handle file error updates
      if (status.type === 'file_error' && status.file_id) {
        setFiles(prev => {
          const updated = new Map(prev);
          const existing = updated.get(status.file_id!);
          updated.set(status.file_id!, {
            ...existing,
            file_id: status.file_id!,
            file_name: status.file_name || existing?.file_name || 'Unknown File',
            file_path: status.file_path || existing?.file_path,
            parent_folder_id: status.parent_folder_id || existing?.parent_folder_id,
            status: 'error',
            error: status.error || 'Processing failed',
          });
          return updated;
        });
      }

      // Handle folder discovery
      if (status.type === 'folder_discovered' && status.folder_id) {
        setFolders(prev => {
          const updated = new Map(prev);
          updated.set(status.folder_id!, {
            folder_id: status.folder_id!,
            folder_name: status.folder_name || 'Unknown Folder',
            folder_path: status.folder_path,
            parent_folder_id: status.parent_folder_id,
            file_count: status.file_count || 0,
            subfolder_count: status.subfolder_count || 0,
            status: 'processing',
          });
          // Auto-expand root folders
          if (!status.parent_folder_id) {
            setExpandedFolders(prev => new Set(prev).add(status.folder_id!));
          }
          return updated;
        });
      }

      // Handle folder completed
      if (status.type === 'folder_completed' && status.folder_id) {
        setFolders(prev => {
          const updated = new Map(prev);
          const existing = updated.get(status.folder_id!);
          if (existing) {
            updated.set(status.folder_id!, {
              ...existing,
              status: 'completed',
            });
          }
          return updated;
        });
      }

      // Handle summarization/learning status
      if (status.type === 'learning_started') {
        setIsSummarizing(true);
        setSummarizationProgress(0);
        setSummarizationMessage(status.message || 'Summarizing folder contents...');
      }

      if (status.type === 'summary_progress') {
        setIsSummarizing(true);
        if (status.progress !== undefined) {
          setSummarizationProgress(status.progress);
        }
        if (status.message) {
          setSummarizationMessage(status.message);
        }
      }

      if (status.type === 'summary_complete') {
        // Summarization is complete - navigation should happen now
        // Knowledge graph building is separate background task
        setIsSummarizing(false);
        setSummarizationProgress(100);
        setSummarizationMessage('');
      }

      if (status.type === 'summary_error') {
        setIsSummarizing(false);
        setSummarizationMessage('');
      }

      // Knowledge graph building messages are informational only
      // They don't affect summarization state or block navigation
      // (These are handled separately and don't change isSummarizing)

      // Handle folder_structure message with hierarchy
      if (status.type === 'folder_structure' && status.subfolders) {
        const folderMap = new Map<string, FolderItem>();
        const rootFolderId = status.folder_id;

        console.log('Processing folder_structure:', {
          rootFolderId,
          totalSubfolders: status.subfolders.length,
          subfolders: status.subfolders
        });

        // Process flattened subfolders from hierarchy
        // IMPORTANT: Filter out any folder that has the same folder_id as parent_folder_id
        // (this would be the root folder appearing as its own child)
        status.subfolders.forEach((folder: any) => {
          // Skip if this folder is trying to be its own parent
          if (folder.folder_id === folder.parent_folder_id) {
            console.warn('Skipping folder that is its own parent:', folder);
            return;
          }
          
          // Skip if this is the root folder itself (shouldn't be in subfolders list)
          if (folder.folder_id === rootFolderId) {
            console.warn('Skipping root folder from subfolders list:', folder);
            return;
          }

          folderMap.set(folder.folder_id, {
            folder_id: folder.folder_id,
            folder_name: folder.folder_name,
            folder_path: folder.folder_path,
            parent_folder_id: folder.parent_folder_id,
            file_count: folder.file_count || 0,
            subfolder_count: folder.subfolder_count || 0,
            status: 'pending',
          });

          // Auto-expand direct children of root folder
          if (folder.parent_folder_id === rootFolderId) {
            setExpandedFolders(prev => new Set(prev).add(folder.folder_id));
          }
        });

        console.log('Processed folders:', {
          totalFolders: folderMap.size,
          folders: Array.from(folderMap.values())
        });

        setFolders(folderMap);
      }
    };

    updateFromStatus();
  }, [status]);

  // Calculate overall progress using backend-provided counts
  const progress = useMemo(() => {
    // Use backend counts if available, otherwise fall back to local file tracking
    const total = backendTotalFiles > 0 ? backendTotalFiles : files.size;
    const processed = backendProcessedFiles > 0 ? backendProcessedFiles :
      Array.from(files.values()).filter(f => f.status === 'completed' || f.status === 'error').length;

    if (total === 0) return 0;
    return Math.round((processed / total) * 100);
  }, [backendTotalFiles, backendProcessedFiles, files]);

  // Use backend counts for display, fall back to local counts if not available
  const totalFiles = backendTotalFiles > 0 ? backendTotalFiles : files.size;
  const completedFiles = backendProcessedFiles > 0 ? backendProcessedFiles :
    Array.from(files.values()).filter(f => f.status === 'completed').length;
  const errorFiles = Array.from(files.values()).filter(f => f.status === 'error').length;
  const processingFiles = Array.from(files.values()).filter(f => f.status === 'processing').length;

  const isComplete = status?.type === 'processing_complete' || (totalFiles > 0 && completedFiles + errorFiles === totalFiles);
  const isFullyComplete = isComplete && !isSummarizing;

  const toggleFolder = (folderId: string) => {
    setExpandedFolders(prev => {
      const newSet = new Set(prev);
      if (newSet.has(folderId)) {
        newSet.delete(folderId);
      } else {
        newSet.add(folderId);
      }
      return newSet;
    });
  };

  // Build hierarchical structure
  // Show only direct children of the root folder (the folder being processed)
  // The root folder ID comes from status.folder_id
  const rootFolderId = status?.folder_id;
  const rootFolders = useMemo(() => {
    if (!rootFolderId) return [];
    const allFolders = Array.from(folders.values());
    // Show only folders whose parent is the root folder being processed
    return allFolders.filter(f => f.parent_folder_id === rootFolderId);
  }, [folders, rootFolderId]);

  const getSubfolders = (parentId: string) => {
    return Array.from(folders.values()).filter(f => f.parent_folder_id === parentId);
  };

  const getFiles = (folderId: string) => {
    return Array.from(files.values()).filter(f => f.parent_folder_id === folderId);
  };

  // Render folder tree recursively
  const renderFolder = (folder: FolderItem, level: number = 0) => {
    const isExpanded = expandedFolders.has(folder.folder_id);
    const subfolders = getSubfolders(folder.folder_id);
    const folderFiles = getFiles(folder.folder_id);
    const hasChildren = subfolders.length > 0 || folderFiles.length > 0;

    return (
      <div key={folder.folder_id} className="mb-1">
        <div
          className="flex items-center gap-2 py-1.5 px-2 hover:bg-gray-700/50 rounded cursor-pointer"
          style={{ paddingLeft: `${level * 16 + 8}px` }}
          onClick={() => hasChildren && toggleFolder(folder.folder_id)}
        >
          {hasChildren && (
            <svg
              className={`w-3 h-3 text-gray-400 transition-transform flex-shrink-0 ${
                isExpanded ? 'rotate-90' : ''
              }`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          )}
          {!hasChildren && <div className="w-3" />}
          
          <svg className="w-4 h-4 text-blue-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
          </svg>
          
          <span className="text-sm text-gray-200 truncate flex-1">{folder.folder_name}</span>
          
          {folder.status === 'completed' && (
            <svg className="w-4 h-4 text-green-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          )}
        </div>

        {isExpanded && (
          <div>
            {subfolders.map(subfolder => renderFolder(subfolder, level + 1))}
            {folderFiles.map(file => renderFile(file, level + 1))}
          </div>
        )}
      </div>
    );
  };

  const renderFile = (file: FileItem, level: number = 0) => {
    return (
      <div
        key={file.file_id}
        className="flex items-center gap-2 py-1.5 px-2 hover:bg-gray-700/50 rounded"
        style={{ paddingLeft: `${level * 16 + 24}px` }}
      >
        <svg className="w-4 h-4 text-gray-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
        </svg>
        
        <span className="text-sm text-gray-300 truncate flex-1">{file.file_name}</span>
        
        {file.status === 'processing' && (
          <svg className="animate-spin w-4 h-4 text-blue-400 flex-shrink-0" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        )}
        
        {file.status === 'completed' && (
          <svg className="w-4 h-4 text-green-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        )}
        
        {file.status === 'error' && (
          <svg className="w-4 h-4 text-red-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        )}
      </div>
    );
  };

  if (isMinimized) {
    return (
      <div className="fixed bottom-6 right-6 z-50">
        <button
          onClick={() => setIsMinimized(false)}
          className="bg-gray-800 hover:bg-gray-700 text-white rounded-full p-4 shadow-2xl border border-gray-600 transition-all"
        >
          <div className="flex items-center gap-3">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
            </svg>
            <div className="text-left">
              <div className="text-sm font-medium">
                {isFullyComplete ? 'Upload Complete' : isSummarizing ? 'Summarizing...' : 'Uploading...'}
              </div>
              <div className="text-xs text-gray-400">
                {isSummarizing ? summarizationMessage || 'Summarizing folder contents...' : `${completedFiles} / ${totalFiles} files`}
              </div>
            </div>
          </div>
        </button>
      </div>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 z-50 w-96 bg-gray-800 rounded-lg shadow-2xl border border-gray-600 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700 bg-gray-800">
        <div className="flex items-center gap-3">
          <div className={`w-2 h-2 rounded-full ${
            isFullyComplete ? 'bg-green-400' : 
            isSummarizing ? 'bg-purple-400 animate-pulse' : 
            'bg-blue-400 animate-pulse'
          }`} />
          <div>
            <h3 className="text-sm font-semibold text-gray-100">
              {isFullyComplete ? 'Upload Complete' : 
               isSummarizing ? 'Summarizing folder contents' : 
               'Foldex is indexing your files'}
            </h3>
            <p className="text-xs text-gray-400">
              {isSummarizing 
                ? (summarizationMessage || 'Analyzing your folder...')
                : `${completedFiles} of ${totalFiles} files • ${errorFiles > 0 ? `${errorFiles} failed` : 'No errors'}`
              }
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-1">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1.5 hover:bg-gray-700 rounded transition-colors"
            title={isExpanded ? 'Collapse' : 'Expand'}
          >
            <svg
              className={`w-4 h-4 text-gray-400 transition-transform ${isExpanded ? '' : 'rotate-180'}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          <button
            onClick={() => setIsMinimized(true)}
            className="p-1.5 hover:bg-gray-700 rounded transition-colors"
            title="Minimize"
          >
            <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
            </svg>
          </button>
        </div>
      </div>

      {/* Progress Bar - Always show if there's any status or files being processed */}
      {(status || files.size > 0 || folders.size > 0) && (
        <div className="px-4 pt-3 pb-2 bg-gray-800">
          {isSummarizing ? (
          <>
            <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
              <div
                className="h-full transition-all duration-300 bg-purple-500"
                style={{ width: `${Math.max(summarizationProgress * 100, 10)}%` }}
              />
            </div>
            <div className="flex justify-between items-center mt-2">
              <span className="text-xs text-gray-400">
                {summarizationMessage || 'Summarizing folder contents...'}
              </span>
              <span className="text-xs text-purple-400">
                {Math.round(summarizationProgress * 100)}%
              </span>
            </div>
          </>
        ) : (
          <>
            <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
              <div
                className={`h-full transition-all duration-300 ${
                  isComplete ? 'bg-green-500' : 'bg-blue-500'
                }`}
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="flex justify-between items-center mt-2">
              <span className="text-xs text-gray-400">{progress}% complete</span>
              {processingFiles > 0 && (
                <span className="text-xs text-blue-400">{processingFiles} processing</span>
              )}
            </div>
          </>
          )}
        </div>
      )}

      {/* File List */}
      {isExpanded && (
        <div className="max-h-96 overflow-y-auto bg-gray-850 custom-scrollbar">
          {error && (
            <div className="m-4 p-3 bg-red-900/20 border border-red-700 rounded-lg">
              <div className="flex items-start gap-2">
                <svg className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                  <p className="text-sm font-medium text-red-300">Processing Error</p>
                  <p className="text-xs text-red-400 mt-1">{error.message || 'An error occurred'}</p>
                </div>
              </div>
            </div>
          )}

          <div className="p-2">
            {(rootFolders.length > 0 || getFiles(rootFolderId || '').length > 0) ? (
              <>
                {/* Render direct child folders */}
                {rootFolders.map(folder => renderFolder(folder))}
                {/* Render direct child files of root */}
                {getFiles(rootFolderId || '').map(file => renderFile(file, 0))}
              </>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <svg className="w-12 h-12 mx-auto mb-3 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
                </svg>
                <p className="text-sm">Preparing files...</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Footer */}
      {isFullyComplete && isExpanded && (
        <div className="p-3 border-t border-gray-700 bg-gray-800">
          <div className="flex items-center justify-between">
            {errorFiles > 0 || backendFailedFiles > 0 ? (
              <div className="flex items-center gap-2 text-yellow-400">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <span className="text-sm font-medium">
                  Processing complete ({errorFiles || backendFailedFiles} file{errorFiles !== 1 && backendFailedFiles !== 1 ? 's' : ''} failed)
                </span>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-green-400">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-sm font-medium">All files processed</span>
              </div>
            )}
            <button
              onClick={() => setIsMinimized(true)}
              className="text-xs text-gray-400 hover:text-gray-200 transition-colors"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default FolderUploadInterface;
