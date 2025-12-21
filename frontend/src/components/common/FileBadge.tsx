import React from 'react';

export type FileTypeCategory = 
  | 'unstructured_native' 
  | 'audio' 
  | 'code' 
  | 'notebook' 
  | 'unsupported';

interface FileBadgeProps {
  fileName: string;
  fileType?: FileTypeCategory;
  mimeType?: string;
  googleDriveUrl?: string;
  onClick?: () => void;
  className?: string;
}

/**
 * FileBadge Component
 * 
 * Displays file type icon, name, and clickable link to Google Drive.
 * Supports different file types with appropriate icons.
 */
const FileBadge: React.FC<FileBadgeProps> = ({
  fileName,
  fileType,
  mimeType,
  googleDriveUrl,
  onClick,
  className = '',
}) => {
  const getFileTypeIcon = (): React.ReactNode => {
    // Determine icon based on file type or mime type
    const type = fileType || getFileTypeFromMime(mimeType);
    
    switch (type) {
      case 'unstructured_native':
        if (mimeType?.includes('pdf')) {
          return <span className="text-red-400">📄</span>;
        } else if (mimeType?.includes('word') || mimeType?.includes('document')) {
          return <span className="text-blue-400">📝</span>;
        } else if (mimeType?.includes('spreadsheet') || mimeType?.includes('excel')) {
          return <span className="text-green-400">📊</span>;
        } else if (mimeType?.includes('presentation') || mimeType?.includes('powerpoint')) {
          return <span className="text-orange-400">📽️</span>;
        } else if (mimeType?.includes('image')) {
          return <span className="text-purple-400">🖼️</span>;
        }
        return <span className="text-gray-400">📄</span>;
      
      case 'audio':
        return <span className="text-pink-400">🎵</span>;
      
      case 'code':
        if (fileName.endsWith('.py')) return <span className="text-yellow-400">🐍</span>;
        if (fileName.endsWith('.js') || fileName.endsWith('.ts')) return <span className="text-yellow-400">⚡</span>;
        if (fileName.endsWith('.java')) return <span className="text-orange-400">☕</span>;
        return <span className="text-blue-400">💻</span>;
      
      case 'notebook':
        return <span className="text-orange-400">📓</span>;
      
      default:
        return <span className="text-gray-400">📄</span>;
    }
  };

  const getFileTypeFromMime = (mime?: string): FileTypeCategory => {
    if (!mime) return 'unstructured_native';
    
    if (mime.includes('audio')) return 'audio';
    if (mime.includes('video')) return 'audio'; // Treat video as audio for transcription
    if (mime.includes('text') && (
      mime.includes('javascript') || 
      mime.includes('python') || 
      mime.includes('java') ||
      mime.includes('x-')
    )) return 'code';
    if (mime.includes('json') && mime.includes('notebook')) return 'notebook';
    
    return 'unstructured_native';
  };

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (googleDriveUrl) {
      window.open(googleDriveUrl, '_blank', 'noopener,noreferrer');
    } else if (onClick) {
      onClick();
    }
  };

  return (
    <button
      onClick={handleClick}
      className={`
        inline-flex items-center gap-2
        px-3 py-1.5
        text-sm
        rounded-lg
        border border-gray-700
        bg-gray-800/50
        hover:bg-gray-800
        hover:border-gray-600
        text-gray-300
        transition-all duration-200
        ${googleDriveUrl || onClick ? 'cursor-pointer' : 'cursor-default'}
        ${className}
      `}
      title={googleDriveUrl ? `Open ${fileName} in Google Drive` : fileName}
    >
      <span className="text-base">{getFileTypeIcon()}</span>
      <span className="truncate max-w-[200px] sm:max-w-none">{fileName}</span>
      {(googleDriveUrl || onClick) && (
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
    </button>
  );
};

export default FileBadge;

