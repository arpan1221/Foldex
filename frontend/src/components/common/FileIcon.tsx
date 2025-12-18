import React from 'react';

interface FileIconProps {
  mimeType: string;
  fileName?: string;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

/**
 * FileIcon Component
 * Displays appropriate icon based on file MIME type or extension
 */
const FileIcon: React.FC<FileIconProps> = ({ 
  mimeType, 
  fileName = '', 
  className = '',
  size = 'md'
}) => {
  const sizeClasses = {
    sm: 'w-3 h-3',
    md: 'w-4 h-4',
    lg: 'w-5 h-5'
  };

  const baseClasses = `${sizeClasses[size]} ${className} flex-shrink-0`;

  // Get file extension from filename as fallback
  const getExtension = (name: string): string => {
    const parts = name.split('.');
    return parts.length > 1 ? parts[parts.length - 1].toLowerCase() : '';
  };

  const extension = getExtension(fileName);
  const mime = mimeType.toLowerCase();

  // PDF
  if (mime.includes('pdf') || extension === 'pdf') {
    return (
      <svg className={`${baseClasses} text-red-500`} fill="currentColor" viewBox="0 0 24 24">
        <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
        <path d="M8,12H16V14H8V12M8,16H13V18H8V16Z" fill="currentColor" />
      </svg>
    );
  }

  // Word Document
  if (mime.includes('word') || mime.includes('document') || extension === 'doc' || extension === 'docx') {
    return (
      <svg className={`${baseClasses} text-blue-600`} fill="currentColor" viewBox="0 0 24 24">
        <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
        <path d="M8,12H16V14H8V12M8,16H13V18H8V16Z" fill="currentColor" />
      </svg>
    );
  }

  // Excel/Spreadsheet
  if (mime.includes('spreadsheet') || mime.includes('excel') || extension === 'xls' || extension === 'xlsx' || extension === 'csv') {
    return (
      <svg className={`${baseClasses} text-green-600`} fill="currentColor" viewBox="0 0 24 24">
        <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
        <path d="M8,12H10V14H8V12M11,12H13V14H11V12M14,12H16V14H14V12M8,16H10V18H8V16M11,16H13V18H11V16M14,16H16V18H14V16Z" fill="currentColor" />
      </svg>
    );
  }

  // PowerPoint/Presentation
  if (mime.includes('presentation') || mime.includes('powerpoint') || extension === 'ppt' || extension === 'pptx') {
    return (
      <svg className={`${baseClasses} text-orange-500`} fill="currentColor" viewBox="0 0 24 24">
        <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
        <path d="M8,12H16V14H8V12M8,16H16V18H8V16Z" fill="currentColor" />
      </svg>
    );
  }

  // Images
  if (mime.includes('image') || ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'webp'].includes(extension)) {
    return (
      <svg className={`${baseClasses} text-purple-500`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    );
  }

  // Audio
  if (mime.includes('audio') || ['mp3', 'wav', 'ogg', 'flac', 'aac', 'm4a'].includes(extension)) {
    return (
      <svg className={`${baseClasses} text-pink-500`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
      </svg>
    );
  }

  // Video
  if (mime.includes('video') || ['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv'].includes(extension)) {
    return (
      <svg className={`${baseClasses} text-red-600`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
      </svg>
    );
  }

  // Code files
  if (['js', 'jsx', 'ts', 'tsx', 'py', 'java', 'cpp', 'c', 'cs', 'php', 'rb', 'go', 'rs', 'swift', 'kt', 'html', 'css', 'scss', 'sass', 'json', 'xml', 'yaml', 'yml', 'sh', 'bash', 'zsh'].includes(extension)) {
    return (
      <svg className={`${baseClasses} text-yellow-500`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
      </svg>
    );
  }

  // Text files
  if (mime.includes('text') || extension === 'txt' || extension === 'md' || extension === 'markdown' || extension === 'rtf') {
    return (
      <svg className={`${baseClasses} text-gray-400`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    );
  }

  // Archive/Zip
  if (mime.includes('zip') || mime.includes('archive') || ['zip', 'rar', '7z', 'tar', 'gz', 'bz2'].includes(extension)) {
    return (
      <svg className={`${baseClasses} text-amber-600`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
      </svg>
    );
  }

  // Google Docs/Sheets/Slides
  if (mime.includes('google-apps')) {
    if (mime.includes('document')) {
      return (
        <svg className={`${baseClasses} text-blue-500`} fill="currentColor" viewBox="0 0 24 24">
          <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
        </svg>
      );
    }
    if (mime.includes('spreadsheet')) {
      return (
        <svg className={`${baseClasses} text-green-500`} fill="currentColor" viewBox="0 0 24 24">
          <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
        </svg>
      );
    }
    if (mime.includes('presentation')) {
      return (
        <svg className={`${baseClasses} text-orange-500`} fill="currentColor" viewBox="0 0 24 24">
          <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
        </svg>
      );
    }
  }

  // Default file icon
  return (
    <svg className={`${baseClasses} text-gray-500`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  );
};

export default FileIcon;

