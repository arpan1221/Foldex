import React from 'react';

interface FileMetadata {
  file_id: string;
  file_name: string;
  mime_type: string;
  size: number;
}

interface FileOverviewProps {
  files: FileMetadata[];
}

const FileOverview: React.FC<FileOverviewProps> = ({ files }) => {
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="mt-6">
      <h2 className="text-xl font-semibold mb-4">Files in Folder</h2>
      <div className="space-y-2">
        {files.map((file) => (
          <div
            key={file.file_id}
            className="flex items-center justify-between p-3 bg-gray-50 rounded"
          >
            <span className="font-medium">{file.file_name}</span>
            <span className="text-sm text-gray-500">
              {formatFileSize(file.size)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default FileOverview;

