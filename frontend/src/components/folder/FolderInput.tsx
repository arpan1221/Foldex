import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useFolderProcessor } from '../../hooks/useFolderProcessor';
import ProcessingStatus from './ProcessingStatus';
import FileOverview from './FileOverview';

const FolderInput: React.FC = () => {
  const [folderUrl, setFolderUrl] = useState('');
  const { processFolder, isProcessing, status } = useFolderProcessor();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!folderUrl.trim()) return;

    const folderId = extractFolderId(folderUrl);
    if (folderId) {
      await processFolder(folderId);
      navigate(`/chat/${folderId}`);
    }
  };

  const extractFolderId = (url: string): string | null => {
    // Extract folder ID from Google Drive URL
    const match = url.match(/\/folders\/([a-zA-Z0-9_-]+)/);
    return match ? match[1] : null;
  };

  return (
    <div className="max-w-2xl mx-auto p-8">
      <h1 className="text-3xl font-bold mb-6">Process Google Drive Folder</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="folder-url" className="block text-sm font-medium mb-2">
            Google Drive Folder URL
          </label>
          <input
            id="folder-url"
            type="text"
            value={folderUrl}
            onChange={(e) => setFolderUrl(e.target.value)}
            placeholder="https://drive.google.com/drive/folders/..."
            className="w-full px-4 py-2 border rounded-lg"
            disabled={isProcessing}
          />
        </div>
        <button
          type="submit"
          disabled={isProcessing}
          className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
        >
          {isProcessing ? 'Processing...' : 'Process Folder'}
        </button>
      </form>
      {isProcessing && status && <ProcessingStatus status={status} />}
    </div>
  );
};

export default FolderInput;

