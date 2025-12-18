import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import { folderService } from '../../services/api';
import { FolderMetadata } from '../../services/types';

/**
 * Sidebar Component
 * 
 * Navigation sidebar with folder list, search, and quick actions.
 * Follows Figma wireframe design with dark theme and responsive behavior.
 */
const Sidebar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { isAuthenticated } = useAuth();
  const [folders, setFolders] = useState<FolderMetadata[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Load folders on mount
  useEffect(() => {
    if (isAuthenticated) {
      loadFolders();
    }
  }, [isAuthenticated]);

  const loadFolders = async () => {
    setIsLoading(true);
    try {
      const foldersData = await folderService.getUserFolders();
      setFolders(foldersData);
    } catch (error) {
      console.error('Failed to load folders:', error);
      // Set empty array on error (will show empty state)
      setFolders([]);
    } finally {
      setIsLoading(false);
    }
  };

  const filteredFolders = folders.filter((folder) =>
    folder.folder_name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleFolderClick = (folderId: string) => {
    navigate(`/chat/${folderId}`);
  };

  const handleNewFolder = () => {
    navigate('/folder');
  };

  if (!isAuthenticated) {
    return null;
  }

  return (
    <aside
      className={`
        bg-gray-900 border-r border-gray-800 flex flex-col
        transition-all duration-300
        ${isCollapsed ? 'w-16' : 'w-80'}
      `}
    >
      {/* Sidebar Header */}
      <div className="p-4 border-b border-gray-800 flex items-center justify-between">
        {!isCollapsed && (
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
            <h2 className="text-lg font-semibold text-gray-100">Folders</h2>
          </div>
        )}
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="p-1.5 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-800 transition-colors"
          title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <svg
            className={`w-5 h-5 transition-transform ${isCollapsed ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M11 19l-7-7 7-7m8 14l-7-7 7-7"
            />
          </svg>
        </button>
      </div>

      {/* Search (when not collapsed) */}
      {!isCollapsed && (
        <div className="p-4 border-b border-gray-800">
          <div className="relative">
            <svg
              className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
            <input
              type="text"
              placeholder="Search folders..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-100 placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-foldex-primary-500 focus:border-transparent"
            />
          </div>
        </div>
      )}

      {/* New Folder Button */}
      {!isCollapsed && (
        <div className="p-4 border-b border-gray-800">
          <button
            onClick={handleNewFolder}
            className="w-full px-4 py-2 bg-gradient-to-r from-foldex-primary-600 to-foldex-accent-600 hover:from-foldex-primary-700 hover:to-foldex-accent-700 text-white rounded-lg font-medium transition-all flex items-center justify-center gap-2"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 4v16m8-8H4"
              />
            </svg>
            <span>New Folder</span>
          </button>
        </div>
      )}

      {/* Folders List */}
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="p-4 flex items-center justify-center">
            <svg
              className="animate-spin h-6 w-6 text-foldex-primary-500"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          </div>
        ) : filteredFolders.length === 0 ? (
          <div className="p-4 text-center">
            <svg
              className="w-12 h-12 text-gray-600 mx-auto mb-3"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M3.75 9.776c.112-.017.227-.026.344-.026h15.812c.117 0 .232.009.344.026m-16.5 0a2.25 2.25 0 00-1.883 2.542l.857 6a2.25 2.25 0 002.227 1.932H19.05a2.25 2.25 0 002.227-1.932l.857-6a2.25 2.25 0 00-1.883-2.542m-16.5 0V6A2.25 2.25 0 016 3.75h3.879a1.5 1.5 0 011.06.44l2.122 2.12a1.5 1.5 0 001.06.44H18A2.25 2.25 0 0120.25 9v.776"
              />
            </svg>
            <p className="text-gray-400 text-sm mb-2">No folders yet</p>
            <button
              onClick={handleNewFolder}
              className="text-foldex-primary-400 hover:text-foldex-primary-300 text-sm font-medium"
            >
              Add your first folder
            </button>
          </div>
        ) : (
          <div className="p-2">
            <div className="text-xs font-medium text-gray-500 px-2 py-1 mb-1">
              {filteredFolders.length} {filteredFolders.length === 1 ? 'folder' : 'folders'}
            </div>
            {filteredFolders.map((folder) => {
              const isActive = location.pathname === `/chat/${folder.folder_id}`;
              return (
                <button
                  key={folder.folder_id}
                  onClick={() => handleFolderClick(folder.folder_id)}
                  className={`
                    w-full flex items-center gap-3 px-3 py-2.5 rounded-lg
                    transition-colors mb-1
                    ${
                      isActive
                        ? 'bg-foldex-primary-950/50 border border-foldex-primary-800/50'
                        : 'hover:bg-gray-800'
                    }
                  `}
                >
                  <svg
                    className={`w-5 h-5 flex-shrink-0 ${
                      isActive ? 'text-foldex-primary-400' : 'text-gray-400'
                    }`}
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
                  <div className="flex-1 min-w-0 text-left">
                    <p
                      className={`text-sm font-medium truncate ${
                        isActive ? 'text-foldex-primary-200' : 'text-gray-200'
                      }`}
                    >
                      {folder.folder_name}
                    </p>
                    <p className="text-xs text-gray-500">
                      {folder.file_count} {folder.file_count === 1 ? 'file' : 'files'}
                    </p>
                  </div>
                  {isActive && (
                    <div className="w-2 h-2 bg-foldex-primary-400 rounded-full"></div>
                  )}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Collapsed View Icons */}
      {isCollapsed && (
        <div className="p-2 space-y-2 border-t border-gray-800">
          <button
            onClick={handleNewFolder}
            className="w-full p-2 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-800 transition-colors"
            title="New Folder"
          >
            <svg
              className="w-5 h-5 mx-auto"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 4v16m8-8H4"
              />
            </svg>
          </button>
        </div>
      )}
    </aside>
  );
};

export default Sidebar;

