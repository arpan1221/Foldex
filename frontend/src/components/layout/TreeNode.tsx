import React from 'react';
import { TreeNode as TreeNodeType } from '../../services/types';
import FileIcon from '../common/FileIcon';

interface TreeNodeProps {
  node: TreeNodeType;
  level: number;
  expandedNodes: Set<string>;
  onToggleExpand: (nodeId: string) => void;
  formatFileSize: (bytes: number) => string;
  formatDate: (dateStr?: string | Date) => string;
  onFileClick?: (fileId: string, fileName: string, folderId: string) => void;
  folderId?: string;
}

/**
 * Recursive TreeNode component for displaying folder/file hierarchy.
 */
const TreeNode: React.FC<TreeNodeProps> = ({
  node,
  level,
  expandedNodes,
  onToggleExpand,
  formatFileSize,
  formatDate,
  onFileClick,
  folderId,
}) => {
  const isExpanded = expandedNodes.has(node.id);
  const hasChildren = node.children && node.children.length > 0;
  const indent = level * 16;
  const isFile = !node.is_folder;

  const handleFileClick = () => {
    if (isFile && onFileClick && folderId && node.id) {
      onFileClick(node.id, node.name, folderId);
    }
  };

  return (
    <div className="select-none">
      <div
        className={`flex items-center gap-1 py-1 px-2 rounded transition-colors group ${
          isFile && onFileClick ? 'hover:bg-gray-800/50 cursor-pointer' : 'hover:bg-gray-800/50'
        }`}
        style={{ paddingLeft: `${8 + indent}px` }}
        onClick={isFile && onFileClick ? handleFileClick : undefined}
      >
        {node.is_folder && hasChildren && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onToggleExpand(node.id);
            }}
            className="p-0.5 rounded hover:bg-gray-700 transition-colors flex-shrink-0"
            title={isExpanded ? 'Collapse' : 'Expand'}
          >
            <svg
              className={`w-3 h-3 text-gray-500 transition-transform ${
                isExpanded ? 'rotate-90' : ''
              }`}
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
          </button>
        )}
        {!node.is_folder && hasChildren && (
          <div className="w-3.5" /> // Spacer for alignment
        )}
        
        <div className="flex items-start gap-2 flex-1 min-w-0">
          {node.is_folder ? (
            <svg
              className="w-4 h-4 text-gray-400 flex-shrink-0 mt-0.5"
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
          ) : (
            <FileIcon
              mimeType={node.mime_type}
              fileName={node.name}
              size="sm"
              className="mt-0.5"
            />
          )}
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium text-gray-300 truncate">
              {node.name}
            </p>
            {!node.is_folder && (
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="text-[10px] text-gray-500">
                  {formatFileSize(node.size || 0)}
                </span>
                {node.created_at && (
                  <>
                    <span className="text-[10px] text-gray-500">•</span>
                    <span className="text-[10px] text-gray-500">
                      {formatDate(node.created_at)}
                    </span>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Render children if expanded */}
      {isExpanded && hasChildren && (
        <div className="border-l border-gray-700 ml-2">
          {node.children.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              level={level + 1}
              expandedNodes={expandedNodes}
              onToggleExpand={onToggleExpand}
              formatFileSize={formatFileSize}
              formatDate={formatDate}
              onFileClick={onFileClick}
              folderId={folderId}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default TreeNode;

