import React from 'react';
import { Relationship } from '../../services/types';

interface GraphControlsProps {
  selectedLayout: 'hierarchical' | 'force' | 'circular';
  onLayoutChange: (layout: 'hierarchical' | 'force' | 'circular') => void;
  selectedFileTypes: string[];
  onFileTypeFilterChange: (types: string[]) => void;
  selectedRelationshipTypes: Relationship['type'][];
  onRelationshipTypeFilterChange: (types: Relationship['type'][]) => void;
  onResetView: () => void;
  onFitView: () => void;
  availableFileTypes: string[];
  availableRelationshipTypes: Relationship['type'][];
}

/**
 * GraphControls Component
 * 
 * Controls for graph visualization including layout selection,
 * filtering by file type and relationship type, and view controls.
 */
const GraphControls: React.FC<GraphControlsProps> = ({
  selectedLayout,
  onLayoutChange,
  selectedFileTypes,
  onFileTypeFilterChange,
  selectedRelationshipTypes,
  onRelationshipTypeFilterChange,
  onResetView,
  onFitView,
  availableFileTypes,
  availableRelationshipTypes,
}) => {
  const handleFileTypeToggle = (fileType: string) => {
    if (selectedFileTypes.includes(fileType)) {
      onFileTypeFilterChange(selectedFileTypes.filter((t) => t !== fileType));
    } else {
      onFileTypeFilterChange([...selectedFileTypes, fileType]);
    }
  };

  const handleRelationshipTypeToggle = (relType: Relationship['type']) => {
    if (selectedRelationshipTypes.includes(relType)) {
      onRelationshipTypeFilterChange(
        selectedRelationshipTypes.filter((t) => t !== relType)
      );
    } else {
      onRelationshipTypeFilterChange([...selectedRelationshipTypes, relType]);
    }
  };

  const getRelationshipTypeLabel = (type: Relationship['type']): string => {
    const labels: Record<Relationship['type'], string> = {
      entity_overlap: 'Entity Overlap',
      temporal: 'Temporal',
      cross_reference: 'Cross Reference',
      topical_similarity: 'Topical Similarity',
      implementation_gap: 'Implementation Gap',
    };
    return labels[type] || type;
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm border-2 border-gray-700 rounded-lg p-4 space-y-4">
      {/* Layout Selection */}
      <div>
        <label className="text-sm font-semibold text-gray-200 mb-2 block">
          Layout
        </label>
        <div className="flex gap-2">
          {(['hierarchical', 'force', 'circular'] as const).map((layout) => (
            <button
              key={layout}
              onClick={() => onLayoutChange(layout)}
              className={`
                flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-all
                ${
                  selectedLayout === layout
                    ? 'bg-foldex-primary-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }
              `}
            >
              {layout.charAt(0).toUpperCase() + layout.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* File Type Filter */}
      {availableFileTypes.length > 0 && (
        <div>
          <label className="text-sm font-semibold text-gray-200 mb-2 block">
            File Types
          </label>
          <div className="space-y-2 max-h-32 overflow-y-auto">
            {availableFileTypes.map((fileType) => (
              <label
                key={fileType}
                className="flex items-center gap-2 cursor-pointer hover:bg-gray-700/50 p-2 rounded"
              >
                <input
                  type="checkbox"
                  checked={selectedFileTypes.includes(fileType)}
                  onChange={() => handleFileTypeToggle(fileType)}
                  className="w-4 h-4 text-foldex-primary-600 bg-gray-700 border-gray-600 rounded focus:ring-foldex-primary-500"
                />
                <span className="text-sm text-gray-300">{fileType}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Relationship Type Filter */}
      {availableRelationshipTypes.length > 0 && (
        <div>
          <label className="text-sm font-semibold text-gray-200 mb-2 block">
            Relationship Types
          </label>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {availableRelationshipTypes.map((relType) => (
              <label
                key={relType}
                className="flex items-center gap-2 cursor-pointer hover:bg-gray-700/50 p-2 rounded"
              >
                <input
                  type="checkbox"
                  checked={selectedRelationshipTypes.includes(relType)}
                  onChange={() => handleRelationshipTypeToggle(relType)}
                  className="w-4 h-4 text-foldex-primary-600 bg-gray-700 border-gray-600 rounded focus:ring-foldex-primary-500"
                />
                <span className="text-sm text-gray-300">
                  {getRelationshipTypeLabel(relType)}
                </span>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* View Controls */}
      <div className="pt-2 border-t border-gray-700">
        <div className="flex gap-2">
          <button
            onClick={onFitView}
            className="flex-1 px-3 py-2 bg-gray-700 text-gray-300 rounded-lg text-sm font-medium hover:bg-gray-600 transition-colors"
            title="Fit view to all nodes"
          >
            <div className="flex items-center justify-center gap-1">
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"
                />
              </svg>
              Fit View
            </div>
          </button>
          <button
            onClick={onResetView}
            className="flex-1 px-3 py-2 bg-gray-700 text-gray-300 rounded-lg text-sm font-medium hover:bg-gray-600 transition-colors"
            title="Reset view"
          >
            <div className="flex items-center justify-center gap-1">
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              Reset
            </div>
          </button>
        </div>
      </div>
    </div>
  );
};

export default GraphControls;

