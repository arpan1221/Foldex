import React, { useState } from 'react';
import { Relationship } from '../../services/types';

interface RelationshipViewProps {
  relationships: Relationship[];
  onRelationshipClick?: (relationship: Relationship) => void;
  selectedRelationshipId?: string | null;
}

/**
 * RelationshipView Component
 * 
 * Sidebar component displaying document relationships in a list format.
 * Shows relationship type, confidence, and source/target documents.
 * Follows Figma wireframe design with card-based layout.
 */
const RelationshipView: React.FC<RelationshipViewProps> = ({
  relationships,
  onRelationshipClick,
  selectedRelationshipId,
}) => {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  if (!relationships || relationships.length === 0) {
    return (
      <div className="bg-gray-800/50 backdrop-blur-sm border-2 border-gray-700 rounded-lg p-6">
        <div className="text-center">
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
              d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
            />
          </svg>
          <p className="text-gray-400 text-sm">No relationships found</p>
        </div>
      </div>
    );
  }

  const getRelationshipTypeColor = (type: Relationship['type']): string => {
    const colors: Record<Relationship['type'], string> = {
      entity_overlap: 'border-blue-500/50 bg-blue-950/30',
      temporal: 'border-purple-500/50 bg-purple-950/30',
      cross_reference: 'border-green-500/50 bg-green-950/30',
      topical_similarity: 'border-amber-500/50 bg-amber-950/30',
      implementation_gap: 'border-red-500/50 bg-red-950/30',
    };
    return colors[type] || 'border-gray-500/50 bg-gray-800/30';
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

  const getRelationshipIcon = (type: Relationship['type']) => {
    switch (type) {
      case 'entity_overlap':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
          </svg>
        );
      case 'temporal':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      case 'cross_reference':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
          </svg>
        );
      case 'topical_similarity':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        );
      case 'implementation_gap':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        );
      default:
        return null;
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-100 flex items-center gap-2">
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
              d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
            />
          </svg>
          Document Relationships
        </h3>
        <span className="text-sm text-gray-400 bg-gray-700 px-2 py-1 rounded">
          {relationships.length}
        </span>
      </div>

      <div className="space-y-2 max-h-[600px] overflow-y-auto">
        {relationships.map((relationship, index) => {
          const isSelected = selectedRelationshipId === `${relationship.source}-${relationship.target}`;
          const isExpanded = expandedIndex === index;

          return (
            <div
              key={index}
              onClick={() => {
                setExpandedIndex(isExpanded ? null : index);
                if (onRelationshipClick) {
                  onRelationshipClick(relationship);
                }
              }}
              className={`
                p-3 rounded-lg border-l-4 transition-all cursor-pointer
                ${getRelationshipTypeColor(relationship.type)}
                ${isSelected ? 'ring-2 ring-foldex-primary-500' : ''}
                hover:bg-opacity-50
              `}
            >
              {/* Relationship Header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <div className="text-foldex-primary-400">
                    {getRelationshipIcon(relationship.type)}
                  </div>
                  <span className="text-xs font-medium text-gray-300">
                    {getRelationshipTypeLabel(relationship.type)}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-400">
                    {(relationship.confidence * 100).toFixed(0)}%
                  </span>
                  <svg
                    className={`w-4 h-4 text-gray-400 transition-transform ${
                      isExpanded ? 'rotate-180' : ''
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 9l-7 7-7-7"
                    />
                  </svg>
                </div>
              </div>

              {/* Source and Target */}
              <div className="mt-2 space-y-1">
                <div className="flex items-center gap-2 text-sm">
                  <span className="text-gray-400">From:</span>
                  <span className="text-gray-200 font-medium truncate">
                    {relationship.source}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <svg
                    className="w-4 h-4 text-gray-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 9l-7 7-7-7"
                    />
                  </svg>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <span className="text-gray-400">To:</span>
                  <span className="text-gray-200 font-medium truncate">
                    {relationship.target}
                  </span>
                </div>
              </div>

              {/* Expanded Details */}
              {isExpanded && (
                <div className="mt-3 pt-3 border-t border-gray-700 space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-400">Confidence</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-foldex-primary-500 transition-all"
                          style={{ width: `${relationship.confidence * 100}%` }}
                        />
                      </div>
                      <span className="text-gray-300 font-medium">
                        {(relationship.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default RelationshipView;
