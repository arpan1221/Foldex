import React from 'react';

interface Relationship {
  source: string;
  target: string;
  type: string;
  confidence: number;
}

interface RelationshipViewProps {
  relationships: Relationship[];
}

const RelationshipView: React.FC<RelationshipViewProps> = ({ relationships }) => {
  return (
    <div className="space-y-2">
      <h3 className="font-semibold">Document Relationships</h3>
      {relationships.map((rel, index) => (
        <div
          key={index}
          className="p-3 bg-gray-50 rounded border-l-4 border-blue-500"
        >
          <div className="flex items-center justify-between">
            <div>
              <span className="font-medium">{rel.source}</span>
              <span className="mx-2 text-gray-400">→</span>
              <span className="font-medium">{rel.target}</span>
            </div>
            <span className="text-sm text-gray-500">{rel.type}</span>
          </div>
          <div className="text-xs text-gray-400 mt-1">
            Confidence: {(rel.confidence * 100).toFixed(1)}%
          </div>
        </div>
      ))}
    </div>
  );
};

export default RelationshipView;

