import React from 'react';

interface KnowledgeGraphProps {
  folderId: string;
}

const KnowledgeGraph: React.FC<KnowledgeGraphProps> = ({ folderId }) => {
  // TODO: Implement knowledge graph visualization
  // Use a library like vis-network, cytoscape, or d3.js

  return (
    <div className="w-full h-96 bg-gray-100 rounded-lg flex items-center justify-center">
      <p className="text-gray-500">Knowledge Graph Visualization</p>
    </div>
  );
};

export default KnowledgeGraph;

