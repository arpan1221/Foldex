/**
 * Knowledge Graph Page
 * 
 * Displays the knowledge graph visualization for a folder.
 */

import React from 'react';
import { useParams } from 'react-router-dom';
import { KnowledgeGraphViz } from '../components/KnowledgeGraph';

const KnowledgeGraphPage: React.FC = () => {
  const { folderId } = useParams<{ folderId: string }>();

  if (!folderId) {
    return (
      <div className="p-6">
        <div className="text-red-400">No folder ID provided</div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto">
      <KnowledgeGraphViz folderId={folderId} />
    </div>
  );
};

export default KnowledgeGraphPage;

