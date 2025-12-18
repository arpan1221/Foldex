import { useState, useCallback, useMemo, useEffect } from 'react';
import { Node, Edge } from 'reactflow';
import { KnowledgeGraphNode, KnowledgeGraphEdge, Relationship } from '../services/types';
import { folderService } from '../services/api';

/**
 * Graph layout algorithms
 */
export type GraphLayout = 'hierarchical' | 'force' | 'circular';

interface UseGraphVisualizationReturn {
  // Data
  nodes: KnowledgeGraphNode[];
  edges: KnowledgeGraphEdge[];
  relationships: Relationship[];
  
  // State
  isLoading: boolean;
  error: Error | null;
  selectedLayout: GraphLayout;
  selectedFileTypes: string[];
  selectedRelationshipTypes: Relationship['type'][];
  selectedNodeId: string | null;
  
  // Actions
  setSelectedLayout: (layout: GraphLayout) => void;
  setSelectedFileTypes: (types: string[]) => void;
  setSelectedRelationshipTypes: (types: Relationship['type'][]) => void;
  setSelectedNodeId: (nodeId: string | null) => void;
  loadGraph: (folderId: string) => Promise<void>;
  resetFilters: () => void;
  
  // Computed
  filteredNodes: KnowledgeGraphNode[];
  filteredEdges: KnowledgeGraphEdge[];
  filteredRelationships: Relationship[];
  availableFileTypes: string[];
  availableRelationshipTypes: Relationship['type'][];
  
  // Layout functions
  applyLayout: (nodes: Node[], layout: GraphLayout) => Node[];
  fitView: () => void;
  resetView: () => void;
}

/**
 * useGraphVisualization Hook
 * 
 * Manages knowledge graph visualization state, filtering, and layout.
 * Handles data loading, filtering by file type and relationship type,
 * and provides layout algorithms for graph positioning.
 * 
 * @returns {UseGraphVisualizationReturn} Graph visualization state and methods
 * 
 * @example
 * ```tsx
 * const {
 *   nodes,
 *   edges,
 *   loadGraph,
 *   selectedLayout,
 *   setSelectedLayout,
 * } = useGraphVisualization();
 * 
 * useEffect(() => {
 *   loadGraph(folderId);
 * }, [folderId]);
 * ```
 */
export const useGraphVisualization = (): UseGraphVisualizationReturn => {
  const [nodes, setNodes] = useState<KnowledgeGraphNode[]>([]);
  const [edges, setEdges] = useState<KnowledgeGraphEdge[]>([]);
  const [relationships, setRelationships] = useState<Relationship[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);
  
  // Filter state
  const [selectedLayout, setSelectedLayout] = useState<GraphLayout>('force');
  const [selectedFileTypes, setSelectedFileTypes] = useState<string[]>([]);
  const [selectedRelationshipTypes, setSelectedRelationshipTypes] = useState<Relationship['type'][]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  /**
   * Load graph data for a folder
   */
  const loadGraph = useCallback(async (folderId: string) => {
    setIsLoading(true);
    setError(null);

    try {
      // TODO: Replace with actual API call when endpoint is available
      // For now, use mock data structure
      const response = await folderService.getKnowledgeGraph(folderId);
      
      setNodes(response.nodes || []);
      setEdges(response.edges || []);
      setRelationships(response.relationships || []);
    } catch (err) {
      // If API fails, use empty data (graph will show empty state)
      console.warn('Graph loading error (using empty data):', err);
      setNodes([]);
      setEdges([]);
      setRelationships([]);
      
      // Only set error if it's a critical failure
      if (err instanceof Error && !err.message.includes('404')) {
        setError(err);
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Get available file types from nodes
   */
  const availableFileTypes = useMemo(() => {
    const types = new Set<string>();
    nodes.forEach((node) => {
      const fileType = node.metadata?.mime_type as string;
      if (fileType) {
        // Extract file extension or type
        const type = fileType.split('/')[1]?.toUpperCase() || 'UNKNOWN';
        types.add(type);
      }
    });
    return Array.from(types).sort();
  }, [nodes]);

  /**
   * Get available relationship types
   */
  const availableRelationshipTypes = useMemo(() => {
    const types = new Set<Relationship['type']>();
    relationships.forEach((rel) => {
      types.add(rel.type);
    });
    return Array.from(types);
  }, [relationships]);

  /**
   * Filter nodes by file type
   */
  const filteredNodes = useMemo(() => {
    if (selectedFileTypes.length === 0) {
      return nodes;
    }

    return nodes.filter((node) => {
      const fileType = node.metadata?.mime_type as string;
      if (!fileType) return false;
      
      const type = fileType.split('/')[1]?.toUpperCase() || 'UNKNOWN';
      return selectedFileTypes.includes(type);
    });
  }, [nodes, selectedFileTypes]);

  /**
   * Filter edges by relationship type and filtered nodes
   */
  const filteredEdges = useMemo(() => {
    let filtered = edges;

    // Filter by relationship type
    if (selectedRelationshipTypes.length > 0) {
      filtered = filtered.filter((edge) => {
        const rel = relationships.find(
          (r) => r.source === edge.source && r.target === edge.target
        );
        return rel && selectedRelationshipTypes.includes(rel.type);
      });
    }

    // Filter by node visibility
    const visibleNodeIds = new Set(filteredNodes.map((n) => n.node_id));
    filtered = filtered.filter(
      (edge) =>
        visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)
    );

    return filtered;
  }, [edges, relationships, selectedRelationshipTypes, filteredNodes]);

  /**
   * Filter relationships to match filtered edges
   */
  const filteredRelationships = useMemo(() => {
    const visibleEdgePairs = new Set(
      filteredEdges.map((e) => `${e.source}-${e.target}`)
    );

    return relationships.filter((rel) =>
      visibleEdgePairs.has(`${rel.source}-${rel.target}`)
    );
  }, [relationships, filteredEdges]);

  /**
   * Reset all filters
   */
  const resetFilters = useCallback(() => {
    setSelectedFileTypes([]);
    setSelectedRelationshipTypes([]);
    setSelectedNodeId(null);
  }, []);

  /**
   * Apply layout algorithm to nodes
   */
  const applyLayout = useCallback(
    (flowNodes: Node[], layout: GraphLayout): Node[] => {
      const nodeCount = flowNodes.length;
      if (nodeCount === 0) return flowNodes;

      const radius = Math.max(200, nodeCount * 20);
      const centerX = 400;
      const centerY = 300;

      switch (layout) {
        case 'circular': {
          return flowNodes.map((node, index) => {
            const angle = (2 * Math.PI * index) / nodeCount;
            return {
              ...node,
              position: {
                x: centerX + radius * Math.cos(angle),
                y: centerY + radius * Math.sin(angle),
              },
            };
          });
        }

        case 'hierarchical': {
          const levels = Math.ceil(Math.sqrt(nodeCount));
          const nodesPerLevel = Math.ceil(nodeCount / levels);
          
          return flowNodes.map((node, index) => {
            const level = Math.floor(index / nodesPerLevel);
            const positionInLevel = index % nodesPerLevel;
            const levelWidth = nodesPerLevel * 150;
            const startX = centerX - levelWidth / 2;
            
            return {
              ...node,
              position: {
                x: startX + positionInLevel * 150,
                y: 100 + level * 150,
              },
            };
          });
        }

        case 'force':
        default: {
          // Force-directed layout (simplified - React Flow handles this)
          return flowNodes.map((node, index) => {
            const angle = (2 * Math.PI * index) / nodeCount;
            return {
              ...node,
              position: {
                x: centerX + (radius * 0.5) * Math.cos(angle) + (Math.random() - 0.5) * 100,
                y: centerY + (radius * 0.5) * Math.sin(angle) + (Math.random() - 0.5) * 100,
              },
            };
          });
        }
      }
    },
    []
  );

  /**
   * Fit view to all nodes (placeholder - will be handled by React Flow)
   */
  const fitView = useCallback(() => {
    // This will be called by React Flow's fitView
    console.log('Fitting view to nodes');
  }, []);

  /**
   * Reset view (placeholder)
   */
  const resetView = useCallback(() => {
    setSelectedNodeId(null);
    resetFilters();
    console.log('Resetting view');
  }, [resetFilters]);

  return {
    // Data
    nodes,
    edges,
    relationships,
    
    // State
    isLoading,
    error,
    selectedLayout,
    selectedFileTypes,
    selectedRelationshipTypes,
    selectedNodeId,
    
    // Actions
    setSelectedLayout,
    setSelectedFileTypes,
    setSelectedRelationshipTypes,
    setSelectedNodeId,
    loadGraph,
    resetFilters,
    
    // Computed
    filteredNodes,
    filteredEdges,
    filteredRelationships,
    availableFileTypes,
    availableRelationshipTypes,
    
    // Layout
    applyLayout,
    fitView,
    resetView,
  };
};

