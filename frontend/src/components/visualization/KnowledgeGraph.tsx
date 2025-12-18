import React, { useCallback, useMemo, useEffect } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  NodeTypes,
  Connection,
  addEdge,
  useNodesState,
  useEdgesState,
  MarkerType,
  ReactFlowProvider,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { KnowledgeGraphNode, KnowledgeGraphEdge } from '../../services/types';

interface KnowledgeGraphProps {
  folderId: string;
  nodes: KnowledgeGraphNode[];
  edges: KnowledgeGraphEdge[];
  isLoading?: boolean;
  layout?: 'hierarchical' | 'force' | 'circular';
  onNodeClick?: (nodeId: string) => void;
  onFitView?: () => void;
}

/**
 * KnowledgeGraph Component
 * 
 * Interactive graph visualization using React Flow.
 * Displays documents, entities, and their relationships.
 * Follows Figma wireframe design with dark theme.
 */
const KnowledgeGraphInner: React.FC<KnowledgeGraphProps> = ({
  folderId: _folderId,
  nodes,
  edges,
  isLoading = false,
  layout = 'force',
  onNodeClick,
  onFitView: _onFitView,
}) => {

  // Transform data for React Flow
  const flowNodes: Node[] = useMemo(() => {
    const nodeCount = nodes.length;
    const radius = Math.max(200, nodeCount * 20);
    const centerX = 400;
    const centerY = 300;

    return nodes.map((node, index) => {
      const nodeType = node.type || 'document';
      
      // Determine node size based on connections
      const connections = edges.filter(
        (e) => e.source === node.node_id || e.target === node.node_id
      ).length;
      const size = Math.max(50, Math.min(100, 50 + connections * 5));

      // Calculate position based on layout
      let position = { x: 0, y: 0 };
      
      if (layout === 'circular') {
        const angle = (2 * Math.PI * index) / nodeCount;
        position = {
          x: centerX + radius * Math.cos(angle),
          y: centerY + radius * Math.sin(angle),
        };
      } else if (layout === 'hierarchical') {
        const levels = Math.ceil(Math.sqrt(nodeCount));
        const nodesPerLevel = Math.ceil(nodeCount / levels);
        const level = Math.floor(index / nodesPerLevel);
        const positionInLevel = index % nodesPerLevel;
        const levelWidth = nodesPerLevel * 150;
        const startX = centerX - levelWidth / 2;
        position = {
          x: startX + positionInLevel * 150,
          y: 100 + level * 150,
        };
      } else {
        // Force layout - initial random positions
        const angle = (2 * Math.PI * index) / nodeCount;
        position = {
          x: centerX + (radius * 0.5) * Math.cos(angle) + (Math.random() - 0.5) * 100,
          y: centerY + (radius * 0.5) * Math.sin(angle) + (Math.random() - 0.5) * 100,
        };
      }

      return {
        id: node.node_id,
        type: 'custom',
        position,
        data: {
          label: node.label,
          type: nodeType,
          metadata: node.metadata,
          connections,
        },
        style: {
          width: size,
          height: size,
        },
      };
    });
  }, [nodes, edges, layout]);

  const flowEdges: Edge[] = useMemo(() => {
    return edges.map((edge, index) => {
      // Determine edge color based on relationship type
      const getEdgeColor = () => {
        switch (edge.relationship_type) {
          case 'entity_overlap':
            return '#3b82f6'; // blue
          case 'temporal':
            return '#8b5cf6'; // purple
          case 'cross_reference':
            return '#10b981'; // green
          case 'topical_similarity':
            return '#f59e0b'; // amber
          case 'implementation_gap':
            return '#ef4444'; // red
          default:
            return '#6b7280'; // gray
        }
      };

      return {
        id: `edge-${index}`,
        source: edge.source,
        target: edge.target,
        type: 'smoothstep',
        animated: false,
        style: {
          stroke: getEdgeColor(),
          strokeWidth: Math.max(1, Math.min(3, edge.confidence * 3)),
          opacity: 0.6,
        },
        label: edge.relationship_type.replace('_', ' '),
        labelStyle: {
          fill: getEdgeColor(),
          fontWeight: 500,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: getEdgeColor(),
        },
        data: {
          relationship_type: edge.relationship_type,
          confidence: edge.confidence,
        },
      };
    });
  }, [edges]);

  const [reactFlowNodes, setNodes, onNodesChange] = useNodesState(flowNodes);
  const [reactFlowEdges, setEdges, onEdgesChange] = useEdgesState(flowEdges);

  // Update nodes when layout or data changes
  useEffect(() => {
    setNodes(flowNodes);
  }, [flowNodes, setNodes]);

  // Update edges when data changes
  useEffect(() => {
    setEdges(flowEdges);
  }, [flowEdges, setEdges]);

  // Fit view when requested (handled by React Flow's fitView prop)

  // Handle node click
  const onNodeClickHandler = useCallback(
    (_: React.MouseEvent, node: Node) => {
      if (onNodeClick) {
        onNodeClick(node.id);
      }
    },
    [onNodeClick]
  );

  // Handle edge connection (for future drag-and-drop)
  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) => addEdge(params, eds));
    },
    [setEdges]
  );

  // Custom node component
  const CustomNode = ({ data, selected }: { data: any; selected: boolean }) => {
    const nodeType = data.type || 'document';
    const connections = data.connections || 0;

    return (
      <div
        className={`
          relative rounded-full
          bg-gradient-to-br
          ${nodeType === 'document' ? 'from-foldex-primary-500 to-foldex-primary-600' : ''}
          ${nodeType === 'entity' ? 'from-foldex-accent-500 to-foldex-accent-600' : ''}
          ${nodeType === 'chunk' ? 'from-gray-600 to-gray-700' : ''}
          ${selected ? 'ring-4 ring-foldex-primary-400 ring-offset-2 ring-offset-gray-900' : ''}
          shadow-lg hover:shadow-xl
          transition-all cursor-pointer
          flex items-center justify-center
          w-full h-full
        `}
      >
        <div className="text-center px-2">
          <div className="text-white font-semibold text-xs truncate max-w-[60px]">
            {data.label}
          </div>
          {connections > 0 && (
            <div className="text-white/70 text-[10px] mt-0.5">
              {connections} links
            </div>
          )}
        </div>
        {selected && (
          <div className="absolute -top-1 -right-1 w-4 h-4 bg-foldex-primary-400 rounded-full border-2 border-gray-900"></div>
        )}
      </div>
    );
  };

  const nodeTypes: NodeTypes = useMemo(
    () => ({
      custom: CustomNode as any,
    }),
    []
  );

  if (isLoading) {
    return (
      <div className="w-full h-full bg-gray-900 rounded-lg flex items-center justify-center border border-gray-700">
        <div className="text-center">
          <svg
            className="animate-spin h-8 w-8 text-foldex-primary-500 mx-auto mb-4"
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
          <p className="text-gray-400">Loading knowledge graph...</p>
        </div>
      </div>
    );
  }

  if (nodes.length === 0) {
    return (
      <div className="w-full h-full bg-gray-900 rounded-lg flex items-center justify-center border border-gray-700">
        <div className="text-center p-8">
          <svg
            className="w-16 h-16 text-gray-600 mx-auto mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
          <p className="text-gray-400 text-lg font-medium">No graph data available</p>
          <p className="text-gray-500 text-sm mt-2">
            Process a folder to see the knowledge graph
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-gray-900 rounded-lg border border-gray-700 relative overflow-hidden">
      <ReactFlow
        nodes={reactFlowNodes}
        edges={reactFlowEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClickHandler}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2, duration: 300 }}
        attributionPosition="bottom-left"
        className="bg-gray-900"
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#374151" gap={16} />
        <Controls
          className="bg-gray-800 border border-gray-700 rounded-lg [&_button]:bg-gray-700 [&_button]:border-gray-600 [&_button:hover]:bg-gray-600"
          showInteractive={false}
        />
        <MiniMap
          className="bg-gray-800 border border-gray-700 rounded-lg"
          nodeColor={(node) => {
            const type = node.data?.type || 'document';
            if (type === 'document') return '#3b82f6';
            if (type === 'entity') return '#a855f7';
            return '#6b7280';
          }}
          maskColor="rgba(0, 0, 0, 0.5)"
          pannable
          zoomable
        />
      </ReactFlow>

      {/* Graph Info Overlay */}
      <div className="absolute top-4 left-4 bg-gray-800/90 backdrop-blur-sm border border-gray-700 rounded-lg p-3 shadow-lg">
        <div className="flex items-center gap-2 mb-2">
          <svg
            className="w-4 h-4 text-foldex-primary-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
          <span className="text-sm font-semibold text-gray-200">Knowledge Graph</span>
        </div>
        <div className="text-xs text-gray-400 space-y-1">
          <div>{nodes.length} nodes</div>
          <div>{edges.length} relationships</div>
        </div>
      </div>
    </div>
  );
};

/**
 * KnowledgeGraph Component (with ReactFlowProvider wrapper)
 */
const KnowledgeGraph: React.FC<KnowledgeGraphProps> = (props) => {
  return (
    <ReactFlowProvider>
      <KnowledgeGraphInner {...props} />
    </ReactFlowProvider>
  );
};

export default KnowledgeGraph;
