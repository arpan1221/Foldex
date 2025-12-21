/**
 * Knowledge Graph Visualization Component
 * 
 * Displays a D3.js force-directed graph showing relationships
 * between documents, entities, and concepts.
 */

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface Node extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  type: string;
  node_type: string;
}

interface Link {
  source: string | Node;
  target: string | Node;
  relation: string;
}

interface GraphData {
  nodes: Node[];
  links: Link[];
  stats?: {
    node_count: number;
    link_count: number;
    document_count: number;
  };
}

interface KnowledgeGraphVizProps {
  folderId: string;
  onEntityClick?: (entity: Node) => void;
}

export function KnowledgeGraphViz({ folderId, onEntityClick }: KnowledgeGraphVizProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const simulationRef = useRef<d3.Simulation<Node, Link> | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);

  useEffect(() => {
    if (!folderId) return;

    // Fetch graph data
    const fetchGraphData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const token = localStorage.getItem('access_token');
        const response = await fetch(
          `http://localhost:8000/api/v1/knowledge-graph/graph/${folderId}`,
          {
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
          }
        );

        if (!response.ok) {
          throw new Error(`Failed to fetch graph: ${response.statusText}`);
        }

        const data = await response.json();
        setGraphData(data);
      } catch (err) {
        console.error('Error fetching knowledge graph:', err);
        setError(err instanceof Error ? err.message : 'Failed to load graph');
      } finally {
        setLoading(false);
      }
    };

    fetchGraphData();
  }, [folderId]);

  useEffect(() => {
    if (!graphData || !svgRef.current || loading) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // Clear previous render

    // Get container dimensions
    const container = containerRef.current;
    const width = container ? container.clientWidth : 1000;
    const height = container ? container.clientHeight : 700;

    svg.attr('width', width).attr('height', height);

    // Create zoom behavior
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
        setZoomLevel(event.transform.k);
      });

    zoomRef.current = zoom;
    svg.call(zoom);

    // Create container group for zoomable content
    const g = svg.append('g');

    // Create color scale
    const colorScale = (type: string, nodeType: string) => {
      if (nodeType === 'document') return '#4CAF50'; // Green
      switch (type) {
        case 'person': return '#2196F3'; // Blue
        case 'algorithm': return '#FF9800'; // Orange
        case 'concept': return '#9C27B0'; // Purple
        case 'dataset': return '#F44336'; // Red
        default: return '#757575'; // Grey
      }
    };

    // Create radius scale
    const radiusScale = (nodeType: string) => {
      return nodeType === 'document' ? 12 : 8;
    };

    // Create simulation
    const simulation = d3
      .forceSimulation<Node>(graphData.nodes)
      .force(
        'link',
        d3
          .forceLink<Node, Link>(graphData.links)
          .id((d) => d.id)
          .distance(100)
      )
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(20));

    simulationRef.current = simulation;

    // Draw links (in dark mode)
    const link = g
      .append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(graphData.links)
      .enter()
      .append('line')
      .attr('stroke', '#6b7280') // Gray for dark mode
      .attr('stroke-opacity', 0.4)
      .attr('stroke-width', 1.5);

    // Draw nodes (with dark mode styling)
    const node = g
      .append('g')
      .attr('class', 'nodes')
      .selectAll('circle')
      .data(graphData.nodes)
      .enter()
      .append('circle')
      .attr('r', (d) => radiusScale(d.node_type))
      .attr('fill', (d) => colorScale(d.type, d.node_type))
      .attr('stroke', '#1f2937') // Dark gray border for dark mode
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation(); // Prevent zoom on click
        if (onEntityClick) {
          onEntityClick(d);
        }
      })
      .call(
        d3
          .drag<SVGCircleElement, Node>()
          .on('start', (event, d) => {
            event.sourceEvent.stopPropagation();
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    // Add labels (dark mode colors)
    const label = g
      .append('g')
      .attr('class', 'labels')
      .selectAll('text')
      .data(graphData.nodes)
      .enter()
      .append('text')
      .text((d) => d.label.length > 20 ? d.label.substring(0, 20) + '...' : d.label)
      .attr('font-size', 11)
      .attr('dx', 15)
      .attr('dy', 4)
      .attr('fill', '#e5e7eb') // Light gray text for dark mode
      .style('pointer-events', 'none')
      .style('font-family', 'system-ui, -apple-system, sans-serif');

    // Add tooltips (dark mode styling)
    const tooltip = d3
      .select('body')
      .append('div')
      .style('position', 'absolute')
      .style('padding', '10px 12px')
      .style('background', '#1f2937')
      .style('color', '#e5e7eb')
      .style('border', '1px solid #374151')
      .style('border-radius', '6px')
      .style('font-size', '12px')
      .style('pointer-events', 'none')
      .style('opacity', 0)
      .style('z-index', 1000)
      .style('box-shadow', '0 4px 6px rgba(0, 0, 0, 0.3)');

    node
      .on('mouseover', (_event, d) => {
        tooltip
          .style('opacity', 1)
          .html(`<strong>${d.label}</strong><br/>Type: ${d.type}`);
      })
      .on('mousemove', (event) => {
        tooltip
          .style('left', event.pageX + 10 + 'px')
          .style('top', event.pageY - 10 + 'px');
      })
      .on('mouseout', () => {
        tooltip.style('opacity', 0);
      });

    // Update positions on simulation tick
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    simulation.on('tick', () => {
      link
        .attr('x1', (d) => (d.source as Node).x!)
        .attr('y1', (d) => (d.source as Node).y!)
        .attr('x2', (d) => (d.target as Node).x!)
        .attr('y2', (d) => (d.target as Node).y!);

      node.attr('cx', (d) => d.x!).attr('cy', (d) => d.y!);

      label.attr('x', (d) => d.x!).attr('y', (d) => d.y!);
    });

    // Initialize zoom to fit content
    const bounds = g.node()?.getBBox();
    if (bounds && bounds.width && bounds.height) {
      const fullWidth = bounds.width;
      const fullHeight = bounds.height;
      const midX = bounds.x + fullWidth / 2;
      const midY = bounds.y + fullHeight / 2;
      const scale = Math.min(width / fullWidth, height / fullHeight, 1) * 0.9;
      const translate = [width / 2 - scale * midX, height / 2 - scale * midY];
      svg.call(
        zoom.transform,
        d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
      );
    }

    // Cleanup on unmount
    return () => {
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
      tooltip.remove();
    };
  }, [graphData, loading, onEntityClick]);

  if (loading) {
    return (
      <div className="knowledge-graph-container p-6">
        <div className="flex items-center justify-center h-96">
          <div className="text-gray-400">Loading knowledge graph...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="knowledge-graph-container p-6">
        <div className="flex items-center justify-center h-96">
          <div className="text-red-400">Error: {error}</div>
        </div>
      </div>
    );
  }

  if (!graphData || graphData.nodes.length === 0) {
    return (
      <div className="knowledge-graph-container p-6">
        <div className="flex items-center justify-center h-96">
          <div className="text-gray-400">
            No knowledge graph data available. Process a folder first.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="knowledge-graph-container p-6 bg-gray-900 rounded-lg">
      <div className="mb-4">
        <h2 className="text-2xl font-bold text-white mb-2">Knowledge Graph</h2>
        {graphData.stats && (
          <div className="text-sm text-gray-400 mb-4">
            {graphData.stats.node_count} nodes • {graphData.stats.link_count} links •{' '}
            {graphData.stats.document_count} documents
          </div>
        )}
      </div>

      <div className="flex gap-4 mb-4">
        <div className="legend flex flex-wrap gap-4 text-sm">
          <div className="legend-item flex items-center gap-2">
            <span
              className="dot w-3 h-3 rounded-full"
              style={{ backgroundColor: '#4CAF50' }}
            ></span>
            <span className="text-gray-300">Document</span>
          </div>
          <div className="legend-item flex items-center gap-2">
            <span
              className="dot w-3 h-3 rounded-full"
              style={{ backgroundColor: '#2196F3' }}
            ></span>
            <span className="text-gray-300">Person</span>
          </div>
          <div className="legend-item flex items-center gap-2">
            <span
              className="dot w-3 h-3 rounded-full"
              style={{ backgroundColor: '#FF9800' }}
            ></span>
            <span className="text-gray-300">Algorithm</span>
          </div>
          <div className="legend-item flex items-center gap-2">
            <span
              className="dot w-3 h-3 rounded-full"
              style={{ backgroundColor: '#9C27B0' }}
            ></span>
            <span className="text-gray-300">Concept</span>
          </div>
          <div className="legend-item flex items-center gap-2">
            <span
              className="dot w-3 h-3 rounded-full"
              style={{ backgroundColor: '#F44336' }}
            ></span>
            <span className="text-gray-300">Dataset</span>
          </div>
        </div>
      </div>

      <div className="relative border border-gray-700 rounded-lg overflow-hidden bg-gray-900">
        {/* Zoom Controls */}
        <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
          <button
            onClick={() => {
              if (svgRef.current && zoomRef.current) {
                d3.select(svgRef.current).transition().call(
                  zoomRef.current.scaleBy,
                  1.2
                );
              }
            }}
            className="px-3 py-2 bg-gray-800 hover:bg-gray-700 text-gray-200 rounded-lg border border-gray-600 transition-colors"
            title="Zoom In"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v6m3-3H7" />
            </svg>
          </button>
          <button
            onClick={() => {
              if (svgRef.current && zoomRef.current) {
                d3.select(svgRef.current).transition().call(
                  zoomRef.current.scaleBy,
                  1 / 1.2
                );
              }
            }}
            className="px-3 py-2 bg-gray-800 hover:bg-gray-700 text-gray-200 rounded-lg border border-gray-600 transition-colors"
            title="Zoom Out"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
            </svg>
          </button>
          <button
            onClick={() => {
              if (svgRef.current && zoomRef.current && graphData) {
                const svgEl = svgRef.current;
                const width = svgEl.clientWidth || 1000;
                const height = svgEl.clientHeight || 700;
                const bounds = { x: -width/4, y: -height/4, width: width*1.5, height: height*1.5 };
                const fullWidth = bounds.width;
                const fullHeight = bounds.height;
                const midX = bounds.x + fullWidth / 2;
                const midY = bounds.y + fullHeight / 2;
                const scale = Math.min(width / fullWidth, height / fullHeight, 1) * 0.9;
                const translate = [width / 2 - scale * midX, height / 2 - scale * midY];
                d3.select(svgEl).transition().duration(750).call(
                  zoomRef.current.transform,
                  d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
                );
              }
            }}
            className="px-3 py-2 bg-gray-800 hover:bg-gray-700 text-gray-200 rounded-lg border border-gray-600 transition-colors"
            title="Fit to Screen"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
            </svg>
          </button>
        </div>

        {/* Zoom Level Display */}
        {zoomLevel !== 1 && (
          <div className="absolute top-4 left-4 z-10 px-3 py-1 bg-gray-800 text-gray-300 text-xs rounded-lg border border-gray-600">
            {Math.round(zoomLevel * 100)}%
          </div>
        )}

        <div ref={containerRef} className="w-full h-[700px]">
          <svg ref={svgRef} className="w-full h-full cursor-move"></svg>
        </div>
      </div>

      <div className="mt-4 text-xs text-gray-400">
        <p>Click and drag nodes to reposition. Use mouse wheel or buttons to zoom. Click on nodes to view details.</p>
      </div>
    </div>
  );
}

