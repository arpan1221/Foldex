/**
 * Knowledge Graph Visualization Component
 * 
 * Displays a D3.js force-directed graph showing relationships
 * between documents, entities, and concepts.
 */

import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface Node {
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
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const simulationRef = useRef<d3.Simulation<Node, Link> | null>(null);

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

    const width = 1000;
    const height = 700;

    svg.attr('width', width).attr('height', height);

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

    // Draw links
    const link = svg
      .append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(graphData.links)
      .enter()
      .append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', 2);

    // Draw nodes
    const node = svg
      .append('g')
      .attr('class', 'nodes')
      .selectAll('circle')
      .data(graphData.nodes)
      .enter()
      .append('circle')
      .attr('r', (d) => radiusScale(d.node_type))
      .attr('fill', (d) => colorScale(d.type, d.node_type))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        if (onEntityClick) {
          onEntityClick(d);
        }
      })
      .call(
        d3
          .drag<SVGCircleElement, Node>()
          .on('start', (event, d) => {
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

    // Add labels
    const label = svg
      .append('g')
      .attr('class', 'labels')
      .selectAll('text')
      .data(graphData.nodes)
      .enter()
      .append('text')
      .text((d) => d.label.length > 20 ? d.label.substring(0, 20) + '...' : d.label)
      .attr('font-size', 10)
      .attr('dx', 15)
      .attr('dy', 4)
      .attr('fill', '#333')
      .style('pointer-events', 'none');

    // Add tooltips
    const tooltip = d3
      .select('body')
      .append('div')
      .style('position', 'absolute')
      .style('padding', '8px')
      .style('background', 'rgba(0, 0, 0, 0.8)')
      .style('color', 'white')
      .style('border-radius', '4px')
      .style('font-size', '12px')
      .style('pointer-events', 'none')
      .style('opacity', 0);

    node
      .on('mouseover', (event, d) => {
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
    simulation.on('tick', () => {
      link
        .attr('x1', (d) => (d.source as Node).x!)
        .attr('y1', (d) => (d.source as Node).y!)
        .attr('x2', (d) => (d.target as Node).x!)
        .attr('y2', (d) => (d.target as Node).y!);

      node.attr('cx', (d) => d.x!).attr('cy', (d) => d.y!);

      label.attr('x', (d) => d.x!).attr('y', (d) => d.y!);
    });

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

      <div className="border border-gray-700 rounded-lg overflow-hidden bg-white">
        <svg ref={svgRef} className="w-full"></svg>
      </div>

      <div className="mt-4 text-xs text-gray-500">
        <p>Click and drag nodes to reposition. Click on nodes to view details.</p>
      </div>
    </div>
  );
}

