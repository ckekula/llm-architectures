import { useEffect, useMemo, useState } from 'react';
import { ReactFlow, Background, Controls, Handle, Position, type Node, type Edge } from '@xyflow/react';
import { buildArchitectureGraph, type GraphNode, type LLMArchitecture } from 'schema';
import { layoutArchitectureGraph } from '../lib/layout';

interface ArchitectureDiagramProps {
  architecture: LLMArchitecture;
}

interface NodeData extends Record<string, unknown> {
  label: string;
  category: GraphNode['category'];
  path: string;
}

/** Flattens the schema's nested GraphNode tree into id -> node lookups, since ELK's layout output is already flat (see layout.ts). */
function indexGraphNodes(nodes: GraphNode[], out: Map<string, GraphNode>): void {
  for (const node of nodes) {
    out.set(node.id, node);
    if (node.children) indexGraphNodes(node.children, out);
  }
}

const CATEGORY_COLOR: Partial<Record<GraphNode['category'], string>> = {
  layerStack: 'bg-slate-100 border-slate-300',
  block: 'bg-indigo-50 border-indigo-300',
  attention: 'bg-amber-50 border-amber-300',
  'attention.mechanism': 'bg-amber-100 border-amber-300',
  'attention.pattern': 'bg-amber-100 border-amber-300',
  'attention.kernel': 'bg-amber-100 border-amber-300',
  feedForward: 'bg-emerald-50 border-emerald-300',
  'feedForward.activation': 'bg-emerald-100 border-emerald-300',
  'feedForward.gating': 'bg-emerald-100 border-emerald-300',
  moe: 'bg-emerald-100 border-emerald-400',
  normalization: 'bg-sky-50 border-sky-300',
  residual: 'bg-rose-50 border-rose-300',
  tokenization: 'bg-slate-50 border-slate-300',
  embedding: 'bg-slate-50 border-slate-300',
  positionalEncoding: 'bg-slate-50 border-slate-300',
  outputHead: 'bg-slate-50 border-slate-300',
};

function ArchNode({ data }: { data: NodeData }) {
  return (
    <div className={`rounded-md border px-3 py-2 text-xs shadow-sm ${CATEGORY_COLOR[data.category] ?? 'bg-white border-gray-300'}`}>
      <Handle type="target" position={Position.Top} />
      <div className="font-medium text-gray-800">{data.label}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

const nodeTypes = { archNode: ArchNode };

export function ArchitectureDiagram({ architecture }: ArchitectureDiagramProps) {
  const graph = useMemo(() => buildArchitectureGraph(architecture), [architecture]);
  const [flowNodes, setFlowNodes] = useState<Node<NodeData>[]>([]);
  const [flowEdges, setFlowEdges] = useState<Edge[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const nodeById = new Map<string, GraphNode>();
    indexGraphNodes(graph.nodes, nodeById);

    layoutArchitectureGraph(graph)
      .then((positioned) => {
        if (cancelled) return;

        const nodes: Node<NodeData>[] = positioned.map((pos) => {
          const source = nodeById.get(pos.id);
          if (!source) {
            throw new Error(`Layout produced position for unknown node id "${pos.id}"`);
          }
          const isContainer = Boolean(source.children && source.children.length > 0);
          return {
            id: pos.id,
            position: { x: pos.x, y: pos.y },
            data: { label: source.label, category: source.category, path: source.path },
            parentId: pos.parentId,
            extent: pos.parentId ? ('parent' as const) : undefined,
            type: isContainer ? undefined : 'archNode',
            style: { width: pos.width, height: pos.height },
            ...(isContainer && {
              className: 'rounded-lg border border-dashed border-gray-300 bg-white/40',
            }),
          };
        });

        const edges: Edge[] = graph.edges.map((edge) => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: edge.label,
        }));

        setFlowNodes(nodes);
        setFlowEdges(edges);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to lay out diagram');
      });

    return () => {
      cancelled = true;
    };
  }, [graph]);

  if (error) {
    return <div className="p-4 text-sm text-red-600">Failed to render diagram: {error}</div>;
  }

  return (
    <div className="h-full w-full">
      <ReactFlow nodes={flowNodes} edges={flowEdges} nodeTypes={nodeTypes} fitView>
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
}