import { useEffect, useMemo, useState } from 'react';
import { ReactFlow, Controls, Handle, Position, MarkerType, type Node, type Edge } from '@xyflow/react';
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
  crossAttention: 'bg-violet-50 border-violet-300',
  'crossAttention.mechanism': 'bg-violet-100 border-violet-300',
  'crossAttention.pattern': 'bg-violet-100 border-violet-300',
  'crossAttention.kernel': 'bg-violet-100 border-violet-300',
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

function ArchNode({ id, data }: { id: string; data: NodeData }) {
  return (
    <div
      className={`relative h-full w-full box-border flex flex-col justify-center rounded-md border px-3 py-2 text-xs shadow-sm ${CATEGORY_COLOR[data.category] ?? 'bg-white border-gray-300'}`}
    >
      <Handle type="target" position={Position.Top} />
      <div className="font-medium text-gray-800">{data.label}</div>
      {DEBUG_LABELS && <div className="text-[9px] text-gray-400">{id}</div>}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function ContainerNode({ id, data }: { id: string; data: NodeData }) {
  return (
    <div className="relative h-full w-full box-border rounded-lg border border-dashed border-gray-300 bg-white/40">
      <Handle type="target" position={Position.Top} />
      <div className="h-12 flex items-center justify-between border-b border-dashed border-gray-300 px-3 text-xs font-medium text-gray-500">
        <span>{data.label}</span>
        {DEBUG_LABELS && <span className="text-[9px] text-gray-400">{id}</span>}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

const nodeTypes = { archNode: ArchNode, containerNode: ContainerNode };

// Toggle off to silence without removing the instrumentation.
const DEBUG = true;
const DEBUG_LABELS = true; // shows each node's id on-canvas, next to its label
function debugLog(label: string, payload: unknown): void {
  if (DEBUG) console.log(`[ArchitectureDiagram] ${label}`, payload);
}

export function ArchitectureDiagram({ architecture }: ArchitectureDiagramProps) {
  const graph = useMemo(() => {
    const g = buildArchitectureGraph(architecture);

    // Every node id anywhere in the tree (top-level + nested), so we can
    // catch an edge whose source/target doesn't exist ANYWHERE — this is
    // exactly the bug that made intra-Attention edges silently vanish.
    const allIds = new Set<string>();
    const collectIds = (nodes: GraphNode[]) => {
      for (const n of nodes) {
        allIds.add(n.id);
        if (n.children) collectIds(n.children);
      }
    };
    collectIds(g.nodes);
    const dangling = g.edges.filter((e) => !allIds.has(e.source) || !allIds.has(e.target));

    debugLog('1. buildArchitectureGraph()', {
      topLevelNodes: g.nodes.length,
      totalNodesIncludingNested: allIds.size,
      edgeCount: g.edges.length,
      edges: g.edges,
      danglingEdges: dangling, // should be [] — non-empty means graph.ts isn't wiring something up
    });

    return g;
  }, [architecture]);

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

        // ELK positions are PARENT-RELATIVE. Rather than eyeballing a
        // (possibly console-truncated) array, compute containment
        // directly: for every node with a parent, does its box actually
        // fit inside the parent's box? A violation here is the real
        // signature of "a dot rendered outside its container's boundary".
        const byId = new Map(positioned.map((p) => [p.id, p]));
        const violations = positioned
          .filter((p) => p.parentId)
          .map((p) => {
            const parent = byId.get(p.parentId!);
            if (!parent) return { child: p.id, issue: `parent "${p.parentId}" not found in layout output` };
            const overflowsRight = p.x + p.width > parent.width;
            const overflowsBottom = p.y + p.height > parent.height;
            const negative = p.x < 0 || p.y < 0;
            if (!overflowsRight && !overflowsBottom && !negative) return null;
            return {
              child: p.id,
              parent: p.parentId,
              childBox: { x: p.x, y: p.y, width: p.width, height: p.height },
              parentBox: { width: parent.width, height: parent.height },
              overflowsRight,
              overflowsBottom,
              negative,
            };
          })
          .filter((v): v is NonNullable<typeof v> => v !== null);

        debugLog('2. layoutArchitectureGraph() — full positions', positioned);
        debugLog('2b. containment violations (should be [])', violations);
        if (violations.length > 0) {
          console.warn('[ArchitectureDiagram] child node box exceeds its parent box:', violations);
        }

        const nodes: Node<NodeData>[] = positioned.map((pos) => {
          const source = nodeById.get(pos.id);
          if (!source) throw new Error(`Layout produced position for unknown node id "${pos.id}"`);
          const isContainer = Boolean(source.children && source.children.length > 0);
          return {
            id: pos.id,
            position: { x: pos.x, y: pos.y },
            data: { label: source.label, category: source.category, path: source.path },
            parentId: pos.parentId,
            extent: pos.parentId ? ('parent' as const) : undefined,
            type: isContainer ? 'containerNode' : 'archNode',
            style: { width: pos.width, height: pos.height },
          };
        });

        const edges: Edge[] = graph.edges.map((edge) => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: edge.label,
          style: { strokeWidth: 1.5, stroke: '#000000' },
          markerEnd: { type: MarkerType.ArrowClosed }
        }));

        // Final check: cross-reference every edge's endpoints against the
        // actual React Flow node id list. If an id passed step 1 (existed
        // in the schema graph) but fails here, the bug is in the
        // position-mapping step above, not in graph.ts.
        const flowNodeIds = new Set(nodes.map((n) => n.id));
        const brokenEdges = edges.filter((e) => !flowNodeIds.has(e.source) || !flowNodeIds.has(e.target));
        debugLog('3. final React Flow input', {
          nodeCount: nodes.length,
          edgeCount: edges.length,
          nodes,
          edges,
          brokenEdges, // should be [] — non-empty means an id got dropped between steps 1 and 3
        });
        if (brokenEdges.length > 0) {
          console.warn('[ArchitectureDiagram] edges with missing endpoints in final node list:', brokenEdges);
        }

        setFlowNodes(nodes);
        setFlowEdges(edges);
      })
      .catch((err: unknown) => {
        debugLog('layoutArchitectureGraph() threw', err);
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
        <Controls />
      </ReactFlow>
    </div>
  );
}