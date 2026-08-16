import { useEffect, useMemo, useState } from 'react';
import { ReactFlow, Controls, Handle, Position, MarkerType, type Node, type Edge } from '@xyflow/react';
import { buildArchitectureGraph, type GraphNode, type LLMArchitecture } from 'schema';
import { layoutArchitectureGraph } from '../lib/layout';


// Toggle off to silence without removing the instrumentation.
const DEBUG = false;
const DEBUG_LABELS = false;
function debugLog(label: string, payload: unknown): void {
  if (DEBUG) console.log(`[ArchitectureDiagram] ${label}`, payload);
}

interface ArchitectureDiagramProps {
  architecture: LLMArchitecture;
}

interface NodeData extends Record<string, unknown> {
  label: string;
  category: GraphNode['category'];
  path: string;
  isExpandable: boolean;
  isExpanded: boolean;
  collapsedBackgroundColor?: string;
  collapsedBorderColor?: string;
}

interface TintColor {
  bg: string;
  border: string;
}

function indexGraphNodes(nodes: GraphNode[], out: Map<string, GraphNode>): void {
  for (const node of nodes) {
    out.set(node.id, node);
    if (node.children) indexGraphNodes(node.children, out);
  }
}

function collectGraphIds(nodes: GraphNode[], out: Set<string>): void {
  for (const node of nodes) {
    out.add(node.id);
    if (node.children) collectGraphIds(node.children, out);
  }
}

function collectCollapsibleNodeIds(nodes: GraphNode[], out: Set<string>, inBlockSubtree = false): void {
  for (const node of nodes) {
    const hasChildren = Boolean(node.children && node.children.length > 0);
    const nextInBlockSubtree = inBlockSubtree || node.category === 'block';
    const isCollapsible = hasChildren && (node.category === 'block' || inBlockSubtree);

    if (isCollapsible) out.add(node.id);
    if (node.children) collectCollapsibleNodeIds(node.children, out, nextInBlockSubtree);
  }
}

function filterNodesByExpansion(nodes: GraphNode[], expandedNodeIds: Set<string>, inBlockSubtree = false): GraphNode[] {
  return nodes.map((node) => {
    if (!node.children || node.children.length === 0) return node;

    const nextInBlockSubtree = inBlockSubtree || node.category === 'block';
    const isCollapsible = node.category === 'block' || inBlockSubtree;
    const showChildren = !isCollapsible || expandedNodeIds.has(node.id);

    return {
      ...node,
      children: showChildren ? filterNodesByExpansion(node.children, expandedNodeIds, nextInBlockSubtree) : undefined,
    };
  });
}

const CONSTITUENT_TINT: Partial<Record<GraphNode['category'], TintColor>> = {
  attention: { bg: '#fef3c7', border: '#f59e0b' },
  'attention.mechanism': { bg: '#fef3c7', border: '#f59e0b' },
  'attention.pattern': { bg: '#fef3c7', border: '#f59e0b' },
  'attention.kernel': { bg: '#fef3c7', border: '#f59e0b' },
  crossAttention: { bg: '#ede9fe', border: '#8b5cf6' },
  'crossAttention.mechanism': { bg: '#ede9fe', border: '#8b5cf6' },
  'crossAttention.pattern': { bg: '#ede9fe', border: '#8b5cf6' },
  'crossAttention.kernel': { bg: '#ede9fe', border: '#8b5cf6' },
  feedForward: { bg: '#d1fae5', border: '#10b981' },
  'feedForward.activation': { bg: '#d1fae5', border: '#10b981' },
  'feedForward.gating': { bg: '#d1fae5', border: '#10b981' },
  moe: { bg: '#a7f3d0', border: '#059669' },
  normalization: { bg: '#e0f2fe', border: '#0ea5e9' },
  residual: { bg: '#ffe4e6', border: '#f43f5e' },
  tokenization: { bg: '#f8fafc', border: '#94a3b8' },
  embedding: { bg: '#f8fafc', border: '#94a3b8' },
  positionalEncoding: { bg: '#f8fafc', border: '#94a3b8' },
  outputHead: { bg: '#f8fafc', border: '#94a3b8' },
};

const NON_CONSTITUENT_CATEGORIES: Set<GraphNode['category']> = new Set(['layerStack', 'block', 'stackGroup', 'unknown']);

function findFirstConstituentTint(node: GraphNode): TintColor | undefined {
  if (!node.children || node.children.length === 0) {
    if (NON_CONSTITUENT_CATEGORIES.has(node.category)) return undefined;
    return CONSTITUENT_TINT[node.category];
  }

  for (const child of node.children) {
    const tint = findFirstConstituentTint(child);
    if (tint) return tint;
  }

  return undefined;
}

function buildCollapsedNodeColors(nodes: GraphNode[], out: Map<string, { bg: string; border: string }>): void {
  for (const node of nodes) {
    if (node.children && node.children.length > 0) {
      const tint = findFirstConstituentTint(node);
      if (tint) {
        out.set(node.id, {
          bg: tint.bg,
          border: tint.border,
        });
      }
      buildCollapsedNodeColors(node.children, out);
    }
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
  // A pure layout grouping has no label and renders no visible box at all
  // It exists purely to give ELK something to apply layoutDirection:'horizontal' to,
  // It's top padding is already reduced to match so this doesn't leave a dead gap.
  if (data.label === '') {
    if (data.category === 'stackGroup') {
      return (
        <div className="relative h-full w-full box-border rounded-lg border border-slate-400 bg-slate-50/70">
          <Handle type="target" position={Position.Top} />
          {DEBUG_LABELS && <span className="absolute left-1 top-1 text-[9px] text-gray-400">{id}</span>}
          <Handle type="source" position={Position.Bottom} />
        </div>
      );
    }

    return (
      <div className="relative h-full w-full">
        <Handle type="target" position={Position.Top} />
        {DEBUG_LABELS && <span className="absolute left-1 top-1 text-[9px] text-gray-400">{id}</span>}
        <Handle type="source" position={Position.Bottom} />
      </div>
    );
  }

  return (
    <div
      className={`relative h-full w-full box-border rounded-lg border border-dashed border-gray-700 bg-white/40 ${data.isExpandable ? 'cursor-pointer' : ''}`}
      style={
        data.isExpandable && !data.isExpanded && data.collapsedBackgroundColor && data.collapsedBorderColor
          ? { backgroundColor: data.collapsedBackgroundColor, borderColor: data.collapsedBorderColor }
          : undefined
      }
    >
      <Handle type="target" position={Position.Top} />
      <div className="h-12 flex items-center justify-between px-3 text-xs font-medium text-gray-800">
        <span>{data.label}</span>
        {DEBUG_LABELS && <span className="text-[9px] text-gray-400">{id}</span>}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

const nodeTypes = { archNode: ArchNode, containerNode: ContainerNode };

export function ArchitectureDiagram({ architecture }: ArchitectureDiagramProps) {
  const fullGraph = useMemo(() => {
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

  const fullNodeById = useMemo(() => {
    const index = new Map<string, GraphNode>();
    indexGraphNodes(fullGraph.nodes, index);
    return index;
  }, [fullGraph]);

  const collapsibleNodeIds = useMemo(() => {
    const ids = new Set<string>();
    collectCollapsibleNodeIds(fullGraph.nodes, ids);
    return ids;
  }, [fullGraph]);

  const [expandedNodeIds, setExpandedNodeIds] = useState<Set<string>>(new Set());

  const collapsedNodeColors = useMemo(() => {
    const colors = new Map<string, { bg: string; border: string }>();
    buildCollapsedNodeColors(fullGraph.nodes, colors);
    return colors;
  }, [fullGraph]);

  const graph = useMemo(() => {
    const nodes = filterNodesByExpansion(fullGraph.nodes, expandedNodeIds);
    const visibleIds = new Set<string>();
    collectGraphIds(nodes, visibleIds);
    const edges = fullGraph.edges.filter((e) => visibleIds.has(e.source) && visibleIds.has(e.target));

    debugLog('1b. visibility after collapse/expand', {
      expandedNodeIds: [...expandedNodeIds],
      visibleNodeCount: visibleIds.size,
      visibleEdgeCount: edges.length,
    });

    return { nodes, edges };
  }, [expandedNodeIds, fullGraph]);

  const [flowNodes, setFlowNodes] = useState<Node<NodeData>[]>([]);
  const [flowEdges, setFlowEdges] = useState<Edge[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

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
          const source = fullNodeById.get(pos.id);
          if (!source) throw new Error(`Layout produced position for unknown node id "${pos.id}"`);
          const hasChildren = Boolean(source.children && source.children.length > 0);
          const isExpandable = hasChildren && collapsibleNodeIds.has(source.id);
          const isExpanded = !isExpandable || expandedNodeIds.has(source.id);
          const collapsedColors = collapsedNodeColors.get(source.id);
          return {
            id: pos.id,
            position: { x: pos.x, y: pos.y },
            data: {
              label: source.label,
              category: source.category,
              path: source.path,
              isExpandable,
              isExpanded,
              collapsedBackgroundColor: collapsedColors?.bg,
              collapsedBorderColor: collapsedColors?.border,
            },
            parentId: pos.parentId,
            extent: pos.parentId ? ('parent' as const) : undefined,
            type: hasChildren ? 'containerNode' : 'archNode',
            style: { width: pos.width, height: pos.height },
          };
        });

        const edges: Edge[] = graph.edges.map((edge) => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: edge.label,
          style: { strokeWidth: 1.5, stroke: '#000000' },
          markerEnd: { type: MarkerType.ArrowClosed },
          type: 'smoothstep',
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

        setError(null);
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
  }, [collapsedNodeColors, collapsibleNodeIds, expandedNodeIds, fullNodeById, graph]);

  const handleNodeClick = (_event: React.MouseEvent, node: Node<NodeData>) => {
    if (!node.data.isExpandable) return;

    setExpandedNodeIds((prev) => {
      const next = new Set(prev);
      if (next.has(node.id)) {
        next.delete(node.id);
      } else {
        next.add(node.id);
      }
      return next;
    });
  };

  if (error) {
    return <div className="p-4 text-sm text-red-600">Failed to render diagram: {error}</div>;
  }

  return (
    <div className="h-full w-full">
      <ReactFlow nodes={flowNodes} edges={flowEdges} nodeTypes={nodeTypes} onNodeClick={handleNodeClick} fitView>
        <Controls />
      </ReactFlow>
    </div>
  );
}