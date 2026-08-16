import ELK from 'elkjs/lib/elk.bundled.js';
import type { ArchitectureGraph, GraphNode } from 'schema';

const elk = new ELK();

const LEAF_WIDTH = 200;
const LEAF_HEIGHT = 56;

interface ElkNode {
  id: string;
  width?: number;
  height?: number;
  children?: ElkNode[];
  layoutOptions?: Record<string, string>;
}

interface ElkLayoutResult extends ElkNode {
  x?: number;
  y?: number;
  children?: ElkLayoutResult[];
}

export interface PositionedNode {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  /** ELK gives child coordinates relative to their parent container; React Flow's parentId + relative position matches this directly. */
  parentId?: string;
}

/**
 * `buildArchitectureGraph` (schema) never nests one leaf under itself as
 * both container and leaf — every node either has children or doesn't —
 * so a compound node (`children.length > 0`) always gets ELK's layered
 * algorithm applied to its own children, independent of its parent's
 * layout direction.
 *
 * Direction defaults to DOWN unless the node opts into `layoutDirection: 'horizontal'`
 *
 * Top padding is reserved for the header bar `ContainerNode` renders
 */
function toElkNode(node: GraphNode): ElkNode {
  if (node.children && node.children.length > 0) {
    const hasHeader = node.label !== '';
    const isStacksGroup = node.category === 'stackGroup';

    return {
      id: node.id,
      children: node.children.map(toElkNode),
      layoutOptions: {
        'elk.algorithm': 'layered',
        'elk.direction': node.layoutDirection === 'horizontal' ? 'RIGHT' : 'DOWN',
        'elk.padding': isStacksGroup ? '[top=40,left=24,bottom=40,right=24]'
          : hasHeader ? '[top=48,left=24,bottom=48,right=24]'
          : '[top=24,left=24,bottom=24,right=24]',
        'elk.spacing.nodeNode': '24',
      },
    };
  }
  return { id: node.id, width: LEAF_WIDTH, height: LEAF_HEIGHT };
}

function flattenPositions(elkNode: ElkLayoutResult, parentId: string | undefined, out: PositionedNode[]): void {
  if (elkNode.id !== 'root') {
    out.push({
      id: elkNode.id,
      x: elkNode.x ?? 0,
      y: elkNode.y ?? 0,
      width: elkNode.width ?? LEAF_WIDTH,
      height: elkNode.height ?? LEAF_HEIGHT,
      parentId,
    });
  }
  const nextParentId = elkNode.id === 'root' ? undefined : elkNode.id;
  for (const child of elkNode.children ?? []) {
    flattenPositions(child, nextParentId, out);
  }
}

export async function layoutArchitectureGraph(graph: ArchitectureGraph): Promise<PositionedNode[]> {
  const elkGraph: ElkNode = {
    id: 'root',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'DOWN',
      'elk.spacing.nodeNode': '40',
      'elk.layered.spacing.nodeNodeBetweenLayers': '80',
    },
    children: graph.nodes.map(toElkNode),
  };

  // ELK's typings don't include `edges` on the plain node shape we build
  // above; it's a valid top-level graph field, so it's attached here.
  const elkGraphWithEdges = {
    ...elkGraph,
    edges: graph.edges.map((edge) => ({ id: edge.id, sources: [edge.source], targets: [edge.target] })),
  };

  const layouted = (await elk.layout(elkGraphWithEdges)) as ElkLayoutResult;

  const positioned: PositionedNode[] = [];
  flattenPositions(layouted, undefined, positioned);
  return positioned;
}