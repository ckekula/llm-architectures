/**
 * graph.ts
 * --------
 * A DERIVED projection of `LLMArchitecture` into nodes/edges suitable for
 * ELK.js layout + React Flow rendering.
 *
 * Every node carries a `path`: the same dotted-path convention `walk()`
 * uses to build `DiffEntry.path` in model.ts (e.g.
 * 'layerOrganization.layers.blocks.0.attention.mechanism'). That's the
 * single link between "a box in the diagram" and "an entry in the
 * encyclopedia / a diff row" — no separate id-to-path table to maintain.
 *
 * Repeated layers are NOT unrolled into one node per layer.
 */
import type { LLMArchitecture } from './model';
import type { LayerPattern } from './base';
import type { TransformerBlock } from './categories';

export interface GraphNode {
  id: string;
  label: string;
  /**
   * Which architectural category this node represents, for frontend
   * styling (icon/color per category) — not for layout logic.
   */
  category:
    | 'tokenization'
    | 'embedding'
    | 'positionalEncoding'
    | 'layerStack'
    | 'block'
    | 'attention'
    | 'attention.mechanism'
    | 'attention.pattern'
    | 'attention.kernel'
    | 'feedForward'
    | 'feedForward.activation'
    | 'feedForward.gating'
    | 'moe'
    | 'normalization'
    | 'residual'
    | 'outputHead'
    | 'unknown';
  path: string; // Dotted path into the LLMArchitecture instance
  children?: GraphNode[]; // ELK/React Flow compound nodes: nested structure, not a second graph.
  meta?: Record<string, unknown>; // Diagram-only annotations that aren't schema fields, e.g. repeat counts.
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
}

export interface ArchitectureGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export function buildArchitectureGraph(arch: LLMArchitecture): ArchitectureGraph {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const chain: string[] = []; // ids of top-level pipeline nodes, in order, for sequential edges

  nodes.push({ id: 'tokenization', label: 'Tokenization', category: 'tokenization', path: 'tokenization' });
  chain.push('tokenization');

  nodes.push({ id: 'embedding', label: 'Embedding', category: 'embedding', path: 'embedding' });
  chain.push('embedding');

  nodes.push({
    id: 'positionalEncoding',
    label: `Positional Encoding (${arch.positionalEncoding.primitive})`,
    category: 'positionalEncoding',
    path: 'positionalEncoding',
  });
  chain.push('positionalEncoding');

  const layerStack = buildLayerStackNode(arch.layerOrganization.layers, edges);
  nodes.push(layerStack);
  chain.push(layerStack.id);

  nodes.push({ id: 'outputHead', label: 'Output Head', category: 'outputHead', path: 'outputHead' });
  chain.push('outputHead');

  for (let i = 0; i < chain.length - 1; i++) {
    edges.push({ id: `pipeline-${i}`, source: chain[i]!, target: chain[i + 1]! });
  }

  return { nodes, edges };
}

/**
 * Groups layers by which distinct block (index into `blocks[]`) they use,
 * without unrolling per-layer nodes. Returns, per distinct block index, the
 * full list of layer indices that use it — enough for the frontend to
 * render a "×N" badge and, later, an expandable "which layers exactly"
 * detail without re-deriving it from the raw pattern array itself.
 */
function groupLayersByBlock(layers: LayerPattern<TransformerBlock>): Map<number, number[]> {
  const groups = new Map<number, number[]>();

  if (layers.pattern === 'uniform') {
    groups.set(
      0,
      Array.from({ length: layers.totalLayers }, (_, i) => i),
    );
    return groups;
  }

  for (let i = 0; i < layers.totalLayers; i++) {
    const blockIndex = layers.pattern[i];
    if (blockIndex === undefined) {
      throw new RangeError(`No pattern entry for layer ${i} (totalLayers=${layers.totalLayers}).`);
    }
    const indices = groups.get(blockIndex) ?? [];
    indices.push(i);
    groups.set(blockIndex, indices);
  }

  return groups;
}

function buildLayerStackNode(layers: LayerPattern<TransformerBlock>, edges: GraphEdge[]): GraphNode {
  const layersByBlock = groupLayersByBlock(layers);
  const distinctBlockIndices = [...layersByBlock.keys()].sort((a, b) => a - b);

  const children = distinctBlockIndices.map((blockIndex) => {
    const block = layers.blocks[blockIndex];
    if (!block) {
      throw new RangeError(`layers.blocks is missing index ${blockIndex} referenced by the pattern.`);
    }
    const layerIndices = layersByBlock.get(blockIndex) ?? [];
    return buildBlockNode(block, blockIndex, layerIndices, edges);
  });

  // Edges between distinct block types, in the order they actually
  // transition in the pattern (deduped) — e.g. GPT-3's dense/local-banded
  // alternation becomes one bidirectional pair of edges, not 95 of them.
  if (layers.pattern !== 'uniform') {
    const seenTransitions = new Set<string>();
    for (let i = 0; i < layers.pattern.length - 1; i++) {
      const from = layers.pattern[i];
      const to = layers.pattern[i + 1];
      if (from === undefined || to === undefined || from === to) continue;
      const key = `${from}->${to}`;
      if (seenTransitions.has(key)) continue;
      seenTransitions.add(key);
      edges.push({
        id: `block-transition-${key}`,
        source: `block-${from}`,
        target: `block-${to}`,
        label: 'alternates',
      });
    }
  }

  return {
    id: 'layerStack',
    label: `Layer Stack (${layers.totalLayers} layers)`,
    category: 'layerStack',
    path: 'layerOrganization.layers',
    children,
    meta: { totalLayers: layers.totalLayers, distinctBlockCount: distinctBlockIndices.length },
  };
}

function buildBlockNode(
  block: TransformerBlock,
  blockIndex: number,
  layerIndices: number[],
  edges: GraphEdge[],
): GraphNode {
  const blockId = `block-${blockIndex}`;
  const blockPath = `layerOrganization.layers.blocks.${blockIndex}`;
  const children: GraphNode[] = [];

  // Sublayer steps become chained child nodes, in the order the schema
  // itself already specifies (block.sublayerOrder) — the diagram's dataflow
  // follows authored data, it doesn't guess at execution order.
  let stepCounter = 0;
  let previousStepId: string | null = null;
  for (const step of block.sublayerOrder) {
    const stepId = `${blockId}-step-${stepCounter}`;
    const stepNode = buildSublayerStepNode(step, stepId, blockPath, block);
    children.push(stepNode);
    if (previousStepId) {
      edges.push({ id: `${blockId}-edge-${stepCounter}`, source: previousStepId, target: stepId });
    }
    previousStepId = stepId;
    stepCounter++;
  }

  return {
    id: blockId,
    label: `Block ${String.fromCharCode(65 + blockIndex)}`, // Block A, Block B, ...
    category: 'block',
    path: blockPath,
    children,
    meta: {
      occurrenceCount: layerIndices.length,
      layerIndices,
    },
  };
}

function buildSublayerStepNode(
  step: string,
  stepId: string,
  blockPath: string,
  block: TransformerBlock,
): GraphNode {
  const normalized = step.toLowerCase();

  if (normalized === 'attention') {
    const attentionPath = `${blockPath}.attention`;
    return {
      id: stepId,
      label: 'Attention',
      category: 'attention',
      path: attentionPath,
      children: [
        {
          id: `${stepId}-mechanism`,
          label: block.attention.mechanism.primitive,
          category: 'attention.mechanism',
          path: `${attentionPath}.mechanism`,
        },
        {
          id: `${stepId}-pattern`,
          label: block.attention.pattern.primitive,
          category: 'attention.pattern',
          path: `${attentionPath}.pattern`,
        },
        {
          id: `${stepId}-kernel`,
          label: block.attention.kernel.primitive,
          category: 'attention.kernel',
          path: `${attentionPath}.kernel`,
        },
      ],
    };
  }

  if (normalized === 'ffn' || normalized === 'feedforward' || normalized === 'feed_forward') {
    const ffnPath = `${blockPath}.feedForward`;
    const children: GraphNode[] = [
      {
        id: `${stepId}-activation`,
        label: block.feedForward.config.activation.primitive,
        category: 'feedForward.activation',
        path: `${ffnPath}.config.activation`,
      },
      {
        id: `${stepId}-gating`,
        label: block.feedForward.config.gating.primitive,
        category: 'feedForward.gating',
        path: `${ffnPath}.config.gating`,
      },
    ];
    if (block.moe) {
      children.push({
        id: `${stepId}-moe`,
        label: block.moe.primitive,
        category: 'moe',
        path: `${blockPath}.moe`,
      });
    }
    return { id: stepId, label: 'Feed-Forward', category: 'feedForward', path: ffnPath, children };
  }

  if (normalized === 'norm' || normalized === 'normalization') {
    return {
      id: stepId,
      label: block.normalization.primitive,
      category: 'normalization',
      path: `${blockPath}.normalization`,
    };
  }

  if (normalized === 'residual_add' || normalized === 'residual') {
    return {
      id: stepId,
      label: 'Residual Add',
      category: 'residual',
      path: `${blockPath}.residual`,
    };
  }

  // Forward-compatible fallback for sublayer step names we don't recognize
  // yet, rather than throwing — a new model shouldn't break the diagram.
  return { id: stepId, label: step, category: 'unknown', path: blockPath };
}