/**
 * graph.ts
 * --------
 * A DERIVED projection of `LLMArchitecture` into nodes/edges suitable for
 * ELK.js layout + React Flow rendering: computed on demand from the schema.
 *
 * No positions (x/y) are produced here — that's ELK's job, downstream,
 * client-side. This module only decides WHAT nodes and edges exist and how
 * they nest, not WHERE they sit on screen.
 *
 * Every node carries a `path`: the same dotted-path convention `walk()`
 * uses to build `DiffEntry.path` in model.ts (e.g.
 * 'layerOrganization.layers.blocks.0.attention.mechanism'). That's the
 * single link between "a box in the diagram" and "an entry in the
 * encyclopedia / a diff row" — no separate id-to-path table to maintain.
 *
 * Repeated layers are NOT unrolled into one node per layer. A model like
 * GPT-3 (96 layers, 2 distinct blocks) renders as 2 block subgraphs inside
 * a single "layer stack" compound node, each annotated with how many
 * layers use it — mirroring the LayerPattern design (base.ts) that avoids
 * allocating N objects for N layers.
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
  /** Dotted path into the LLMArchitecture instance — see module doc. */
  path: string;
  /** ELK/React Flow compound nodes: nested structure, not a second graph. */
  children?: GraphNode[];
  /** Diagram-only annotations that aren't schema fields, e.g. repeat counts. */
  meta?: Record<string, unknown>;
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
  // alternation becomes ONE edge, not one per direction. Keying on the
  // UNORDERED pair (not `${from}->${to}`) matters here: a strictly
  // alternating pattern produces both a 0->1 and a 1->0 transition, and
  // adding both creates an actual cycle for ELK's layered algorithm, which
  // then has to arbitrarily break it — observed in practice as block-1
  // being placed above block-0 despite block-0 being layer 0. One edge
  // (direction = first occurrence) conveys "these alternate" without
  // giving ELK a cycle to resolve.
  if (layers.pattern !== 'uniform') {
    const seenPairs = new Set<string>();
    for (let i = 0; i < layers.pattern.length - 1; i++) {
      const from = layers.pattern[i];
      const to = layers.pattern[i + 1];
      if (from === undefined || to === undefined || from === to) continue;
      const pairKey = [from, to].sort((a, b) => a - b).join('-');
      if (seenPairs.has(pairKey)) continue;
      seenPairs.add(pairKey);
      edges.push({
        id: `block-transition-${from}-${to}`,
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
    const stepNode = buildSublayerStepNode(step, stepId, blockPath, block, edges);
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
  edges: GraphEdge[],
): GraphNode {
  const normalized = step.toLowerCase();

  if (normalized === 'attention') {
    const attentionPath = `${blockPath}.attention`;
    const mechanismId = `${stepId}-mechanism`;
    const patternId = `${stepId}-pattern`;
    const kernelId = `${stepId}-kernel`;

    // Attention's own 3 sub-choices are a chain too (mechanism -> pattern
    // -> kernel), not just siblings — previously these were only nested as
    // children with no edges between them, which is why they rendered as
    // disconnected dots.
    edges.push({ id: `${stepId}-edge-0`, source: mechanismId, target: patternId });
    edges.push({ id: `${stepId}-edge-1`, source: patternId, target: kernelId });

    return {
      id: stepId,
      label: 'Attention',
      category: 'attention',
      path: attentionPath,
      children: [
        {
          id: mechanismId,
          label: block.attention.mechanism.primitive,
          category: 'attention.mechanism',
          path: `${attentionPath}.mechanism`,
        },
        {
          id: patternId,
          label: block.attention.pattern.primitive,
          category: 'attention.pattern',
          path: `${attentionPath}.pattern`,
        },
        {
          id: kernelId,
          label: block.attention.kernel.primitive,
          category: 'attention.kernel',
          path: `${attentionPath}.kernel`,
        },
      ],
    };
  }

  if (normalized === 'ffn' || normalized === 'feedforward' || normalized === 'feed_forward') {
    const ffnPath = `${blockPath}.feedForward`;
    const activationId = `${stepId}-activation`;
    const gatingId = `${stepId}-gating`;

    const children: GraphNode[] = [
      {
        id: activationId,
        label: block.feedForward.config.activation.primitive,
        category: 'feedForward.activation',
        path: `${ffnPath}.config.activation`,
      },
      {
        id: gatingId,
        label: block.feedForward.config.gating.primitive,
        category: 'feedForward.gating',
        path: `${ffnPath}.config.gating`,
      },
    ];

    // Same fix as attention: activation -> gating -> moe (when present) is
    // a chain, not disconnected siblings.
    edges.push({ id: `${stepId}-edge-0`, source: activationId, target: gatingId });

    if (block.moe) {
      const moeId = `${stepId}-moe`;
      children.push({
        id: moeId,
        label: block.moe.primitive,
        category: 'moe',
        path: `${blockPath}.moe`,
      });
      edges.push({ id: `${stepId}-edge-1`, source: gatingId, target: moeId });
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