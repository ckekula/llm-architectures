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
 * 'stacks.decoder.layers.blocks.0.attention.mechanism'). That's the
 * single link between "a box in the diagram" and "an entry in the
 * encyclopedia / a diff row" — no separate id-to-path table to maintain.
 *
 * Repeated layers are NOT unrolled into one node per layer. A model like
 * GPT-3 (96 layers, 2 distinct blocks) renders as 2 block subgraphs inside
 * a single stack compound node, each annotated with how many layers use
 * it — mirroring the LayerPattern design (base.ts) that avoids allocating
 * N objects for N layers.
 *
 * A model can have an encoder stack, a decoder stack, or both.
 * Encoder-decoder models get two stack subgraphs, linked by an "encoder output"
 * edge representing the data dependency cross-attention introduces
 */
import type { LLMArchitecture } from './model';
import type { LayerPattern } from './base';
import type { Attention, TransformerBlock } from './categories';

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
    | 'crossAttention'
    | 'crossAttention.mechanism'
    | 'crossAttention.pattern'
    | 'crossAttention.kernel'
    | 'feedForward'
    | 'feedForward.activation'
    | 'feedForward.gating'
    | 'moe'
    | 'normalization'
    | 'residual'
    | 'outputHead'
    | 'stackGroup'
    | 'unknown';
  /** Dotted path into the LLMArchitecture instance — see module doc. */
  path: string;
  /** ELK/React Flow compound nodes: nested structure, not a second graph. */
  children?: GraphNode[];
  /**
   * How THIS node's own children should be arranged relative to each
   * other. Defaults to 'vertical'. 'horizontal' is for children
   * that don't have a real execution-order dependency on each other.
   */
  layoutDirection?: 'horizontal' | 'vertical';
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

type StackName = 'encoder' | 'decoder';

export function buildArchitectureGraph(arch: LLMArchitecture): ArchitectureGraph {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];

  const tokenizationNode: GraphNode = { id: 'tokenization', label: 'Tokenization', category: 'tokenization', path: 'tokenization' };
  const embeddingNode: GraphNode = { id: 'embedding', label: 'Embedding', category: 'embedding', path: 'embedding' };
  const positionalEncodingNode: GraphNode = {
    id: 'positionalEncoding',
    label: `Positional Encoding (${arch.positionalEncoding.primitive})`,
    category: 'positionalEncoding',
    path: 'positionalEncoding',
  };
  const outputHeadNode: GraphNode = { id: 'outputHead', label: 'Output Head', category: 'outputHead', path: 'outputHead' };

  nodes.push(tokenizationNode, embeddingNode, positionalEncodingNode);
  edges.push(
    { id: 'pipeline-0', source: tokenizationNode.id, target: embeddingNode.id },
    { id: 'pipeline-1', source: embeddingNode.id, target: positionalEncodingNode.id },
  );

  const encoderStack = arch.stacks.encoder ? buildStackNode('encoder', arch.stacks.encoder.layers, edges) : undefined;
  const decoderStack = arch.stacks.decoder ? buildStackNode('decoder', arch.stacks.decoder.layers, edges) : undefined;

  // The node that represents "the stacks" at the TOP level of the
  // pipeline — the stacksGroup wrapper when both exist, otherwise
  // whichever single stack exists. Pipeline edges connect to THIS.
  let topLevelStacksNode: GraphNode | undefined;

  if (encoderStack && decoderStack) {
    edges.push({
      id: 'pipeline-encoder-decoder',
      source: encoderStack.id,
      target: decoderStack.id,
      label: 'encoder output',
    });

    // Without this wrapper, the encoder->decoder edge above gives ELK's
    // layered algorithm a real dependency, so it ranks them into
    // different vertical layers.
    const stacksGroup: GraphNode = {
      id: 'stacksGroup',
      label: '', // purely a layout grouping, not a real architectural component — no header text
      category: 'stackGroup',
      path: 'stacks',
      layoutDirection: 'horizontal',
      children: [encoderStack, decoderStack],
    };
    nodes.push(stacksGroup);
    topLevelStacksNode = stacksGroup;
  } else if (encoderStack) {
    nodes.push(encoderStack);
    topLevelStacksNode = encoderStack;
  } else if (decoderStack) {
    nodes.push(decoderStack);
    topLevelStacksNode = decoderStack;
  }

  if (topLevelStacksNode) {
    edges.push({ id: 'pipeline-pe-stacks', source: positionalEncodingNode.id, target: topLevelStacksNode.id });
  }

  nodes.push(outputHeadNode);
  if (topLevelStacksNode) {
    edges.push({ id: 'pipeline-final-output', source: topLevelStacksNode.id, target: outputHeadNode.id });
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

function buildStackNode(stackName: StackName, layers: LayerPattern<TransformerBlock>, edges: GraphEdge[]): GraphNode {
  const layersByBlock = groupLayersByBlock(layers);
  const distinctBlockIndices = [...layersByBlock.keys()].sort((a, b) => a - b);
  const stackId = `${stackName}Stack`;
  const stackLabel = stackName === 'encoder' ? 'Encoder Stack' : 'Decoder Stack';

  const children = distinctBlockIndices.map((blockIndex) => {
    const block = layers.blocks[blockIndex];
    if (!block) {
      throw new RangeError(`stacks.${stackName}.layers.blocks is missing index ${blockIndex} referenced by the pattern.`);
    }
    const layerIndices = layersByBlock.get(blockIndex) ?? [];
    return buildBlockNode(stackName, block, blockIndex, layerIndices, edges);
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
        id: `${stackName}-block-transition-${from}-${to}`,
        source: `${stackName}-block-${from}`,
        target: `${stackName}-block-${to}`,
        label: 'alternates',
      });
    }
  }

  return {
    id: stackId,
    label: `${stackLabel} (${layers.totalLayers} layers)`,
    category: 'layerStack',
    path: `stacks.${stackName}.layers`,
    // Distinct block TYPES alternate across depth, they don't run in parallel.
    layoutDirection: 'horizontal',
    children,
    meta: { stackName, totalLayers: layers.totalLayers, distinctBlockCount: distinctBlockIndices.length },
  };
}

function buildBlockNode(
  stackName: StackName,
  block: TransformerBlock,
  blockIndex: number,
  layerIndices: number[],
  edges: GraphEdge[],
): GraphNode {
  const blockId = `${stackName}-block-${blockIndex}`;
  const blockPath = `stacks.${stackName}.layers.blocks.${blockIndex}`;
  const labelPrefix = stackName === 'encoder' ? 'Encoder' : 'Decoder';
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
    label: `${labelPrefix} Block ${String.fromCharCode(65 + blockIndex)}`, // Encoder Block A, Decoder Block A, ...
    category: 'block',
    path: blockPath,
    children,
    meta: {
      stackName,
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

  if (normalized === 'attention' || normalized === 'self_attention') {
    return buildAttentionNode(block.attention, 'Self-Attention', 'attention', stepId, blockPath, edges);
  }

  if (normalized === 'cross_attention' || normalized === 'crossattention') {
    if (!block.crossAttention) {
      throw new Error(
        `sublayerOrder references "cross_attention" at ${stepId}, but this block has no crossAttention configured.`,
      );
    }
    return buildAttentionNode(block.crossAttention, 'Cross-Attention', 'crossAttention', stepId, blockPath, edges);
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

/**
 * Shared by self-attention and cross-attention: both are just an
 * `Attention` value (mechanism/pattern/kernel) — cross-attention isn't a
 * structurally different thing, only a different data source for its
 * keys/values (which this graph doesn't need to represent explicitly;
 * the encoder->decoder pipeline edge already conveys that dependency).
 */
function buildAttentionNode(
  attention: Attention,
  label: string,
  pathSegment: 'attention' | 'crossAttention',
  stepId: string,
  blockPath: string,
  edges: GraphEdge[],
): GraphNode {
  const attentionPath = `${blockPath}.${pathSegment}`;
  const mechanismId = `${stepId}-mechanism`;
  const patternId = `${stepId}-pattern`;
  const kernelId = `${stepId}-kernel`;

  edges.push({ id: `${stepId}-edge-0`, source: mechanismId, target: patternId });
  edges.push({ id: `${stepId}-edge-1`, source: patternId, target: kernelId });

  return {
    id: stepId,
    label,
    category: pathSegment,
    path: attentionPath,
    children: [
      {
        id: mechanismId,
        label: attention.mechanism.primitive,
        category: `${pathSegment}.mechanism`,
        path: `${attentionPath}.mechanism`,
      },
      {
        id: patternId,
        label: attention.pattern.primitive,
        category: `${pathSegment}.pattern`,
        path: `${attentionPath}.pattern`,
      },
      {
        id: kernelId,
        label: attention.kernel.primitive,
        category: `${pathSegment}.kernel`,
        path: `${attentionPath}.kernel`,
      },
    ],
  };
}