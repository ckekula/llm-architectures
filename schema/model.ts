/**
 * model.ts
 * --------
 * LLMArchitecture is the canonical architectural object.
 *
 * Authored fields describe the architecture.
 * Parameter accounting and KV-cache information are derived from it.
 */

import type {
  ModelOverview,
  IOArchitecture,
  Tokenization,
  EmbeddingArchitecture,
  PositionalEncoding,
  LayerOrganization,
  OutputHead,
  ParameterAccounting,
  KVCache,
  KVCacheConfig,
  ContextArchitecture,
  TransformerBlock,
  Attention,
  FeedForwardNetwork,
} from './categories';

import {
  FFNType,
  FFNGating,
  NormType,
  PositionalEncodingType,
} from './primitives';

import type { Explanation } from './base';

import { CONCEPT_EXPLANATIONS } from './explanations';

// =============================================================================
// LLMArchitecture
// =============================================================================

export class LLMArchitecture {
  constructor(
    public overview: ModelOverview,
    public io: IOArchitecture,
    public tokenization: Tokenization,
    public embedding: EmbeddingArchitecture,
    public positionalEncoding: PositionalEncoding,
    public layerOrganization: LayerOrganization,
    public outputHead: OutputHead,
    public contextArchitecture: ContextArchitecture,
    public kvCacheConfig?: KVCacheConfig,
  ) {}

  /**
   * Number of layers.
   */
  get numLayers(): number {
    return this.layerOrganization.layers.totalLayers;
  }

  /**
   * Model hidden dimension.
   */
  get hiddenDimension(): number {
    return this.embedding.dimension;
  }

  /**
   * Returns the canonical block configuration for a layer.
   */
  getLayer(layerIndex: number): TransformerBlock {
    const layers = this.layerOrganization.layers;

    const firstBlock = layers.blocks[0];
    if (firstBlock === undefined) {
      throw new Error(`Uniform LayerPattern is missing its single block.`);
    }

    if (layers.pattern === 'uniform') {
      return firstBlock;
    }

    const blockIndex = layers.pattern[layerIndex];

    if (blockIndex === undefined) {
      throw new RangeError(
        `No block pattern entry for layer ${layerIndex}.`,
      );
    }

    const block = layers.blocks[blockIndex];

    if (!block) {
      throw new RangeError(
        `Layer ${layerIndex} references invalid block ${blockIndex}.`,
      );
    }

    return block;
  }

  /**
   * Plain-data projection for persistence.
   *
   * Derived properties are deliberately NOT serialized as authoritative
   * architecture data.
   */
  toJSON(): Record<string, unknown> {
    return {
      overview: this.overview,
      io: this.io,
      tokenization: this.tokenization,
      embedding: this.embedding,
      positionalEncoding: this.positionalEncoding,
      layerOrganization: this.layerOrganization,
      outputHead: this.outputHead,
      contextArchitecture: this.contextArchitecture,
      kvCacheConfig: this.kvCacheConfig,
    };
  }

  static fromJSON(
    data: Record<string, any>,
  ): LLMArchitecture {
    return new LLMArchitecture(
      data.overview,
      data.io,
      data.tokenization,
      data.embedding,
      data.positionalEncoding,
      data.layerOrganization,
      data.outputHead,
      data.contextArchitecture,
      data.kvCacheConfig,
    );
  }
}

// =============================================================================
// Parameter accounting
// =============================================================================

export function computeParameterAccounting(
  arch: LLMArchitecture,
): ParameterAccounting {
  const dModel = arch.hiddenDimension;
  const vocabSize = arch.tokenization.vocabSize;

  // ---------------------------------------------------------------------------
  // Embedding
  // ---------------------------------------------------------------------------

  const embedding =
    vocabSize * arch.embedding.dimension;

  // ---------------------------------------------------------------------------
  // Positional encoding
  // ---------------------------------------------------------------------------

  let positionalEncoding = 0;

  if (
    arch.positionalEncoding.primitive ===
    PositionalEncodingType.LearnedAbsolute
  ) {
    const maxPosition =
      arch.positionalEncoding.config.maxTrainedPosition ??
      arch.contextArchitecture.trainedContextLength;

    positionalEncoding =
      maxPosition * dModel;
  }

  // ---------------------------------------------------------------------------
  // Count layers by block configuration
  // ---------------------------------------------------------------------------

  const layerCounts = countBlockOccurrences(
    arch.layerOrganization.layers,
  );

  let attention = 0;
  let feedForward = 0;
  let moeExperts = 0;
  let normalization = 0;

  for (const [blockIndex, count] of layerCounts.entries()) {
    const block =
      arch.layerOrganization.layers.blocks[blockIndex];

    if (!block) {
      throw new Error(
        `Invalid block index ${blockIndex} in layer pattern.`,
      );
    }

    attention +=
      count * calculateAttentionParameters(
        dModel,
        block.attention,
      );

    feedForward +=
      count * calculateFeedForwardParameters(
        dModel,
        block.feedForward,
        block.moe,
      );

    if (
      block.feedForward.primitive === FFNType.MoE &&
      block.moe
    ) {
      const expertParams =
        calculateMoEExpertParameters(
          dModel,
          block.moe.config.expertHiddenDim,
          block.moe.config.numExperts,
          block.moe.config.numSharedExperts ?? 0,
        );

      moeExperts += count * expertParams;
    }

    normalization +=
      count * calculateNormalizationParameters(
        dModel,
        block,
      );
  }

  // ---------------------------------------------------------------------------
  // Output head
  // ---------------------------------------------------------------------------

  const outputHead =
    arch.outputHead.tiedWithEmbedding
      ? 0
      : dModel * vocabSize;

  // ---------------------------------------------------------------------------
  // Total
  // ---------------------------------------------------------------------------

  const other = 0;

  const total =
    embedding +
    positionalEncoding +
    attention +
    feedForward +
    normalization +
    outputHead +
    other;

  const reported =
    arch.overview.totalParameters;

  const differenceFromReported =
    total - reported;

  const differencePercent =
    reported === 0
      ? 0
      : (differenceFromReported / reported) * 100;

  return {
    embedding,
    positionalEncoding,
    attention,
    feedForward,
    moeExperts:
      moeExperts > 0 ? moeExperts : undefined,
    normalization,
    outputHead,
    other,
    total,
    differenceFromReported,
    differencePercent,

    explanation: parameterAccountingExplanation(),
  };
}

// =============================================================================
// Parameter helpers
// =============================================================================

function calculateAttentionParameters(
  dModel: number,
  attention: Attention,
): number {
  const {
    numQueryHeads,
    numKeyValueHeads,
    headDim,
  } = attention.mechanism.config;

  const qProjection =
    dModel *
    numQueryHeads *
    headDim;

  const kProjection =
    dModel *
    numKeyValueHeads *
    headDim;

  const vProjection =
    dModel *
    numKeyValueHeads *
    headDim;

  const outputProjection =
    numQueryHeads *
    headDim *
    dModel;

  return (
    qProjection +
    kProjection +
    vProjection +
    outputProjection
  );
}

function calculateFeedForwardParameters(
  dModel: number,
  ffn: FeedForwardNetwork,
  moe?: any,
): number {
  const hiddenDim = ffn.config.hiddenDim;

  if (ffn.primitive === FFNType.MoE) {
    /*
     * Expert parameters are handled separately in the MoE accounting.
     * The base FFN contribution is therefore zero here.
     */
    return 0;
  }

  const inputProjection =
    dModel * hiddenDim;

  const outputProjection =
    hiddenDim * dModel;

  const gating =
    ffn.config.gating.primitive === FFNGating.None
      ? 0
      : dModel * hiddenDim;

  return (
    inputProjection +
    gating +
    outputProjection
  );
}

function calculateMoEExpertParameters(
  dModel: number,
  expertHiddenDim: number,
  numExperts: number,
  numSharedExperts: number,
): number {
  const parametersPerExpert =
    2 * dModel * expertHiddenDim;

  const sharedExpertParameters =
    numSharedExperts *
    parametersPerExpert;

  const routedExpertParameters =
    numExperts *
    parametersPerExpert;

  return (
    routedExpertParameters +
    sharedExpertParameters
  );
}

function calculateNormalizationParameters(
  dModel: number,
  block: TransformerBlock,
): number {
  if (
    block.normalization.primitive ===
    NormType.LayerNorm
  ) {
    // gamma + beta
    return 2 * dModel * countNormalizationSites(block);
  }

  if (
    block.normalization.primitive ===
    NormType.RMSNorm
  ) {
    // gamma only
    return dModel * countNormalizationSites(block);
  }

  return dModel * countNormalizationSites(block);
}

function countNormalizationSites(
  block: TransformerBlock,
): number {
  const matches =
    block.sublayerOrder.filter(
      (x) => x === 'norm',
    );

  return matches.length;
}

function countBlockOccurrences<T>(
  layerPattern: {
    totalLayers: number;
    blocks: T[];
    pattern: 'uniform' | number[];
  },
): Map<number, number> {
  const counts = new Map<number, number>();

  if (layerPattern.pattern === 'uniform') {
    counts.set(0, layerPattern.totalLayers);
    return counts;
  }

  for (let i = 0; i < layerPattern.totalLayers; i++) {
    const blockIndex =
      layerPattern.pattern[i];

    if (blockIndex === undefined) {
      throw new Error(
        `Layer pattern is missing entry for layer ${i}.`,
      );
    }

    counts.set(
      blockIndex,
      (counts.get(blockIndex) ?? 0) + 1,
    );
  }

  return counts;
}

// =============================================================================
// KV cache
// =============================================================================

export function computeKVCache(
  arch: LLMArchitecture,
  sequenceLength: number,
): KVCache {
  if (sequenceLength < 0) {
    throw new RangeError(
      'sequenceLength must be non-negative.',
    );
  }

  const layers =
    arch.layerOrganization.layers;

  let bytesPerTokenPerLayer = 0;

  /*
   * KV cache size can vary between block types in a heterogeneous model.
   *
   * Calculate the total cache required for one token across all layers.
   */
  for (let layerIndex = 0;
       layerIndex < layers.totalLayers;
       layerIndex++) {

    const block =
      arch.getLayer(layerIndex);

    const {
      numKeyValueHeads,
      headDim,
    } = block.attention.mechanism.config;

    const keyBytes =
      arch.kvCacheConfig?.keyBytesPerElement ?? 2;

    const valueBytes =
      arch.kvCacheConfig?.valueBytesPerElement ?? 2;

    const layerBytes =
      numKeyValueHeads *
      headDim *
      (keyBytes + valueBytes);

    bytesPerTokenPerLayer += layerBytes;
  }

  const bytesPerToken =
    bytesPerTokenPerLayer;

  const totalBytes =
    bytesPerToken * sequenceLength;

  return {
    bytesPerTokenPerLayer,
    bytesPerToken,
    totalBytes,
    sequenceLength,

    explanation: kvCacheExplanation(),
  };
}

// =============================================================================
// Generic explanations for derived concepts
// =============================================================================

function parameterAccountingExplanation(): Explanation {
  return {
    ...CONCEPT_EXPLANATIONS.parameterAccounting,
  };
}

function kvCacheExplanation(): Explanation {
  return {
    ...CONCEPT_EXPLANATIONS.kvCache,
  };
}

// =============================================================================
// Architecture diff
// =============================================================================

export interface DiffEntry {
  path: string;
  a: unknown;
  b: unknown;
  same: boolean;
}

export function diffArchitectures(
  a: LLMArchitecture,
  b: LLMArchitecture,
): DiffEntry[] {
  const entries: DiffEntry[] = [];

  walk(
    a.toJSON(),
    b.toJSON(),
    '',
    entries,
  );

  return entries;
}

function walk(
  a: unknown,
  b: unknown,
  path: string,
  out: DiffEntry[],
): void {
  const bothObjects =
    isPlainObject(a) &&
    isPlainObject(b);

  if (!bothObjects) {
    out.push({
      path,
      a,
      b,
      same:
        JSON.stringify(a) ===
        JSON.stringify(b),
    });

    return;
  }

  const keys = new Set([
    ...Object.keys(a),
    ...Object.keys(b),
  ]);

  for (const key of keys) {
    const childA =
      a[key];

    const childB =
      b[key];

    walk(
      childA,
      childB,
      path
        ? `${path}.${key}`
        : key,
      out,
    );
  }
}

function isPlainObject(
  value: unknown,
): value is Record<string, unknown> {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value)
  );
}