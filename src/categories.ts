/**
 * categories.ts
 * -------------
 * The architectural categories.
 *
 * These describe what an architecture contains.
 * The primitive fields describe what each component IS.
 */

import {
  TokenizerAlgorithm,
  PositionalEncodingType,
  AttentionMechanism,
  AttentionPattern,
  AttentionKernel,
  NormType,
  NormPlacement,
  FFNType,
  ActivationFunction,
  FFNGating,
  MoERoutingType,
  ResidualType,
} from './primitives';

import type {
  ArchitecturalChoice,
  Explainable,
  LayerPattern,
} from './base';

export interface ModelOverview extends Explainable {
  name: string;
  organization: string;
  releaseDate: string;

  family?: string;
  paperUrl?: string;
  license?: string;

  /**
   * Reported/citable parameter count.
   *
   * This is intentionally authored rather than calculated because papers
   * often report rounded values such as "175B".
   */
  totalParameters: number;

  /**
   * For MoE models, reported active parameters per forward pass.
   * Dense models may omit this because it is derivable.
   */
  reportedActiveParameters?: number;
}

export interface IOArchitecture extends Explainable {
  inputModality: 'text';
  outputType: 'next_token_logits';

  vocabSize: number;
  maxSequenceLength: number;
}

export interface Tokenization extends Explainable {
  algorithm: TokenizerAlgorithm;

  vocabSize: number;

  specialTokens: {
    bos?: string;
    eos?: string;
    pad?: string;
    unk?: string;
    additional?: string[];
  };
}

export interface EmbeddingArchitecture extends Explainable {
  dimension: number;
  tiedWithOutputHead: boolean;
  embeddingScale?: number; // Multiplicative scaling applied to embeddings, if any.
  embeddingDropout?: number; // Dropout applied after embedding, if any.
}

export type PositionalEncoding = ArchitecturalChoice<
  PositionalEncodingType,
  {
    appliedAt: 'embedding' | 'attention'; // Where the positional mechanism is applied.

    ropeTheta?: number; // RoPE-specific configuration.
    maxTrainedPosition?: number; // Highest position used during native training.
    scalingFactor?: number; // Optional scaling/extension details.
  }
> &
  Explainable;

export interface TransformerBlock extends Explainable {
  attention: Attention;
  feedForward: FeedForwardNetwork;
  normalization: Normalization;
  residual: ResidualArchitecture;

  /**
   * Explicit execution order.
   *
   * Kept for now as a compact representation of the block's computation
   * structure. This can later be replaced by a full computation graph.
   */
  sublayerOrder: string[];
  moe?: MixtureOfExperts; // Present iff feedForward.primitive === FFNType.MoE.
  attentionDropout?: number;
  residualDropout?: number;
}

// -----------------------------------------------------------------------------
// Attention
// -----------------------------------------------------------------------------

export interface AttentionConfig {
  numQueryHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  causal: boolean;

  windowSize?: number; // Window size is meaningful for local/sliding-window patterns.
}

export interface AttentionMechanismConfig {
  numQueryHeads: number;
  numKeyValueHeads: number;
  headDim: number;
}

export interface AttentionPatternConfig {
  windowSize?: number;
}

export interface Attention extends Explainable {
  /**
   * How Q/K/V heads are constructed and shared.
   *
   * Examples:
   * - MHA: numQueryHeads === numKeyValueHeads
   * - MQA: numKeyValueHeads === 1
   * - GQA: numKeyValueHeads < numQueryHeads
   * - MLA: latent attention configuration
   */
  mechanism: ArchitecturalChoice<
    AttentionMechanism,
    AttentionMechanismConfig
  >;

  /**
   * Which tokens are allowed to interact.
   *
   * Examples:
   * - Dense
   * - SlidingWindow
   * - LocalBanded
   */
  pattern: ArchitecturalChoice<
    AttentionPattern,
    AttentionPatternConfig
  >;

  /**
   * Mathematical kernel used to calculate attention.
   */
  kernel: ArchitecturalChoice<
    AttentionKernel,
    Record<string, never>
  >;

  causal: boolean; // Whether the attention operation is causally masked.
}
// -----------------------------------------------------------------------------
// Feed-forward network
// -----------------------------------------------------------------------------

export interface FeedForwardConfig {
  hiddenDim: number;
  activation: ArchitecturalChoice<
    ActivationFunction,
    Record<string, never>
  >;
  gating: ArchitecturalChoice<
    FFNGating,
    Record<string, never>
  >;
}

export type FeedForwardNetwork = ArchitecturalChoice<
  FFNType,
  FeedForwardConfig
> &
  Explainable;

// -----------------------------------------------------------------------------

export type MixtureOfExperts = ArchitecturalChoice<
  MoERoutingType,
  {
    numExperts: number;
    numActiveExperts: number;
    numSharedExperts?: number;
    expertHiddenDim: number;
    loadBalancingLoss: boolean;
  }
> &
  Explainable;

export type Normalization = ArchitecturalChoice<
  NormType,
  {
    placement: NormPlacement;
    epsilon: number;
  }
> &
  Explainable;

export type ResidualArchitecture = ArchitecturalChoice<
  ResidualType,
  {
    numStreams: number;
  }
> &
  Explainable;

/**
 * Cross-cutting index for encyclopedia browsing.
 *
 * The actual activation configuration remains owned by the FFN.
 */
export interface ActivationProfile extends Explainable {
  usage: Record<string, ActivationFunction>;
}

export interface LayerOrganization extends Explainable {
  layers: LayerPattern<TransformerBlock>;
}

export interface OutputHead extends Explainable {
  tiedWithEmbedding: boolean;

  finalNormalization: boolean;
}

/**
 * ParameterAccounting is a computed result, not model-authored architecture.
 *
 * `reportedTotal` is kept separately in ModelOverview because it represents
 * the citable number reported by the model authors.
 */
export interface ParameterAccounting extends Explainable {
  embedding: number;
  positionalEncoding: number;
  attention: number;
  feedForward: number;
  normalization: number;
  outputHead: number;
  other: number;
  total: number;
  differenceFromReported: number; // Difference between the reported parameter count and the bottom-up calculated count.
  differencePercent: number;

  moeExperts?: number;
  active?: number;
}

// -----------------------------------------------------------------------------
// KV cache — DERIVED
// -----------------------------------------------------------------------------

/**
 * KV cache configuration describes the storage representation.
 *
 * Actual cache size is calculated from the architecture.
 */
export interface KVCacheConfig extends Explainable {
  keyBytesPerElement: number;
  valueBytesPerElement: number;
}

/**
 * Derived KV cache result.
 */
export interface KVCache extends Explainable {
  bytesPerTokenPerLayer: number;
  bytesPerToken: number;
  totalBytes: number;
  sequenceLength: number;
}

// -----------------------------------------------------------------------------

export interface ContextArchitecture extends Explainable {
  trainedContextLength: number;

  extendedContextLength?: number;
  extensionMethod?:
    | 'rope_scaling'
    | 'position_interpolation'
    | 'yarn'
    | 'none';
  slidingWindow?: number;
}