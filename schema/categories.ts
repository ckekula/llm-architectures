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

import type { ArchitecturalChoice, Explainable, LayerPattern } from './base';

export interface ModelOverview extends Explainable {
  name: string;
  organization: string;
  releaseDate: string;

  family?: string;
  paperUrl?: string;
  license?: string;

  totalParameters: number; // Reported/citable parameter count (intentionally authored rather than calculated).
  activeParameters?: number; // Reported for MoE models. Dense models may omit this because it is derivable.
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
  embeddingScale?: number; // Multiplicative scaling applied to embeddings.
  embeddingDropout?: number; // Dropout applied after embedding.
}

export type PositionalEncoding = ArchitecturalChoice<
  PositionalEncodingType,
  {
    appliedAt: 'embedding' | 'attention';

    ropeTheta?: number;
    maxTrainedPosition?: number
    scalingFactor?: number;
  }
> &
  Explainable;

// -----------------------------------------------------------------------------
// Attention
// -----------------------------------------------------------------------------

export interface AttentionMechanismConfig {
  numQueryHeads: number;
  numKeyValueHeads: number;
  headDim: number;
}

export interface AttentionPatternConfig {
  causal: boolean;
  windowSize?: number;
}

export type AttentionKernelConfig = ArchitecturalChoice<AttentionKernel, Record<string, never>> & Explainable;

export interface Attention {
  mechanism: ArchitecturalChoice <AttentionMechanism, AttentionMechanismConfig>;
  pattern: ArchitecturalChoice <AttentionPattern, AttentionPatternConfig>;
  kernel: ArchitecturalChoice <AttentionKernel, AttentionKernelConfig>;
}

// -----------------------------------------------------------------------------
// Feed-forward network
// -----------------------------------------------------------------------------

export interface FeedForwardConfig {
  hiddenDim: number;
  activation: ArchitecturalChoice<ActivationFunction, Record<string, never>>;
  // Whether the FFN uses a gated formulation. Independent of `activation`.
  gating: ArchitecturalChoice<FFNGating, Record<string, never>>;
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

export interface TransformerBlock extends Explainable {
  attention: Attention;
  feedForward: FeedForwardNetwork;
  moe?: MixtureOfExperts;
  normalization: Normalization;
  residual: ResidualArchitecture;
  sublayerOrder: string[];

  attentionDropout?: number;
  residualDropout?: number;
}

// The actual activation configuration remains owned by the FFN.
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

// Parameter accounting — DERIVED, not authored.
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

// Actual cache size is calculated from the architecture.
export interface KVCacheConfig extends Explainable {
  keyBytesPerElement: number;
  valueBytesPerElement: number;
}

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