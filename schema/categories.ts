/**
 * categories.ts
 * -------------
 * The architectural categories. Each interface is plain data (the
 * "contains" side); semantics come from the `primitive` fields, which are
 * ArchitecturalChoice<...> pointing into primitives.ts (the "is" side).
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

// 1. Model overview
export interface ModelOverview extends Explainable {
  name: string;
  organization: string;
  releaseDate: string; // ISO date
  family?: string; // e.g. "Llama 3"
  paperUrl?: string;
  license?: string;
  
  totalParameters: number; // Reported/citable parameter count (intentionally authored rather than calculated).
  activeParameters?: number; // Reported for MoE models. Dense models may omit this because it is derivable.
}

// 2. IO architecture
export interface IOArchitecture extends Explainable {
  /**
   * MVP is restricted to text. Multimodal models get their own, wider
   * IOArchitecture variant later (discriminated union on this field) —
   * they are NOT forced into this shape.
   */
  inputModality: 'text';
  outputType: 'next_token_logits';
  vocabSize: number;
  maxSequenceLength: number;
}

// 3. Tokenization
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

// 4. Embedding architecture
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
    maxTrainedPosition?: number;
  }
> &
  Explainable;

// 7. Attention
export type AttentionMechanismConfig = ArchitecturalChoice<
  AttentionMechanism,
  {
    numQueryHeads: number;
    numKeyValueHeads: number;
    headDim: number;
  }
> &
  Explainable;

export type AttentionPatternConfig = ArchitecturalChoice<
  AttentionPattern,
  {
    causal: boolean;
    windowSize?: number; // Only for SlidingWindow / LocalBanded.
  }
> &
  Explainable;

export type AttentionKernelConfig = ArchitecturalChoice<AttentionKernel, Record<string, never>> & Explainable;

export interface Attention {
  mechanism: AttentionMechanismConfig;
  pattern: AttentionPatternConfig;
  kernel: AttentionKernelConfig;
}

// 9. Feed-forward network
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

// 9. Mixture-of-Experts
// present on a TransformerBlock iff feedForward.primitive === FFNType.MoE.
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

// 10. Normalization
export type Normalization = ArchitecturalChoice<
  NormType,
  {
    placement: NormPlacement;
    epsilon: number;
  }
> &
  Explainable;

// 11. Residual architecture
export type ResidualArchitecture = ArchitecturalChoice<
  ResidualType,
  {
    numStreams: number;
  }
> &
  Explainable;

// 6. Transformer block — composes 7, 8, 9?, 10, 11 plus sublayer order.
// This is the ONE canonical block definition; repetition is handled by
// LayerPattern in category 13.
export interface TransformerBlock extends Explainable {
  attention: Attention;
  /**
   * A second attention sub-layer that attends over ANOTHER stack's output
   * present iff `sublayerOrder` includes a cross-attention step. Reuses
   * the same `Attention` shape as self-attention.
   */
  crossAttention?: Attention;
  feedForward: FeedForwardNetwork;
  moe?: MixtureOfExperts;
  normalization: Normalization;
  residual: ResidualArchitecture;
  /**
   * Order sublayers execute in, e.g. ['norm','attention','residual_add','norm','ffn','residual_add']
   * (pre-norm) or ['attention','residual_add','norm','ffn','residual_add','norm']
   * (post-norm, e.g. the original Transformer). A 'cross_attention' entry
   * requires `crossAttention` to be set.
   */
  sublayerOrder: string[];

  attentionDropout?: number;
  residualDropout?: number;
}

// 12. Activation functions — cross-cutting summary. Values here are
// references INTO the owning components (FFN, attention gating, etc.),
export interface ActivationProfile extends Explainable {
  usage: Record<string, ActivationFunction>;
}

// 13. Layer organization — one stack (a LayerPattern of TransformerBlocks).
// Reused as-is for encoder and decoder stacks; the encoder/decoder split
// itself lives one level up, in `ModelStacks`.
export interface LayerOrganization extends Explainable {
  layers: LayerPattern<TransformerBlock>;
}

/**
 * A model has an encoder stack, a decoder stack, or both:
 *  - decoder-only (GPT-style): { decoder }
 *  - encoder-only (BERT-style): { encoder }
 *  - encoder-decoder (the original Transformer, T5): { encoder, decoder }
 * `LLMArchitecture`'s constructor enforces at least one is present
 */
export interface ModelStacks {
  encoder?: LayerOrganization;
  decoder?: LayerOrganization;
}

// 14. Output head
export interface OutputHead extends Explainable {
  tiedWithEmbedding: boolean;
  finalNormalization: boolean;
}

// 15. Parameter accounting — DERIVED, not authored.
export interface ParameterAccounting extends Explainable {
  embedding: number;
  positionalEncoding: number;
  attention: number;
  crossAttention?: number; // Cross-attention params, summed across all blocks that have crossAttention
  feedForward: number;
  normalization: number;
  outputHead: number;
  other: number;
  total: number;

  moeExperts?: number;
  active?: number;
}

// 16. Context architecture
export interface ContextArchitecture extends Explainable {
  trainedContextLength: number;
  extendedContextLength?: number;
  extensionMethod?: 'rope_scaling' | 'position_interpolation' | 'yarn' | 'none';
  slidingWindow?: number;
}

// 17. KV cache
/** Authored, optional per-model configuration for KV cache sizing. */
export interface KVCacheConfig extends Explainable {
  keyBytesPerElement: number;
  valueBytesPerElement: number;
}

/**
 * DERIVED result of `computeKVCache(arch, sequenceLength)` in model.ts.
 * `explanation` comes from CONCEPT_EXPLANATIONS.kvCache — generic, not per-model.
 */
export interface KVCache extends Explainable {
  bytesPerTokenPerLayer: number;
  bytesPerToken: number;
  totalBytes: number;
  sequenceLength: number;
}
