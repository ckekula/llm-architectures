/**
 * primitives.ts
 * -------------
 * Shared architectural vocabulary.
 *
 * These primitives are the encyclopedia's reusable concepts.
 * Model-specific configuration and rationale live on ArchitecturalChoice.
 */

export enum TokenizerAlgorithm {
  BPE = 'bpe',
  ByteLevelBPE = 'byte_level_bpe',
  Unigram = 'unigram',
  WordPiece = 'wordpiece',
  SentencePiece = 'sentencepiece',
}

export enum PositionalEncodingFamily {
  Absolute = 'absolute',
  Relative = 'relative',
  Rotary = 'rotary',
  Bias = 'bias',
  None = 'none',
}

export enum PositionalEncodingType {
  LearnedAbsolute = 'learned_absolute',
  Sinusoidal = 'sinusoidal',
  RoPE = 'rope',
  ALiBi = 'alibi',
  RelativeBias = 'relative_bias',
  None = 'none',
}

/**
 * Positional encoding family is derived from the concrete type.
 *
 * This avoids storing redundant information that could drift.
 */
const POSITIONAL_ENCODING_FAMILY: Record<PositionalEncodingType, PositionalEncodingFamily> = {
  [PositionalEncodingType.LearnedAbsolute]: PositionalEncodingFamily.Absolute,
  [PositionalEncodingType.Sinusoidal]: PositionalEncodingFamily.Absolute,
  [PositionalEncodingType.RoPE]: PositionalEncodingFamily.Rotary,
  [PositionalEncodingType.ALiBi]: PositionalEncodingFamily.Bias,
  [PositionalEncodingType.RelativeBias]: PositionalEncodingFamily.Relative,
  [PositionalEncodingType.None]: PositionalEncodingFamily.None,
};

export function familyOf(type: PositionalEncodingType): PositionalEncodingFamily {
  return POSITIONAL_ENCODING_FAMILY[type];
}

// -----------------------------------------------------------------------------
// Attention
// Split into three independent axes:
//  - Mechanism: how Q/K/V are constructed and shared across heads.
//  - Pattern:   which tokens are allowed to attend to which other tokens.
//  - Kernel:    the score function itself (softmax vs. a linear-attention
//               approximation). Orthogonal to Mechanism — a linear-attention
//               model can still choose MHA- or MQA-style K/V sharing.
// -----------------------------------------------------------------------------

export enum AttentionMechanism {
  MHA = 'multi_head_attention',
  MQA = 'multi_query_attention',
  GQA = 'grouped_query_attention',
  MLA = 'multi_head_latent_attention',
}

export enum AttentionPattern {
  Dense = 'dense',
  SlidingWindow = 'sliding_window',
  LocalBanded = 'local_banded',
  Sparse = 'sparse',
}

export enum AttentionKernel {
  Softmax = 'softmax',
  Linear = 'linear',
}

// -----------------------------------------------------------------------------
// Normalization
// -----------------------------------------------------------------------------

export enum NormType {
  LayerNorm = 'layer_norm',
  RMSNorm = 'rms_norm',
  DeepNorm = 'deep_norm',
}

export enum NormPlacement {
  PreNorm = 'pre_norm',
  PostNorm = 'post_norm',
  Sandwich = 'sandwich_norm',
}

// -----------------------------------------------------------------------------
// Feed-forward network
// -----------------------------------------------------------------------------

export enum FFNType {
  Dense = 'dense_mlp',
  GatedLinearUnit = 'gated_linear_unit',
  MoE = 'mixture_of_experts',
}

// `ActivationFunction` is the elementwise nonlinearity. `FFNGating` is a
// separate architectural axis: whether the FFN uses a gated formulation
// (e.g. SwiGLU = SiLU activation + a GLU-style gating structure).
export enum ActivationFunction {
  ReLU = 'relu',
  GELU = 'gelu',
  SiLU = 'silu',
}

export enum FFNGating {
  None = 'none',
  GLU = 'glu',
  SwiGLU = 'swiglu',
  GeGLU = 'geglu',
}

// -----------------------------------------------------------------------------
// Mixture-of-Experts routing
// -----------------------------------------------------------------------------

export enum MoERoutingType {
  TopK = 'top_k',
  ExpertChoice = 'expert_choice',
  Switch = 'switch_top_1',
}

// -----------------------------------------------------------------------------
// Residual stream
// -----------------------------------------------------------------------------

export enum ResidualType {
  Standard = 'standard_additive',
  Parallel = 'parallel_streams',
}
