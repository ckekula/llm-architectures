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
export function familyOf(
  type: PositionalEncodingType,
): PositionalEncodingFamily {
  switch (type) {
    case PositionalEncodingType.LearnedAbsolute:
    case PositionalEncodingType.Sinusoidal:
      return PositionalEncodingFamily.Absolute;

    case PositionalEncodingType.RoPE:
      return PositionalEncodingFamily.Rotary;

    case PositionalEncodingType.ALiBi:
      return PositionalEncodingFamily.Bias;

    case PositionalEncodingType.RelativeBias:
      return PositionalEncodingFamily.Relative;

    case PositionalEncodingType.None:
      return PositionalEncodingFamily.None;
  }
}

// -----------------------------------------------------------------------------
// Attention
// -----------------------------------------------------------------------------

/**
 * How Q/K/V heads are constructed and shared.
 */
export enum AttentionMechanism {
  MHA = 'multi_head_attention',
  MQA = 'multi_query_attention',
  GQA = 'grouped_query_attention',
  MLA = 'multi_head_latent_attention',
}

/**
 * Which tokens are allowed to interact.
 */
export enum AttentionPattern {
  Dense = 'dense',
  SlidingWindow = 'sliding_window',
  LocalBanded = 'local_banded',
  Sparse = 'sparse',
}

/**
 * The mathematical kernel used to calculate attention scores.
 */
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
  MoE = 'mixture_of_experts',
}

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

