/**
 * explanations.ts
 * ---------------
 * Explanations for the architectural primitives.
 */

// -----------------------------------------------------------------------------
// Primitive explanations
// -----------------------------------------------------------------------------

import type { Explanation } from "./base";
import {
    ActivationFunction,
    AttentionKernel,
    AttentionMechanism,
    AttentionPattern,
    FFNGating,
    FFNType,
    NormType,
    PositionalEncodingType
} from "./primitives";

export const PRIMITIVE_EXPLANATIONS: Record<string, Explanation> = {
  // ---------------------------------------------------------------------------
  // Attention mechanisms
  // ---------------------------------------------------------------------------

  [AttentionMechanism.MHA]: {
    whatIsIt:
      'Multi-Head Attention: each attention head has its own query, key, and value projections.',
    howIsItBuilt:
      'The model projects the hidden state independently into Q, K, and V for each attention head, computes attention within each head, then concatenates and projects the results.',
  },

  [AttentionMechanism.MQA]: {
    whatIsIt:
      'Multi-Query Attention: all query heads share a single key and value head.',
    howIsItBuilt:
      'Multiple query heads are computed independently, while K and V are projected only once and shared across all query heads. This substantially reduces KV-cache size.',
  },

  [AttentionMechanism.GQA]: {
    whatIsIt:
      'Grouped-Query Attention: groups of query heads share key and value heads.',
    howIsItBuilt:
      'N query heads are divided into G groups. Each group shares one K/V head, reducing K/V projections and KV-cache memory compared with standard multi-head attention.',
  },

  [AttentionMechanism.MLA]: {
    whatIsIt:
      'Multi-head Latent Attention: attention uses a compressed latent representation of keys and values.',
    howIsItBuilt:
      'Keys and values are projected into a lower-dimensional latent representation that can be cached and subsequently expanded or transformed for attention.',
  },

  // ---------------------------------------------------------------------------
  // Attention patterns
  // ---------------------------------------------------------------------------

  [AttentionPattern.Dense]: {
    whatIsIt:
      'Dense attention allows every token to attend to every permitted token.',
    howIsItBuilt:
      'For a causal language model, each token attends to itself and all preceding tokens, producing an O(n²) attention connectivity pattern.',
  },

  [AttentionPattern.SlidingWindow]: {
    whatIsIt:
      'Sliding-window attention restricts each token to a local window of nearby tokens.',
    howIsItBuilt:
      'Each query attends only to keys within a fixed-size window around its position, reducing the number of attention interactions compared with full attention.',
  },

  [AttentionPattern.LocalBanded]: {
    whatIsIt:
      'Local-banded attention restricts attention to a band around the current token position.',
    howIsItBuilt:
      'The attention matrix is sparse, with non-zero entries concentrated around its diagonal rather than spanning the full sequence.',
  },

  // ---------------------------------------------------------------------------
  // Attention kernels
  // ---------------------------------------------------------------------------

  [AttentionKernel.Softmax]: {
    whatIsIt:
      'Softmax attention converts scaled query-key similarities into normalized attention weights.',
    howIsItBuilt:
      'Attention is computed as softmax(QKᵀ / √d)V, optionally masked to enforce causality or a local attention pattern.',
  },

  [AttentionKernel.Linear]: {
    whatIsIt:
      'Linear attention replaces the quadratic softmax attention calculation with a kernelized formulation.',
    howIsItBuilt:
      'Queries and keys are transformed into feature representations that allow attention to be rearranged so that sequence-dependent computation scales approximately linearly with sequence length.',
  },

  // ---------------------------------------------------------------------------
  // Positional encoding
  // ---------------------------------------------------------------------------

  [PositionalEncodingType.LearnedAbsolute]: {
    whatIsIt:
      'Learned absolute positional embeddings assign a learned vector to each position.',
    howIsItBuilt:
      'A position-indexed embedding table is added to the token representation before entering the Transformer.',
  },

  [PositionalEncodingType.Sinusoidal]: {
    whatIsIt:
      'Sinusoidal positional encoding represents position using deterministic sine and cosine functions.',
    howIsItBuilt:
      'Each position is mapped to sinusoidal values at different frequencies, producing a fixed positional representation without learned parameters.',
  },

  [PositionalEncodingType.RoPE]: {
    whatIsIt:
      'Rotary Positional Embedding encodes position by rotating query and key vectors.',
    howIsItBuilt:
      'Pairs of Q/K dimensions are interpreted as 2D vectors and rotated by position-dependent angles. Because the rotation is applied directly to Q and K, relative positional information emerges naturally in their dot product.',
  },

  [PositionalEncodingType.ALiBi]: {
    whatIsIt:
      'ALiBi adds position-dependent linear biases directly to attention scores.',
    howIsItBuilt:
      'A head-specific slope is multiplied by token distance and added to the attention score before softmax.',
  },

  [PositionalEncodingType.RelativeBias]: {
    whatIsIt:
      'Relative position bias modifies attention scores according to the relative distance between tokens.',
    howIsItBuilt:
      'A learned or fixed bias indexed by relative token displacement is added to the QK attention scores.',
  },

  [PositionalEncodingType.None]: {
    whatIsIt:
      'No explicit positional encoding is applied to the token representation or attention scores.',
    howIsItBuilt:
      'The architecture relies on another mechanism or inductive bias to represent sequence order.',
  },

  // ---------------------------------------------------------------------------
  // Normalization
  // ---------------------------------------------------------------------------

  [NormType.LayerNorm]: {
    whatIsIt:
      'Layer Normalization normalizes activations using their mean and variance.',
    howIsItBuilt:
      'Each hidden vector is centered and scaled using its mean and variance, followed by learned scale and bias parameters.',
  },

  [NormType.RMSNorm]: {
    whatIsIt:
      'RMSNorm rescales activations using their root mean square without re-centering them.',
    howIsItBuilt:
      'The normalized vector is x / RMS(x) · g, where RMS(x) = sqrt(mean(x²) + ε) and g is a learned per-dimension gain.',
  },

  [NormType.DeepNorm]: {
    whatIsIt:
      'DeepNorm modifies normalization and residual scaling to stabilize very deep Transformers.',
    howIsItBuilt:
      'Layer normalization is combined with carefully chosen residual scaling factors designed to control activation growth in deep networks.',
  },

  // ---------------------------------------------------------------------------
  // FFN
  // ---------------------------------------------------------------------------

  [FFNType.Dense]: {
    whatIsIt:
      'A dense feed-forward network independently transforms each token representation.',
    howIsItBuilt:
      'The hidden representation is projected to a larger intermediate dimension, passed through an activation or gated activation, and projected back to the model dimension.',
  },

  [FFNType.MoE]: {
    whatIsIt:
      'A Mixture-of-Experts feed-forward network routes each token to a subset of multiple expert networks.',
    howIsItBuilt:
      'A router selects one or more experts for each token. Only the selected experts process that token, allowing the model to contain many parameters while activating fewer parameters per token.',
  },

  // ---------------------------------------------------------------------------
  // Activations
  // ---------------------------------------------------------------------------

  [ActivationFunction.ReLU]: {
    whatIsIt:
      'ReLU sets negative activations to zero.',
    howIsItBuilt:
      'The function computes max(0, x) independently for each activation.',
  },

  [ActivationFunction.GELU]: {
    whatIsIt:
      'GELU smoothly gates activations according to their magnitude.',
    howIsItBuilt:
      'The activation approximately computes x · Φ(x), where Φ is the standard normal cumulative distribution function.',
  },

  [ActivationFunction.SiLU]: {
    whatIsIt:
      'SiLU, also called Swish, multiplies an input by its sigmoid.',
    howIsItBuilt:
      'The function computes x · sigmoid(x), producing a smooth, non-monotonic activation.',
  },

  // ---------------------------------------------------------------------------
  // Gating
  // ---------------------------------------------------------------------------

  [FFNGating.None]: {
    whatIsIt:
      'No multiplicative gating is used in the feed-forward network.',
    howIsItBuilt:
      'The intermediate representation is produced by a conventional activation followed by the output projection.',
  },

  [FFNGating.GLU]: {
    whatIsIt:
      'A Gated Linear Unit uses one projection to control another through element-wise multiplication.',
    howIsItBuilt:
      'Two projections are produced from the input; one is transformed by an activation and multiplied element-wise with the other.',
  },

  [FFNGating.SwiGLU]: {
    whatIsIt:
      'SwiGLU is a gated feed-forward architecture using SiLU as its gating activation.',
    howIsItBuilt:
      'Two intermediate projections are produced; SiLU is applied to one and multiplied element-wise with the other before the down projection.',
  },

  [FFNGating.GeGLU]: {
    whatIsIt:
      'GeGLU is a gated feed-forward architecture using GELU as its gating activation.',
    howIsItBuilt:
      'Two intermediate projections are produced; GELU is applied to one and multiplied element-wise with the other before the down projection.',
  },
};

/**
 * Explanations for cross-cutting concepts that are not themselves
 * architectural choices.
 */
export const CONCEPT_EXPLANATIONS: Record<string, Explanation> = {
  dropout: {
    whatIsIt:
      'Dropout randomly removes activations during training to regularize the model.',
    howIsItBuilt:
      'During training, activations are independently zeroed with a specified probability and the remaining activations are rescaled.',
  },

  parameterAccounting: {
    whatIsIt:
      'Parameter accounting breaks a model\'s parameter count into architectural components.',
    howIsItBuilt:
      'Parameter counts are calculated from vocabulary size, dimensions, layer counts, projection matrices, normalization parameters, and other architectural components.',
  },

  kvCache: {
    whatIsIt:
      'The KV cache stores previously computed keys and values during autoregressive generation.',
    howIsItBuilt:
      'For each generated token, the model stores the K and V representations required by future attention operations, avoiding recomputation of previous tokens.',
  },
};