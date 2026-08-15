/**
 * The original Transformer ("Attention Is All You Need", Vaswani et al.
 * 2017), base configuration — Table 3 in the paper.
 *
 * This is the model the whole schema was originally built around a
 * decoder-only assumption for; it's the fixture that forced the
 * encoder/decoder stack split and `crossAttention` onto TransformerBlock.
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
  ResidualType,
  LLMArchitecture,
  type Attention,
  type FeedForwardNetwork,
  type Normalization,
  type ResidualArchitecture,
  type TransformerBlock,
} from 'schema';

const NUM_ENCODER_LAYERS = 6;
const NUM_DECODER_LAYERS = 6;
const DIM = 512;
const NUM_HEADS = 8;
const HEAD_DIM = DIM / NUM_HEADS; // 64
const FFN_HIDDEN = 2048;

const explanation = { whatIsIt: '', howIsItBuilt: '' };

/** MHA throughout — the paper predates GQA/MQA/MLA. `causal` is the only thing that differs between self- and cross-attention here. */
function makeAttention(causal: boolean): Attention {
  return {
    mechanism: {
      primitive: AttentionMechanism.MHA,
      config: { numQueryHeads: NUM_HEADS, numKeyValueHeads: NUM_HEADS, headDim: HEAD_DIM },
      explanation,
    },
    pattern: { primitive: AttentionPattern.Dense, config: { causal }, explanation },
    kernel: { primitive: AttentionKernel.Softmax, config: {}, explanation },
  };
}

function makeFeedForward(): FeedForwardNetwork {
  return {
    primitive: FFNType.Dense,
    config: {
      hiddenDim: FFN_HIDDEN,
      activation: { primitive: ActivationFunction.ReLU, config: {} },
      gating: { primitive: FFNGating.None, config: {} },
    },
    explanation,
  };
}

// Post-norm: LayerNorm(x + Sublayer(x)) — applied AFTER each sub-layer,
// unlike GPT-3's pre-norm. Shared by reference across blocks/sub-layers;
// it's immutable plain data, not per-instance state.
const normalization: Normalization = {
  primitive: NormType.LayerNorm,
  config: { placement: NormPlacement.PostNorm, epsilon: 1e-6 }, // paper doesn't state epsilon explicitly; 1e-6 is the era-typical default
  explanation,
};

const residual: ResidualArchitecture = { primitive: ResidualType.Standard, config: { numStreams: 1 }, explanation };

function makeEncoderBlock(): TransformerBlock {
  return {
    explanation,
    attention: makeAttention(false), // bidirectional self-attention — nothing is masked
    feedForward: makeFeedForward(),
    normalization,
    residual,
    sublayerOrder: ['attention', 'residual_add', 'norm', 'ffn', 'residual_add', 'norm'],
    attentionDropout: 0.1,
    residualDropout: 0.1,
  };
}

function makeDecoderBlock(): TransformerBlock {
  return {
    explanation,
    attention: makeAttention(true), // masked self-attention — can't see future target tokens
    crossAttention: makeAttention(false), // attends over the FULL encoder output, not masked
    feedForward: makeFeedForward(),
    normalization,
    residual,
    sublayerOrder: ['attention', 'residual_add', 'norm', 'cross_attention', 'residual_add', 'norm', 'ffn', 'residual_add', 'norm'],
    attentionDropout: 0.1,
    residualDropout: 0.1,
  };
}

const encoderBlock = makeEncoderBlock();
const decoderBlock = makeDecoderBlock();

export const originalTransformer = new LLMArchitecture(
  {
    explanation,
    name: 'Transformer (base)',
    organization: 'Google',
    releaseDate: '2017-06-12',
    paperUrl: 'https://arxiv.org/abs/1706.03762',
    totalParameters: 65_000_000, // paper, Table 3, "base" row
  },
  { explanation, inputModality: 'text', outputType: 'next_token_logits', vocabSize: 37000, maxSequenceLength: 512 },
  {
    explanation,
    algorithm: TokenizerAlgorithm.BPE,
    vocabSize: 37000, // shared source/target BPE vocabulary, WMT 2014 En-De (paper, section 5.1)
    specialTokens: {},
  },
  {
    explanation,
    dimension: DIM,
    tiedWithOutputHead: true, // input embedding, output embedding, and pre-softmax linear all share one weight matrix (paper, section 3.4)
    embeddingScale: Math.sqrt(DIM), // embeddings are multiplied by sqrt(d_model) (paper, section 3.4)
    embeddingDropout: 0.1, // dropout on the sum of embeddings + positional encoding (paper, section 5.4)
  },
  {
    // Fixed sinusoidal function of position, not a learned table — no
    // maxTrainedPosition, since the formula is defined for any position
    // (the paper explicitly argues this lets it extrapolate beyond
    // training-time sequence lengths).
    primitive: PositionalEncodingType.Sinusoidal,
    config: { appliedAt: 'embedding' },
    explanation,
  },
  {
    encoder: { explanation, layers: { totalLayers: NUM_ENCODER_LAYERS, blocks: [encoderBlock], pattern: 'uniform' } },
    decoder: { explanation, layers: { totalLayers: NUM_DECODER_LAYERS, blocks: [decoderBlock], pattern: 'uniform' } },
  },
  {
    explanation,
    tiedWithEmbedding: true,
    // Post-norm already normalizes after every sub-layer (including the
    // decoder's last FFN) — unlike pre-norm architectures, there's no
    // extra normalization needed before the output projection.
    finalNormalization: false,
  },
  {
    explanation,
    // Approximate: the paper batched sentence pairs by length rather than
    // training against one fixed maximum sequence length.
    trainedContextLength: 512,
  },
);