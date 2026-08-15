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
  type TransformerBlock,
} from 'schema';

const NUM_LAYERS = 96;
const DIM = 12288;
const NUM_HEADS = 96;
const HEAD_DIM = DIM / NUM_HEADS; // 128
const FFN_HIDDEN = DIM * 4; // 49152

const explanation = {
  whatIsIt: '',
  howIsItBuilt: '',
};

function makeBlock(pattern: AttentionPattern): TransformerBlock {
  return {
    explanation,
    attention: {
      mechanism: {
        primitive: AttentionMechanism.MHA,
        config: { numQueryHeads: NUM_HEADS, numKeyValueHeads: NUM_HEADS, headDim: HEAD_DIM },
        explanation,
      },
      pattern: {
        primitive: pattern,
        config: { causal: true, windowSize: pattern === AttentionPattern.LocalBanded ? 256 : undefined },
        explanation,
      },
      kernel: { primitive: AttentionKernel.Softmax, config: {}, explanation },
    },
    feedForward: {
      primitive: FFNType.Dense,
      config: {
        hiddenDim: FFN_HIDDEN,
        activation: { primitive: ActivationFunction.GELU, config: {} },
        gating: { primitive: FFNGating.None, config: {} },
      },
      explanation,
    },
    normalization: {
      primitive: NormType.LayerNorm,
      config: { placement: NormPlacement.PreNorm, epsilon: 1e-5 },
      explanation,
    },
    residual: { primitive: ResidualType.Standard, config: { numStreams: 1 }, explanation },
    sublayerOrder: ['norm', 'attention', 'residual_add', 'norm', 'ffn', 'residual_add'],
    attentionDropout: 0.1,
    residualDropout: 0.1,
  };
}

// GPT-3 alternates dense and locally-banded-sparse attention layers.
const denseBlock = makeBlock(AttentionPattern.Dense);
const sparseBlock = makeBlock(AttentionPattern.LocalBanded);
const pattern: number[] = Array.from({ length: NUM_LAYERS }, (_, i) => (i % 2 === 0 ? 0 : 1));

export const gpt3 = new LLMArchitecture(
  {
    explanation,
    name: 'GPT-3',
    organization: 'OpenAI',
    releaseDate: '2020-05-28',
    totalParameters: 175_000_000_000,
  },
  { explanation, inputModality: 'text', outputType: 'next_token_logits', vocabSize: 50257, maxSequenceLength: 2048 },
  { explanation, algorithm: TokenizerAlgorithm.ByteLevelBPE, vocabSize: 50257, specialTokens: {} },
  { explanation, dimension: DIM, tiedWithOutputHead: true, embeddingDropout: 0.1 },
  {
    primitive: PositionalEncodingType.LearnedAbsolute,
    config: { appliedAt: 'embedding', maxTrainedPosition: 2048 },
    explanation,
  },
  { decoder: { explanation, layers: { totalLayers: NUM_LAYERS, blocks: [denseBlock, sparseBlock], pattern } } },
  { explanation, tiedWithEmbedding: true, finalNormalization: true },
  { explanation, trainedContextLength: 2048 },
);