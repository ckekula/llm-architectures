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
  familyOf,
} from './primitives';
import type { TransformerBlock } from './categories';
import { LLMArchitecture, computeParameterAccounting, computeKVCache } from './model';

const NUM_LAYERS = 96;
const DIM = 12288;
const NUM_HEADS = 96;
const HEAD_DIM = DIM / NUM_HEADS; // 128
const FFN_HIDDEN = DIM * 4; // 49152

const explanation = { whatIsIt: '', howIsItBuilt: '' };

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

const gpt3 = new LLMArchitecture(
  {
    explanation,
    name: 'GPT-3',
    organization: 'OpenAI',
    releaseDate: '2020-05-28',
    totalParameters: 175_000_000_000, // reported figure, independent of the computed total below
  },
  { explanation, inputModality: 'text', outputType: 'next_token_logits', vocabSize: 50257, maxSequenceLength: 2048 },
  { explanation, algorithm: TokenizerAlgorithm.ByteLevelBPE, vocabSize: 50257, specialTokens: {} },
  { explanation, dimension: DIM, tiedWithOutputHead: true, embeddingDropout: 0.1 },
  {
    primitive: PositionalEncodingType.LearnedAbsolute,
    config: { appliedAt: 'embedding', maxTrainedPosition: 2048 },
    explanation,
  },
  { explanation, layers: { totalLayers: NUM_LAYERS, blocks: [denseBlock, sparseBlock], pattern } },
  { explanation, tiedWithEmbedding: true, finalNormalization: true },
  { explanation, trainedContextLength: 2048 },
);

console.log('positional encoding family:', familyOf(gpt3.positionalEncoding.primitive));

const accounting = computeParameterAccounting(gpt3);
console.log('computed total params (B):', (accounting.total / 1e9).toFixed(1));
console.log('reported total params (B):', (gpt3.overview.totalParameters / 1e9).toFixed(1));

const kvCache1 = computeKVCache(gpt3, 1); // bytes for one token, all layers
console.log('KV cache bytes/token (all layers, fp16 default):', kvCache1.totalBytes);

const kvCache2048 = computeKVCache(gpt3, 2048);
console.log('KV cache bytes for 2048-token context (fp16 default):', kvCache2048.totalBytes);
console.log('KV cache explanation:', kvCache2048.explanation.whatIsIt);

console.log('positionalEncoding param share:', accounting.positionalEncoding);
console.log('normalization param share:', accounting.normalization);
console.log('parameterAccounting explanation:', accounting.explanation.whatIsIt);

import { buildArchitectureGraph } from './graph';
const graph = buildArchitectureGraph(gpt3);
console.log('\n--- graph ---');
console.log('top-level nodes:', graph.nodes.map((n) => n.id));
const layerStack = graph.nodes.find((n) => n.id === 'layerStack')!;
console.log('layerStack children (distinct blocks):', layerStack.children?.map((c) => `${c.id} (${JSON.stringify(c.meta)})`));
const blockA = layerStack.children![0]!;
console.log('Block A steps:', blockA.children?.map((c) => c.id + ':' + c.label));
const attnStep = blockA.children!.find((c) => c.category === 'attention')!;
console.log('Block A attention children:', attnStep.children?.map((c) => `${c.category}=${c.label} (${c.path})`));
console.log('total edges:', graph.edges.length);
console.log('block-transition edges:', graph.edges.filter((e) => e.id.startsWith('block-transition')));
console.log('sample intra-block edge:', graph.edges.find((e) => e.id.startsWith('block-0-edge')));