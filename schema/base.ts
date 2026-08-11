/**
 * base.ts
 * -------
 * Core abstractions shared across the architecture model.
 */

export interface Explanation {
  whatIsIt: string;
  howIsItBuilt: string;
  whyBuiltThisWay?: string;
}

export interface Explainable {
  explanation: Explanation;
}

/**
 * `primitive` identifies the architectural concept.
 * `config` contains the concrete numbers/configuration.
 * `rationale` explains why THIS model uses this choice.
 */
export interface ArchitecturalChoice<TPrimitive extends string, TConfig> {
  primitive: TPrimitive;
  config: TConfig;
  rationale?: string;
}

/**
 * Describes repeated blocks without instantiating one object per layer.
 *
 *   uniform:
 *     blocks = [llamaBlock]
 *     totalLayers = 32
 *
 *   heterogeneous:
 *     blocks = [denseBlock, sparseBlock]
 *     pattern = [0, 1, 0, 1, ...]
 */
export interface LayerPattern<TBlock> {
  totalLayers: number;
  blocks: TBlock[];
  pattern: 'uniform' | number[];
}

export function resolveLayer<TBlock>(lp: LayerPattern<TBlock>, layerIndex: number): TBlock {
  if (layerIndex < 0 || layerIndex >= lp.totalLayers) {
    throw new RangeError(`Layer index ${layerIndex} outside [0, ${lp.totalLayers - 1}]`);
  }

  if (lp.pattern === 'uniform') {
    if (lp.blocks.length !== 1) {
      throw new Error(`A uniform LayerPattern must contain exactly one block.`);
    }

    const firstBlock = lp.blocks[0];
    if (firstBlock === undefined) {
      throw new Error(`Uniform LayerPattern is missing its single block.`);
    }

    return firstBlock;
  }

  const blockIndex = lp.pattern[layerIndex];

  if (blockIndex === undefined) {
    throw new RangeError(`No pattern entry for layer ${layerIndex} ` + `(totalLayers=${lp.totalLayers})`,
    );
  }

  const block = lp.blocks[blockIndex];

  if (block === undefined) {
    throw new RangeError(`Pattern for layer ${layerIndex} references invalid block ${blockIndex}.`);
  }

  return block;
}