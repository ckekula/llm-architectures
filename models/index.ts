/**
 * models/index.ts
 * ----------------
 * The "database": every onboarded model, keyed by `${organization}/${slug}`.
 * Adding a model = adding a data module here and registering it below —
 * no schema or server changes required.
 */
import type { LLMArchitecture } from 'schema';
import { gpt3 } from './openai/gpt3';

export interface ModelKey {
  org: string;
  slug: string;
}

const MODEL_REGISTRY: Record<string, LLMArchitecture> = {
  'openai/gpt-3': gpt3,
};

export function getModel(org: string, slug: string): LLMArchitecture | undefined {
  return MODEL_REGISTRY[`${org}/${slug}`];
}

export function listModels(): ModelKey[] {
  return Object.keys(MODEL_REGISTRY).map((key) => {
    const [org, slug] = key.split('/') as [string, string];
    return { org, slug };
  });
}