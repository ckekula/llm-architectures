import { LLMArchitecture } from 'schema';

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:3000';

export interface ModelKey {
  org: string;
  slug: string;
}

export async function fetchModelList(): Promise<ModelKey[]> {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) {
    throw new Error(`Failed to load model list (${res.status})`);
  }
  return res.json();
}

/**
 * The server serves `LLMArchitecture.toJSON()` — plain data, not a class
 * instance. `fromJSON` reconstructs the instance client-side so
 * `buildArchitectureGraph` (and any other schema function) can be called
 * on it directly.
 */
export async function fetchModel(org: string, slug: string): Promise<LLMArchitecture> {
  const res = await fetch(`${API_BASE}/api/models/${org}/${slug}`);
  if (!res.ok) {
    throw new Error(`Failed to load ${org}/${slug} (${res.status})`);
  }
  const data = await res.json();
  return LLMArchitecture.fromJSON(data);
}