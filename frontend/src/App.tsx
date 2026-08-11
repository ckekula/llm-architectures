import { useEffect, useState } from 'react';
import type { LLMArchitecture } from 'schema';
import { fetchModel, fetchModelList, type ModelKey } from './lib/api';
import { ArchitectureDiagram } from './components/architecture-diagram';

export function App() {
  const [models, setModels] = useState<ModelKey[]>([]);
  const [selected, setSelected] = useState<ModelKey | null>(null);
  const [architecture, setArchitecture] = useState<LLMArchitecture | null>(null);
  const [loadedFor, setLoadedFor] = useState<ModelKey | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModelList()
      .then((list) => {
        setModels(list);
        if (list.length > 0) setSelected(list[0]!);
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : 'Failed to load model list'));
  }, []);

  useEffect(() => {
    if (!selected) return;

    fetchModel(selected.org, selected.slug)
      .then((data) => {
        setArchitecture(data);
        setLoadedFor(selected);
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : 'Failed to load model'));
  }, [selected]);

  const isLoading =
    !selected ||
    !architecture ||
    !loadedFor ||
    loadedFor.org !== selected.org ||
    loadedFor.slug !== selected.slug;

  return (
    <div className="flex h-screen w-screen flex-col">
      <header className="flex items-center gap-3 border-b border-gray-200 px-4 py-3">
        <h1 className="text-sm font-semibold text-gray-800">LLM Architecture Encyclopedia</h1>
        <select
          className="rounded border border-gray-300 px-2 py-1 text-sm"
          value={selected ? `${selected.org}/${selected.slug}` : ''}
          onChange={(e) => {
            const [org, slug] = e.target.value.split('/') as [string, string];
            setError(null);
            setSelected({ org, slug });
          }}
        >
          {models.map((m) => (
            <option key={`${m.org}/${m.slug}`} value={`${m.org}/${m.slug}`}>
              {m.org}/{m.slug}
            </option>
          ))}
        </select>
      </header>

      <main className="flex-1">
        {error && <div className="p-4 text-sm text-red-600">{error}</div>}
        {!error && isLoading && <div className="p-4 text-sm text-gray-500">Loading architecture…</div>}
        {!error && !isLoading && architecture && <ArchitectureDiagram architecture={architecture} />}
      </main>
    </div>
  );
}