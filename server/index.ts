import { getModel, listModels } from 'models';

/**
 * Dev-only CORS: the frontend (Vite, a different port) needs to fetch
 * these routes. Restrict `Access-Control-Allow-Origin` to a real origin
 * before deploying anywhere the API is reachable from the public internet.
 */
function json(data: unknown, status = 200): Response {
  return Response.json(data, {
    status,
    headers: { 'Access-Control-Allow-Origin': '*' },
  });
}

function notFound(message: string): Response {
  return json({ error: message }, 404);
}

const server = Bun.serve({
  port: 3000,
  routes: {
    '/api/models': () => json(listModels()),

    '/api/models/:org/:slug': (req) => {
      const { org, slug } = req.params;
      const arch = getModel(org, slug);
      if (!arch) return notFound(`No model found for ${org}/${slug}`);
      return json(arch.toJSON());
    },
  },

  fetch() {
    return notFound('Route not found');
  },
});

console.log(`Server running at http://localhost:${server.port}`);