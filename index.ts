const server = Bun.serve({
  port: 3000,
  fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/") {
      return new Response(JSON.stringify({ message: "Hello from Bun!" }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    if (url.pathname === "/echo" && req.method === "POST") {
      return req.json().then((data) => Response.json({ received: data }));
    }

    return new Response("Not Found", { status: 404 });
  },
});

console.log(`Server running at http://localhost:${server.port}`);
