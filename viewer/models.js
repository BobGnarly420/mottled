/* models.js — the models the hosted viewer offers, and where they cache.
 *
 * "Paste a URL" is not an interface. This is the curated list behind the
 * picker, plus the storage that stops a 400 MB download happening twice.
 *
 * Every entry here was verified before being listed, not assumed:
 *   - the host sends CORS headers a page on another origin can use,
 *   - the file's ggml types are ones `gguf.js` implements,
 *   - `gguf.js` can place *every* tensor in it (an architecture with weights
 *     this stack cannot apply is refused at load, so a listed model that
 *     tripped that guard would be a listed model that cannot run),
 *   - and it embeds its own tokenizer, so a typed prompt can be encoded.
 *
 * Sizes are the real content-length, not an estimate. They are shown before
 * the download starts because several hundred megabytes is a thing a person
 * should get to decline.
 *
 * UMD: `window.MottledModels` in the browser, `module.exports` under Node. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.MottledModels = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const CATALOG = [
    {
      id: "smollm2-360m",
      name: "SmolLM2 360M",
      detail: "32 layers x 960 · llama · instruct",
      bytes: 386_404_992,
      arch: "llama",
      url: "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf",
      note: "Smallest, quickest to load — the one to try first.",
      license: "Apache-2.0",
    },
    {
      id: "qwen3-0.6b",
      name: "Qwen3 0.6B",
      detail: "28 layers x 1024 · qwen3 · GQA + q/k norms",
      bytes: 639_446_688,
      arch: "qwen3",
      url: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
      note: "A current-generation architecture: grouped-query attention and per-head q/k normalisation.",
      license: "Apache-2.0",
    },
    {
      id: "qwen2.5-0.5b",
      name: "Qwen2.5 0.5B",
      detail: "24 layers x 896 · qwen2 · attention bias",
      bytes: 675_000_000,
      arch: "qwen2",
      url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf",
      note: "Has attention biases, which the forward pass applies — the case a reader that ignores them gets silently wrong.",
      license: "Apache-2.0",
    },
  ];

  /* Prompts that show the tool doing something rather than sitting idle.
   * Each is a pair, because a single trajectory is a shape and two
   * trajectories are a comparison — which is what the terrain is for. */
  const EXAMPLES = [
    {
      label: "Capitals (A/B)",
      prompts: ["The capital of France is", "The capital of Germany is"],
      note: "Shared prefix, one token apart — the runs are identical until they are not.",
    },
    {
      label: "The thesis sentence",
      prompts: ["The residual stream moves, turns, and settles"],
      note: "Mottled's own description, through a model.",
    },
    {
      label: "Same question, two framings",
      prompts: ["Water boils at", "The boiling point of water is"],
      note: "Different wording, same fact — do the trajectories converge?",
    },
  ];

  const humanBytes = (n) =>
    n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : `${Math.round(n / 1e6)} MB`;

  // ---- caching ---------------------------------------------------------

  /* The Cache API rather than IndexedDB: these are immutable HTTP responses
   * keyed by URL, which is exactly what it stores, and it keeps them as
   * streams instead of one enormous ArrayBuffer in a transaction.
   *
   * Everything here degrades to "no cache" rather than failing: private
   * windows, storage pressure and browsers with the API disabled all just
   * mean the model downloads again. */
  const CACHE_NAME = "mottled-models-v1";

  async function cacheOpen() {
    try {
      if (typeof caches === "undefined") return null;
      return await caches.open(CACHE_NAME);
    } catch { return null; }
  }

  async function cached(url) {
    const c = await cacheOpen();
    if (!c) return null;
    try { return (await c.match(url)) || null; } catch { return null; }
  }

  async function put(url, response) {
    const c = await cacheOpen();
    if (!c) return false;
    try { await c.put(url, response); return true; }
    catch { return false; }   // quota, most likely; not worth failing over
  }

  async function listCached() {
    const c = await cacheOpen();
    if (!c) return [];
    try { return (await c.keys()).map((r) => r.url); } catch { return []; }
  }

  async function evict(url) {
    const c = await cacheOpen();
    if (!c) return false;
    try { return await c.delete(url); } catch { return false; }
  }

  /* Fetch with progress, serving from cache when the model is already here.
   *
   * `onProgress(received, total, fromCache)` is called as bytes arrive. A
   * cache hit reports instantly at 100% rather than pretending to download,
   * because the honest signal ("this is already here") is the one that makes
   * the second visit feel different from the first. */
  async function fetchWeights(url, onProgress) {
    const hit = await cached(url);
    if (hit) {
      const buf = await hit.arrayBuffer();
      if (onProgress) onProgress(buf.byteLength, buf.byteLength, true);
      return buf;
    }

    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);

    // Tee the stream: one branch feeds the cache, the other is read for
    // progress. Caching the original response would consume the body the
    // reader needs.
    let toCache = null;
    if (res.body && typeof res.body.tee === "function") {
      const [a, b] = res.body.tee();
      toCache = new Response(b, { headers: res.headers, status: res.status });
      Object.defineProperty(res, "_progressBody", { value: a });
    }

    const total = Number(res.headers.get("content-length")) || 0;
    const body = res._progressBody || res.body;

    let buf;
    if (!body) {
      buf = await res.arrayBuffer();
      if (onProgress) onProgress(buf.byteLength, buf.byteLength, false);
    } else {
      const reader = body.getReader();
      const chunks = [];
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        got += value.length;
        if (onProgress) onProgress(got, total, false);
      }
      const merged = new Uint8Array(got);
      let at = 0;
      for (const c of chunks) { merged.set(c, at); at += c.length; }
      buf = merged.buffer;
    }

    if (toCache) put(url, toCache);   // fire and forget; a miss is not fatal
    return buf;
  }

  return {
    CATALOG, EXAMPLES, humanBytes,
    fetchWeights, cached, listCached, evict, CACHE_NAME,
  };
});
