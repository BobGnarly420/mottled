/* weights.js — load `.mwt` model weights in the browser.
 *
 * The reader half of `mweights.py`. Same container idea as `.mtj`: a magic, a
 * JSON header, then raw little-endian buffers — so this is a parser, not a
 * runtime, and it has no dependencies.
 *
 * Tensors are dequantised **lazily and once**: a 0.6B model is hundreds of
 * megabytes, and expanding every weight to float32 up front would multiply
 * that in memory for no benefit when a single forward pass touches each
 * weight once. `get(name)` is the interface `model.js` expects, so the model
 * never knows whether it is reading quantised or full-precision weights.
 *
 * UMD: `window.MottledWeights` in the browser, `module.exports` under Node. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.MottledWeights = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const MAGIC = 0x3154574d;   // "MWT1" little-endian

  function parseHeader(buffer) {
    const view = new DataView(buffer);
    if (view.getUint32(0, true) !== MAGIC)
      throw new Error("not a .mwt file (bad magic)");
    const len = view.getUint32(4, true);
    const json = new TextDecoder("utf-8").decode(
      new Uint8Array(buffer, 8, len));
    return { header: JSON.parse(json), base: 8 + len };
  }

  /* Half-precision decode. There is no Float16Array in browsers Mottled
   * targets, so the conversion is explicit: sign, 5-bit exponent, 10-bit
   * mantissa, including the subnormal and inf/NaN cases — a naive version
   * that ignores subnormals silently zeroes the smallest weights. */
  function f16to32(h) {
    const sign = (h & 0x8000) ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024);
    if (exp === 0x1f) return frac ? NaN : sign * Infinity;
    return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
  }

  function decode(buffer, base, entry) {
    const off = base + entry.offset;

    if (entry.dtype === "q8") {
      const [rows, cols] = entry.shape;
      const q = new Int8Array(buffer, off, entry.bytes);
      const scales = new Float32Array(
        buffer.slice(base + entry.scaleOffset,
                     base + entry.scaleOffset + entry.scaleBytes));
      const out = new Float32Array(rows * cols);
      for (let r = 0; r < rows; r++) {
        const s = scales[r];
        for (let c = 0; c < cols; c++) out[r * cols + c] = q[r * cols + c] * s;
      }
      return out;
    }

    if (entry.dtype === "f16") {
      const raw = new Uint16Array(buffer, off, entry.bytes / 2);
      const out = new Float32Array(raw.length);
      for (let i = 0; i < raw.length; i++) out[i] = f16to32(raw[i]);
      return out;
    }

    // f32 buffers are 32-byte aligned by the writer, so this is a view rather
    // than a copy where the underlying buffer allows it.
    return new Float32Array(buffer.slice(off, off + entry.bytes));
  }

  /* Wrap a fetched ArrayBuffer as the `weights` object model.forward expects. */
  function open(buffer) {
    const { header, base } = parseHeader(buffer);
    const cache = new Map();

    return {
      config: header.config,
      tokenizer: header.tokenizer || null,
      quant: header.quant,
      names: Object.keys(header.tensors),

      get(name, optional) {
        if (cache.has(name)) return cache.get(name);
        const entry = header.tensors[name];
        if (!entry) {
          // Tied embeddings: the head is the embedding table. model.js asks
          // for lm_head optionally and falls back, so this stays null rather
          // than resolving the tie here — one place decides, and it is there.
          if (optional) return null;
          throw new Error(`missing weight ${name}`);
        }
        const arr = decode(buffer, base, entry);
        cache.set(name, arr);
        return arr;
      },

      /* Drop dequantised tensors, keeping the compressed buffer. Lets a long
       * session reclaim memory between captures without re-downloading. */
      evict() { cache.clear(); },
    };
  }

  async function fetchModel(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);

    // Stream so a several-hundred-megabyte download can show progress rather
    // than looking hung; falls back to arrayBuffer() where streams or the
    // length header are unavailable.
    const total = Number(res.headers.get("content-length")) || 0;
    if (!res.body || !onProgress) return open(await res.arrayBuffer());

    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress(received, total);
    }
    const merged = new Uint8Array(received);
    let at = 0;
    for (const c of chunks) { merged.set(c, at); at += c.length; }
    return open(merged.buffer);
  }

  return { open, fetchModel, parseHeader, f16to32 };
});
