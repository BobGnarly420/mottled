/* mtj.js — reader for Mottled `.mtj` files (see docs/mtj-format.md).
 * UMD: `window.MTJ` in the browser, `module.exports` under Node. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.MTJ = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const SUPPORTED_VERSION = 1;
  const HEADER_BYTES = 12;

  function decodeFloat16(u16) {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i];
      const s = (h & 0x8000) ? -1 : 1;
      const e = (h >> 10) & 0x1f;
      const m = h & 0x3ff;
      if (e === 0) out[i] = s * m * Math.pow(2, -24);
      else if (e === 31) out[i] = m ? NaN : s * Infinity;
      else out[i] = s * (1 + m / 1024) * Math.pow(2, e - 15);
    }
    return out;
  }

  // Known dtypes; anything else is ignored for forward compatibility.
  const DTYPES = {
    float32: { itemsize: 4, read: (dv, off, n) => new Float32Array(dv.buffer, dv.byteOffset + off, n) },
    int32: { itemsize: 4, read: (dv, off, n) => new Int32Array(dv.buffer, dv.byteOffset + off, n) },
    float16: {
      itemsize: 2,
      read: (dv, off, n) => decodeFloat16(new Uint16Array(dv.buffer, dv.byteOffset + off, n)),
    },
  };

  function toUint8(input) {
    if (input instanceof ArrayBuffer) return new Uint8Array(input);
    if (ArrayBuffer.isView(input)) {
      // Node Buffers can sit unaligned inside a pool; copy so typed-array
      // views at 16-byte-aligned offsets are always constructible.
      if (input.byteOffset % 16 !== 0) return new Uint8Array(input.slice ? input.slice() : input);
      return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    }
    throw new TypeError("MTJ.parse expects an ArrayBuffer or typed array");
  }

  function parse(input) {
    const u8 = toUint8(input);
    if (u8.byteLength < HEADER_BYTES) throw new Error("not a .mtj file: too short for header");
    if (u8[0] !== 0x4d || u8[1] !== 0x54 || u8[2] !== 0x52 || u8[3] !== 0x4a)
      throw new Error('not a .mtj file: bad magic (expected "MTRJ")');
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const version = dv.getUint32(4, true);
    if (version !== SUPPORTED_VERSION)
      throw new Error(`unsupported .mtj version ${version} (this reader supports ${SUPPORTED_VERSION})`);
    const manifestLen = dv.getUint32(8, true);
    if (HEADER_BYTES + manifestLen > u8.byteLength)
      throw new Error("corrupt .mtj file: manifest extends past end of file");

    let manifest;
    try {
      manifest = JSON.parse(new TextDecoder("utf-8").decode(u8.subarray(HEADER_BYTES, HEADER_BYTES + manifestLen)));
    } catch (e) {
      throw new Error("corrupt .mtj file: manifest is not valid JSON (" + e.message + ")");
    }

    const blobStart = HEADER_BYTES + manifestLen; // spec: 12+M is 16-byte aligned
    const blobLen = u8.byteLength - blobStart;
    const arrays = {};
    for (const [name, ref] of Object.entries(manifest.arrays || {})) {
      const dt = DTYPES[ref && ref.dtype];
      if (!dt || !Array.isArray(ref.shape)) continue; // unknown dtype/shape: ignore
      const count = ref.shape.reduce((a, b) => a * b, 1);
      const bytes = count * dt.itemsize;
      if (typeof ref.offset !== "number" || ref.offset < 0 || ref.offset + bytes > blobLen)
        throw new Error(`corrupt .mtj file: array "${name}" is out of range`);
      if (typeof ref.length === "number" && ref.length !== bytes)
        throw new Error(`corrupt .mtj file: array "${name}" length ${ref.length} != shape*itemsize ${bytes}`);
      arrays[name] = {
        dtype: ref.dtype,
        shape: ref.shape.slice(),
        offset: ref.offset,
        data: dt.read(dv, blobStart + ref.offset, count),
      };
    }
    return { manifest, arrays };
  }

  function loadScene(input) {
    const { manifest, arrays } = parse(input);
    if (manifest.kind === "trajectory") {
      const err = new Error(
        'this is a full "trajectory" capture, not a viewer scene — ' +
        're-export it with kind "scene" (analysis baked in) for the web viewer'
      );
      err.kind = "trajectory";
      throw err;
    }
    if (manifest.kind !== "scene")
      throw new Error(`unsupported .mtj kind "${manifest.kind}" (expected "scene")`);

    const resolve = (name, what) => {
      const a = arrays[name];
      if (!a) throw new Error(`corrupt scene: ${what} references missing array "${name}"`);
      return a;
    };

    const t = manifest.terrain || {};
    const terrain = {
      x: resolve(t.x, "terrain.x"),
      y: resolve(t.y, "terrain.y"),
      z: resolve(t.z, "terrain.z"),
      // optional uncertainty layers (writers >= scene-v3)
      density: t.density ? resolve(t.density, "terrain.density") : null,
      se: t.se ? resolve(t.se, "terrain.se") : null,
    };
    if (terrain.z.shape.length !== 2 ||
        terrain.z.shape[0] !== terrain.y.shape[0] || terrain.z.shape[1] !== terrain.x.shape[0])
      throw new Error("corrupt scene: terrain z shape does not match x/y axes");

    const runs = (manifest.runs || []).map((r, i) => {
      const points = resolve(r.points, `run ${i}`);
      if (points.shape.length !== 3 || points.shape[2] !== 3)
        throw new Error(`corrupt scene: run ${i} points must have shape (N, L, 3)`);
      return {
        label: r.label != null ? String(r.label) : String.fromCharCode(65 + i),
        prompt: r.prompt || "",
        tokens: r.tokens || [],
        trajectoryLabels: r.trajectory_labels || r.tokens || [],
        points,
        entropy: r.entropy ? resolve(r.entropy, `run ${i} entropy`) : null,
        quality: r.quality ? resolve(r.quality, `run ${i} quality`) : null,
        attention: r.attention ? resolve(r.attention, `run ${i} attention`) : null,
        topk: r.topk || null,
      };
    });
    if (!runs.length) throw new Error("corrupt scene: no runs");

    return { manifest, arrays, meta: manifest.meta || {}, terrain, runs, comparisons: manifest.comparisons || [] };
  }

  return { parse, loadScene, decodeFloat16, SUPPORTED_VERSION };
});
