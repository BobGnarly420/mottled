/* gguf.js — read GGUF models in the browser, including ternary ones.
 *
 * The second weight source for `viewer/model.js`, alongside `.mwt`. GGUF is
 * what the open-weight world actually ships — llama.cpp's container — so
 * supporting it means a model can be pointed at Mottled without a conversion
 * step, and in particular means the ternary (1.58-bit) builds are reachable:
 * they are the only way a 4B model is a ~1 GB download rather than an 8 GB one.
 *
 * Scope: the container, plus the quantisation types those models actually use
 * (F32, F16, Q8_0, TQ1_0, TQ2_0). Anything else fails loudly by name rather
 * than silently mis-reading bytes — a wrong dequant produces a model that runs
 * and predicts nonsense, which is the worst possible failure here.
 *
 * The block layouts follow llama.cpp's reference implementation exactly, and
 * `tests/test_gguf_conformance.py` checks them against the `gguf` package's
 * own dequantiser rather than against my reading of it.
 *
 * UMD: `window.MottledGGUF` in the browser, `module.exports` under Node. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.MottledGGUF = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const MAGIC = 0x46554747;   // "GGUF"
  const QK_K = 256;

  // ggml_type -> name, for the types this reader handles and clear errors for
  // the rest. Numbers are ggml's, not ours.
  const TYPES = {
    0: "F32", 1: "F16", 8: "Q8_0", 34: "TQ1_0", 35: "TQ2_0",
  };

  // GGUF metadata value types.
  const VT = {
    UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5,
    FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11,
    FLOAT64: 12,
  };

  function reader(buffer) {
    const dv = new DataView(buffer);
    let off = 0;
    const api = {
      get offset() { return off; },
      set offset(v) { off = v; },
      u8() { return dv.getUint8(off++); },
      i8() { return dv.getInt8(off++); },
      u16() { const v = dv.getUint16(off, true); off += 2; return v; },
      i16() { const v = dv.getInt16(off, true); off += 2; return v; },
      u32() { const v = dv.getUint32(off, true); off += 4; return v; },
      i32() { const v = dv.getInt32(off, true); off += 4; return v; },
      f32() { const v = dv.getFloat32(off, true); off += 4; return v; },
      f64() { const v = dv.getFloat64(off, true); off += 8; return v; },
      // Lengths and offsets are u64. Values beyond 2^53 cannot be represented
      // exactly in a JS number, but a tensor that large cannot be held in a
      // browser anyway, so Number() is safe here and a throw would be noise.
      u64() { const v = dv.getBigUint64(off, true); off += 8; return Number(v); },
      i64() { const v = dv.getBigInt64(off, true); off += 8; return Number(v); },
      str() {
        const n = api.u64();
        const s = new TextDecoder("utf-8").decode(new Uint8Array(buffer, off, n));
        off += n;
        return s;
      },
    };
    return api;
  }

  function readValue(r, type) {
    switch (type) {
      case VT.UINT8: return r.u8();
      case VT.INT8: return r.i8();
      case VT.UINT16: return r.u16();
      case VT.INT16: return r.i16();
      case VT.UINT32: return r.u32();
      case VT.INT32: return r.i32();
      case VT.FLOAT32: return r.f32();
      case VT.BOOL: return r.u8() !== 0;
      case VT.STRING: return r.str();
      case VT.UINT64: return r.u64();
      case VT.INT64: return r.i64();
      case VT.FLOAT64: return r.f64();
      case VT.ARRAY: {
        const itemType = r.u32();
        const n = r.u64();
        const out = new Array(n);
        for (let i = 0; i < n; i++) out[i] = readValue(r, itemType);
        return out;
      }
      default: throw new Error(`GGUF: unknown metadata value type ${type}`);
    }
  }

  function parse(buffer) {
    const r = reader(buffer);
    if (r.u32() !== MAGIC) throw new Error("not a GGUF file (bad magic)");
    const version = r.u32();
    if (version < 2 || version > 3)
      throw new Error(`GGUF version ${version} unsupported (this reads 2 and 3)`);

    const tensorCount = r.u64();
    const kvCount = r.u64();

    const meta = {};
    for (let i = 0; i < kvCount; i++) {
      const key = r.str();
      meta[key] = readValue(r, r.u32());
    }

    const tensors = {};
    for (let i = 0; i < tensorCount; i++) {
      const name = r.str();
      const nDims = r.u32();
      const dims = [];
      for (let d = 0; d < nDims; d++) dims.push(r.u64());
      const type = r.u32();
      const offset = r.u64();
      tensors[name] = { name, dims, type, offset };
    }

    // The tensor data section starts at the next alignment boundary after the
    // header, and every tensor offset is relative to it.
    const alignment = meta["general.alignment"] ?? 32;
    const dataStart = Math.ceil(r.offset / alignment) * alignment;

    return { version, meta, tensors, dataStart, alignment };
  }

  // ---- dequantisation --------------------------------------------------

  function f16to32(h) {
    const sign = (h & 0x8000) ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024);
    if (exp === 0x1f) return frac ? NaN : sign * Infinity;
    return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
  }

  /* TQ2_0: 256 values per block as 2 bits each (64 bytes) plus one f16 scale.
   * The packing is *not* four consecutive values per byte — it is four planes:
   * within each 32-byte group, shift 0 gives values 0..31, shift 2 the next 32,
   * and so on. Reading it as consecutive pairs produces a plausible-looking
   * tensor that is wrong everywhere. */
  function dequantTQ2_0(bytes, base, nBlocks, out) {
    let o = 0;
    for (let b = 0; b < nBlocks; b++) {
      const p = base + b * 66;
      const d = f16to32(bytes[p + 64] | (bytes[p + 65] << 8));
      for (let g = 0; g < 64; g += 32) {
        for (let shift = 0; shift <= 6; shift += 2) {
          for (let m = 0; m < 32; m++) {
            out[o++] = (((bytes[p + g + m] >> shift) & 3) - 1) * d;
          }
        }
      }
    }
  }

  /* TQ1_0: base-3 packing. Five trits ride in one byte via multiplication by
   * 1, 3, 9, 27, 81 -- deliberately overflowing uint8 -- and are recovered
   * with (x * 3) >> 8. The uint8 wraparound is load-bearing, so the mask
   * below is not defensive: without it the high planes decode wrong. */
  function dequantTQ1_0(bytes, base, nBlocks, out) {
    const POW5 = [1, 3, 9, 27, 81];
    const POW4 = [1, 3, 9, 27];
    let o = 0;
    for (let b = 0; b < nBlocks; b++) {
      const p = base + b * 54;
      const d = f16to32(bytes[p + 52] | (bytes[p + 53] << 8));
      const trit = (v) => (((v * 3) >> 8) - 1) * d;

      // qs[0..32) x 5 planes -> 160 values
      for (let k = 0; k < 5; k++)
        for (let m = 0; m < 32; m++)
          out[o++] = trit((bytes[p + m] * POW5[k]) & 0xff);
      // qs[32..48) x 5 planes -> 80 values
      for (let k = 0; k < 5; k++)
        for (let m = 0; m < 16; m++)
          out[o++] = trit((bytes[p + 32 + m] * POW5[k]) & 0xff);
      // qh[0..4) x 4 planes -> 16 values
      for (let k = 0; k < 4; k++)
        for (let m = 0; m < 4; m++)
          out[o++] = trit((bytes[p + 48 + m] * POW4[k]) & 0xff);
    }
  }

  /* Q8_0: 32 values per block, one f16 scale then 32 int8. */
  function dequantQ8_0(bytes, base, nBlocks, out) {
    let o = 0;
    for (let b = 0; b < nBlocks; b++) {
      const p = base + b * 34;
      const d = f16to32(bytes[p] | (bytes[p + 1] << 8));
      for (let m = 0; m < 32; m++) {
        const q = bytes[p + 2 + m];
        out[o++] = (q > 127 ? q - 256 : q) * d;
      }
    }
  }

  function dequantize(buffer, dataStart, info) {
    const n = info.dims.reduce((a, b) => a * b, 1);
    const bytes = new Uint8Array(buffer);
    const base = dataStart + info.offset;
    const out = new Float32Array(n);

    switch (info.type) {
      case 0: {                                    // F32
        const src = new Float32Array(buffer.slice(base, base + n * 4));
        out.set(src);
        return out;
      }
      case 1: {                                    // F16
        for (let i = 0; i < n; i++)
          out[i] = f16to32(bytes[base + 2 * i] | (bytes[base + 2 * i + 1] << 8));
        return out;
      }
      case 8:
        dequantQ8_0(bytes, base, n / 32, out);
        return out;
      case 34:
        dequantTQ1_0(bytes, base, n / QK_K, out);
        return out;
      case 35:
        dequantTQ2_0(bytes, base, n / QK_K, out);
        return out;
      default:
        throw new Error(
          `GGUF: tensor ${info.name} uses ggml type ${info.type}` +
          `${TYPES[info.type] ? ` (${TYPES[info.type]})` : ""}, which this ` +
          `reader does not implement. Supported: ${Object.values(TYPES).join(", ")}.`);
    }
  }

  // ---- model wiring ----------------------------------------------------

  /* GGUF tensor names -> the names viewer/model.js asks for. */
  const NAME_MAP = {
    "token_embd.weight": "embed_tokens",
    "output_norm.weight": "norm",
    "output.weight": "lm_head",
  };
  const BLOCK_MAP = {
    "attn_q": "q_proj", "attn_k": "k_proj", "attn_v": "v_proj",
    "attn_output": "o_proj", "ffn_gate": "gate_proj", "ffn_up": "up_proj",
    "ffn_down": "down_proj", "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "attn_q_norm": "q_norm", "attn_k_norm": "k_norm",
  };

  /* Suffixes carrying weights the forward pass must apply if they exist.
   * A GGUF holding one this reader cannot place is an architecture it does
   * not actually implement, and loading it anyway produces a model that runs
   * and is wrong everywhere -- so `open` refuses instead. */
  const BIAS_MAP = {
    "attn_q": "q_bias", "attn_k": "k_bias", "attn_v": "v_bias",
    "attn_output": "o_bias", "ffn_gate": "gate_bias", "ffn_up": "up_bias",
    "ffn_down": "down_bias",
  };

  function mapName(gguf) {
    if (NAME_MAP[gguf]) return NAME_MAP[gguf];
    const m = /^blk\.(\d+)\.(.+)\.weight$/.exec(gguf);
    if (m && BLOCK_MAP[m[2]]) return `layers.${m[1]}.${BLOCK_MAP[m[2]]}`;
    const b = /^blk\.(\d+)\.(.+)\.bias$/.exec(gguf);
    if (b && BIAS_MAP[b[2]]) return `layers.${b[1]}.${BIAS_MAP[b[2]]}`;
    return null;
  }

  function configFrom(meta) {
    const arch = meta["general.architecture"];
    if (!arch) throw new Error("GGUF: no general.architecture");
    const g = (k, fallback) => {
      const v = meta[`${arch}.${k}`];
      return v === undefined ? fallback : v;
    };
    const hidden = g("embedding_length");
    const heads = g("attention.head_count");
    return {
      architecture: arch,
      hiddenSize: hidden,
      numLayers: g("block_count"),
      numHeads: heads,
      numKvHeads: g("attention.head_count_kv", heads),
      headDim: g("attention.key_length", Math.floor(hidden / heads)),
      intermediateSize: g("feed_forward_length"),
      vocabSize: (meta["tokenizer.ggml.tokens"] || []).length,
      rmsNormEps: g("attention.layer_norm_rms_epsilon", 1e-6),
      ropeTheta: g("rope.freq_base", 10000.0),
    };
  }

  /* Wrap a GGUF buffer as the `weights` object model.forward expects.
   * Dequantisation is lazy and cached: a ternary 4B expands to ~16 GB in f32,
   * so expanding everything up front is not an option — one weight at a time
   * is, and `evict()` releases them again. */
  function open(buffer) {
    const { meta, tensors, dataStart } = parse(buffer);
    const config = configFrom(meta);
    const byName = new Map();
    const unplaced = [];
    for (const info of Object.values(tensors)) {
      const mapped = mapName(info.name);
      if (mapped) byName.set(mapped, info);
      else unplaced.push(info.name);
    }

    /* Refuse a file holding weights this reader cannot place.
     *
     * The failure this prevents is the quiet one: an architecture with, say,
     * per-head attention biases loads without complaint, every forward pass
     * skips them, and the trajectory drawn is of a model that does not exist.
     * A refusal naming the tensors is strictly better than a picture that
     * looks right. `rope_freqs` is precomputed rotary data this implementation
     * derives itself, so it is expected and not a gap.
     */
    const ignorable = /(^rope_freqs\.weight$|\.rope_freqs\.weight$)/;
    const blocking = unplaced.filter((n) => !ignorable.test(n));
    if (blocking.length) {
      const shown = blocking.slice(0, 4).join(", ");
      throw new Error(
        `GGUF: this file holds ${blocking.length} tensor(s) this reader cannot ` +
        `place (${shown}${blocking.length > 4 ? ", …" : ""}). That means the ` +
        `architecture "${config.architecture}" is not fully implemented here, ` +
        `and loading it would produce a model that runs and is wrong. ` +
        `Refusing rather than mis-reading.`);
    }
    const cache = new Map();

    return {
      config,
      meta,
      raw: tensors,
      names: [...byName.keys()],
      tokenizer: meta["tokenizer.ggml.tokens"]
        ? { pieces: meta["tokenizer.ggml.tokens"],
            merges: meta["tokenizer.ggml.merges"] || null,
            size: meta["tokenizer.ggml.tokens"].length }
        : null,

      get(name, optional) {
        if (cache.has(name)) return cache.get(name);
        const info = byName.get(name);
        if (!info) {
          if (optional) return null;
          throw new Error(`missing weight ${name}`);
        }
        const arr = dequantize(buffer, dataStart, info);
        cache.set(name, arr);
        return arr;
      },

      evict() { cache.clear(); },
    };
  }

  return { open, parse, dequantize, mapName, configFrom, f16to32, TYPES };
});
