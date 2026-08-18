/* model.js — an instrumented transformer forward pass, in the browser.
 *
 * Mottled needs something no chat runtime exposes: the **residual stream after
 * every block**. A runtime that only returns generated text is useless here,
 * so the forward pass is implemented rather than borrowed, and the residual is
 * recorded as it is written — `hidden[0]` is the embedding stream and
 * `hidden[l+1] = hidden[l] + attn[l] + mlp[l]`, exactly the layout `capture.py`
 * produces and `StateTrajectory` expects.
 *
 * Architecture: the Llama/Qwen3 family — RMSNorm, rotary embeddings, grouped-
 * query attention, SwiGLU. That covers the open-weight models worth pointing
 * this at (Qwen3 and its quantised derivatives, Llama-3, Mistral); Qwen3's
 * per-head q/k normalisation is supported as an optional weight.
 *
 * Backends: the maths lives behind a small `ops` object. `cpuOps` below is the
 * reference implementation — plain typed arrays, no dependencies, verified
 * against HuggingFace's own outputs (`tests/test_model_conformance.py`). A
 * WebGPU backend implements the same interface for speed; because both satisfy
 * one contract, the GPU path is an optimisation, never a second source of
 * truth.
 *
 * UMD: `window.MottledModel` in the browser, `module.exports` under Node. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.MottledModel = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---- reference ops --------------------------------------------------

  const cpuOps = {
    /* y = x W^T, with W stored (out, in) as HuggingFace's Linear does. */
    matmul(x, w, m, k, n) {
      const out = new Float32Array(m * n);
      for (let i = 0; i < m; i++) {
        for (let o = 0; o < n; o++) {
          let acc = 0;
          const xr = i * k, wr = o * k;
          for (let j = 0; j < k; j++) acc += x[xr + j] * w[wr + j];
          out[i * n + o] = acc;
        }
      }
      return out;
    },

    /* RMSNorm: x * rsqrt(mean(x^2) + eps) * weight.
     * HF computes the reduction in float32 regardless of the model dtype,
     * which is why this stays float32 even when weights are quantised. */
    rmsNorm(x, weight, rows, dim, eps) {
      const out = new Float32Array(rows * dim);
      for (let r = 0; r < rows; r++) {
        let ss = 0;
        for (let j = 0; j < dim; j++) { const v = x[r * dim + j]; ss += v * v; }
        const scale = 1 / Math.sqrt(ss / dim + eps);
        for (let j = 0; j < dim; j++)
          out[r * dim + j] = x[r * dim + j] * scale * weight[j];
      }
      return out;
    },

    /* SwiGLU: down(silu(gate(x)) * up(x)). */
    swiglu(gate, up, rows, inter) {
      const out = new Float32Array(rows * inter);
      for (let i = 0; i < rows * inter; i++) {
        const g = gate[i];
        out[i] = (g / (1 + Math.exp(-g))) * up[i];
      }
      return out;
    },

    add(a, b) {
      const out = new Float32Array(a.length);
      for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
      return out;
    },
  };

  // ---- rotary embeddings ----------------------------------------------

  /* HF's LlamaRotaryEmbedding: inv_freq[i] = 1 / base^(2i/headDim), the angle
   * table is the frequencies concatenated with themselves, and the rotation
   * pairs element j with j + headDim/2 (`rotate_half`) — not adjacent pairs.
   * Getting that pairing wrong produces a model that still runs and quietly
   * predicts nonsense, so it is pinned by the conformance test. */
  function ropeTable(seqLen, headDim, base, positions) {
    const half = headDim >> 1;
    const cos = new Float32Array(seqLen * headDim);
    const sin = new Float32Array(seqLen * headDim);
    for (let t = 0; t < seqLen; t++) {
      const pos = positions ? positions[t] : t;
      for (let i = 0; i < half; i++) {
        const freq = pos / Math.pow(base, (2 * i) / headDim);
        const c = Math.cos(freq), s = Math.sin(freq);
        cos[t * headDim + i] = c;
        cos[t * headDim + i + half] = c;
        sin[t * headDim + i] = s;
        sin[t * headDim + i + half] = s;
      }
    }
    return { cos, sin };
  }

  function applyRope(x, nHeads, seqLen, headDim, cos, sin) {
    const half = headDim >> 1;
    const out = new Float32Array(x.length);
    for (let t = 0; t < seqLen; t++) {
      for (let h = 0; h < nHeads; h++) {
        const off = (t * nHeads + h) * headDim;
        for (let i = 0; i < headDim; i++) {
          const rot = i < half ? -x[off + i + half] : x[off + i - half];
          out[off + i] = x[off + i] * cos[t * headDim + i] +
                         rot * sin[t * headDim + i];
        }
      }
    }
    return out;
  }

  // ---- attention ------------------------------------------------------

  /* Causal grouped-query attention. Query heads are mapped onto key/value
   * heads by integer division, which is how HF's `repeat_kv` expands them. */
  function attention(q, k, v, seqLen, nHeads, nKvHeads, headDim) {
    const out = new Float32Array(seqLen * nHeads * headDim);
    const scale = 1 / Math.sqrt(headDim);
    const group = nHeads / nKvHeads;
    const scores = new Float32Array(seqLen);

    for (let h = 0; h < nHeads; h++) {
      const kv = Math.floor(h / group);
      for (let t = 0; t < seqLen; t++) {
        const qo = (t * nHeads + h) * headDim;
        let mx = -Infinity;
        for (let s = 0; s <= t; s++) {          // causal: only the prefix
          const ko = (s * nKvHeads + kv) * headDim;
          let dot = 0;
          for (let d = 0; d < headDim; d++) dot += q[qo + d] * k[ko + d];
          dot *= scale;
          scores[s] = dot;
          if (dot > mx) mx = dot;
        }
        let sum = 0;
        for (let s = 0; s <= t; s++) { scores[s] = Math.exp(scores[s] - mx); sum += scores[s]; }
        const oo = (t * nHeads + h) * headDim;
        for (let s = 0; s <= t; s++) {
          const w = scores[s] / sum;
          const vo = (s * nKvHeads + kv) * headDim;
          for (let d = 0; d < headDim; d++) out[oo + d] += w * v[vo + d];
        }
      }
    }
    return out;
  }

  // ---- forward pass ---------------------------------------------------

  /* Run `tokenIds` through the model, recording the residual stream.
   *
   * Returns hidden as a flat Float32Array of (nLayers+1) * T * D — the same
   * [layer][token][dim] order `capture.py` returns — plus the final logits and
   * the per-block attention/MLP writes when `captureComponents` is set (the
   * exact additive decomposition the README pins).
   *
   * `weights` supplies flat Float32Arrays by name; a quantised loader is free
   * to dequantise lazily as long as it returns the same shapes. */
  function forward(weights, tokenIds, config, opts = {}) {
    const ops = opts.ops || cpuOps;
    const T = tokenIds.length;
    const D = config.hiddenSize;
    const L = config.numLayers;
    const headDim = config.headDim || (D / config.numHeads);
    const eps = config.rmsNormEps ?? 1e-6;
    const nH = config.numHeads, nKv = config.numKvHeads ?? nH;
    const inter = config.intermediateSize;

    const hidden = new Float32Array((L + 1) * T * D);
    const components = opts.captureComponents
      ? { attn: new Float32Array(L * T * D), mlp: new Float32Array(L * T * D) }
      : null;

    // Layer 0 is the embedding stream, before any block has written to it.
    const embed = weights.get("embed_tokens");
    let h = new Float32Array(T * D);
    for (let t = 0; t < T; t++)
      h.set(embed.subarray(tokenIds[t] * D, (tokenIds[t] + 1) * D), t * D);
    hidden.set(h, 0);

    const { cos, sin } = ropeTable(T, headDim, config.ropeTheta ?? 10000.0,
                                   opts.positions);

    for (let l = 0; l < L; l++) {
      const p = `layers.${l}.`;

      // --- attention block
      let x = ops.rmsNorm(h, weights.get(p + "input_layernorm"), T, D, eps);
      let q = ops.matmul(x, weights.get(p + "q_proj"), T, D, nH * headDim);
      let k = ops.matmul(x, weights.get(p + "k_proj"), T, D, nKv * headDim);
      const v = ops.matmul(x, weights.get(p + "v_proj"), T, D, nKv * headDim);

      // Qwen3 normalises each head before the rotation; Llama has no such
      // weights and the step is simply absent.
      const qNorm = weights.get(p + "q_norm", true);
      const kNorm = weights.get(p + "k_norm", true);
      if (qNorm) q = ops.rmsNorm(q, qNorm, T * nH, headDim, eps);
      if (kNorm) k = ops.rmsNorm(k, kNorm, T * nKv, headDim, eps);

      q = applyRope(q, nH, T, headDim, cos, sin);
      k = applyRope(k, nKv, T, headDim, cos, sin);

      const ctx = attention(q, k, v, T, nH, nKv, headDim);
      const attnOut = ops.matmul(ctx, weights.get(p + "o_proj"), T, nH * headDim, D);

      if (components) components.attn.set(attnOut, l * T * D);
      h = ops.add(h, attnOut);

      // --- MLP block
      x = ops.rmsNorm(h, weights.get(p + "post_attention_layernorm"), T, D, eps);
      const gate = ops.matmul(x, weights.get(p + "gate_proj"), T, D, inter);
      const up = ops.matmul(x, weights.get(p + "up_proj"), T, D, inter);
      const act = ops.swiglu(gate, up, T, inter);
      const mlpOut = ops.matmul(act, weights.get(p + "down_proj"), T, inter, D);

      if (components) components.mlp.set(mlpOut, l * T * D);
      h = ops.add(h, mlpOut);

      hidden.set(h, (l + 1) * T * D);
    }

    // Final norm + unembedding. The norm is *not* written back into the
    // residual record: `hidden` is the stream as the blocks left it, which is
    // what the logit lens is then applied to per layer.
    const normed = ops.rmsNorm(h, weights.get("norm"), T, D, eps);
    const head = weights.get("lm_head", true) || embed;   // tied embeddings
    const logits = ops.matmul(normed, head, T, D, config.vocabSize);

    return { hidden, logits, components, nLayers: L + 1, nTokens: T, dim: D };
  }

  /* Logit lens: decode an intermediate layer with the final head.
   *
   * This is a *readout diagnostic*, not evidence the model has committed to a
   * token at that layer — see docs/validity.md. The name says probe, and the
   * viewer labels it that way. */
  function logitLens(hidden, layer, weights, config, opts = {}) {
    const ops = opts.ops || cpuOps;
    const T = opts.nTokens, D = config.hiddenSize;
    const slice = hidden.subarray(layer * T * D, (layer + 1) * T * D);
    const normed = ops.rmsNorm(slice, weights.get("norm"), T, D,
                               config.rmsNormEps ?? 1e-6);
    const head = weights.get("lm_head", true) || weights.get("embed_tokens");
    return ops.matmul(normed, head, T, D, config.vocabSize);
  }

  return { forward, logitLens, cpuOps, ropeTable, applyRope, attention };
});
