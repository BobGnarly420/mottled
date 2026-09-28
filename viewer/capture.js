/* capture.js — prompt in, scene out, entirely in the page.
 *
 * The piece that joins the others: tokenizer -> forward pass -> logit lens ->
 * joint projection -> density -> terrain -> a scene object in exactly the
 * shape `MTJ.loadScene` produces, so the renderer cannot tell whether a scene
 * arrived as a file or was computed here a moment ago.
 *
 * That equivalence is the design rule. Nothing downstream of this function
 * knows about models, weights or WebGPU; a browser capture and a `mottled
 * export` capture are the same object, which is why the inspector, the layer
 * scrubber, picking and the comparison table all work on it unchanged.
 *
 * UMD: `window.MottledCapture` in the browser, `module.exports` under Node. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory({
    Model: require("./model.js"),
    Scene: require("./scene.js"),
    Tokenizer: require("./tokenizer.js"),
  });
  else root.MottledCapture = factory({
    Model: root.MottledModel, Scene: root.Scene,
    Tokenizer: root.MottledTokenizer,
  });
})(typeof self !== "undefined" ? self : this, function (deps) {
  "use strict";

  const { Model, Scene, Tokenizer } = deps;

  /* Build the tokenizer a weight source describes. `.mwt` and GGUF carry the
   * same three things under different names, so this is the only place that
   * difference exists. */
  function tokenizerFor(weights) {
    const t = weights.tokenizer;
    if (!t) throw new Error(
      "this model file carries no tokenizer, so a typed prompt cannot be " +
      "encoded. Export with a tokenizer (mottled export-weights) or use a " +
      "GGUF, which embeds one.");
    return Tokenizer.create({
      vocab: t.pieces,
      merges: t.merges || [],
      addedTokens: t.addedTokens || [],
    });
  }

  /* Softmax readout at every layer: what the output head would say if pointed
   * at each intermediate state.
   *
   * This is a *probe diagnostic*, not the model's belief at that layer (see
   * docs/validity.md), and it is also the expensive part of a capture -- one
   * vocabulary-sized matmul per layer per token, which dwarfs the forward
   * pass itself. `onProgress` exists because on a real model this is the part
   * a user waits through. */
  async function readout(weights, config, hidden, nLayers, nTokens, tok, opts) {
    const topK = opts.topK ?? 5;
    const entropy = new Float32Array(nLayers * nTokens);
    const topk = [];

    for (let l = 0; l < nLayers; l++) {
      const logits = await Model.logitLens(hidden, l, weights, config,
                                           { ops: opts.ops, nTokens });
      const perToken = [];
      for (let t = 0; t < nTokens; t++) {
        const row = logits.subarray(t * config.vocabSize, (t + 1) * config.vocabSize);
        const { entropy: h, topk: rows } = Scene.entropyAndTopK(row, topK, null);
        entropy[l * nTokens + t] = h;
        perToken.push(rows.map(([id, p]) => [tok.pieceText(id), p]));
      }
      topk.push(perToken);
      if (opts.onProgress) opts.onProgress("readout", l + 1, nLayers);
    }
    return { entropy, topk };
  }

  /* Capture one or more prompts and assemble them into a single scene.
   *
   * All runs share one projection and one terrain, exactly as
   * `pipeline.run_scene` does in Python -- otherwise two trajectories drawn
   * together would be in different spaces and their apparent distance would
   * mean nothing. */
  async function capture(weights, prompts, opts = {}) {
    const config = weights.config;
    const tok = tokenizerFor(weights);
    const progress = opts.onProgress || (() => {});

    const runs = [];
    for (let i = 0; i < prompts.length; i++) {
      const ids = tok.encode(prompts[i]);
      if (!ids.length) throw new Error(`prompt ${i + 1} is empty after tokenizing`);
      progress("forward", i, prompts.length);

      const out = await Model.forward(weights, ids, config, { ops: opts.ops });
      const { entropy, topk } = await readout(
        weights, config, out.hidden, out.nLayers, out.nTokens, tok, {
          ops: opts.ops, topK: opts.topK,
          onProgress: (phase, a, b) => progress(phase, a, b, i),
        });

      runs.push({
        prompt: prompts[i],
        ids,
        tokens: ids.map((id) => tok.pieceText(id)),
        hidden: out.hidden,
        nLayers: out.nLayers,
        nTokens: out.nTokens,
        dim: out.dim,
        entropy,
        topk,
        model: opts.modelName || config.architecture || "in-browser",
      });
    }

    progress("scene", 0, 1);
    const built = Scene.buildScene(runs, opts);

    // Reshape into the viewer's layout. buildScene drapes in layer-major
    // order (the projection's own order); a run's `points` is per *trajectory*
    // -- one polyline per token, walking the layers -- so this transposes.
    const outRuns = built.runs.map((r, i) => {
      const src = runs[i];
      const { nLayers: L, nTokens: T } = src;
      const points = new Float32Array(T * L * 3);
      for (let t = 0; t < T; t++) {
        for (let l = 0; l < L; l++) {
          const from = (l * T + t) * 3, to = (t * L + l) * 3;
          points[to] = r.draped[from];
          points[to + 1] = r.draped[from + 1];
          points[to + 2] = r.draped[from + 2];
        }
      }
      return {
        label: String.fromCharCode(65 + i),
        prompt: src.prompt,
        tokens: src.tokens,
        trajectoryLabels: src.tokens,
        points: { shape: [T, L, 3], data: points },
        entropy: { shape: [L, T], data: src.entropy },
        quality: null,
        attention: null,
        topk: src.topk,
        generation: null,
        features: null,
        inspector: null,
        model: src.model,
      };
    });

    const g = built.terrain.gridSize;
    return {
      manifest: { format: "in-browser" },
      arrays: null,
      meta: {
        model: outRuns[0].model,
        prompt: prompts[0],
        source: "browser",
        explained_variance: built.explainedVariance,
      },
      terrain: {
        x: { shape: [g], data: Float32Array.from(built.terrain.gridX) },
        y: { shape: [g], data: Float32Array.from(built.terrain.gridY) },
        z: { shape: [g, g], data: Float32Array.from(built.terrain.z) },
        density: { shape: [g, g], data: Float32Array.from(built.landscape.density) },
        se: null,
      },
      runs: outRuns,
      comparisons: [],
    };
  }

  return { capture, tokenizerFor };
});
