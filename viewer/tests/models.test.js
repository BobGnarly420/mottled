"use strict";
const test = require("node:test");
const assert = require("node:assert");

const M = require("../models.js");

/* The catalog is a list of buttons a person will click. A typo in a URL is a
 * dead button; a wrong size is a lie told before a 600 MB download. Neither
 * shows up in any other test, so the shape is pinned here.
 *
 * What is NOT checked here: that each URL still resolves, and still holds only
 * ggml types this stack implements. That needs the network. It was verified
 * per entry when the entry was added — including that `gguf.js` can place
 * *every* tensor in the file, since an architecture carrying weights the
 * forward pass cannot apply is refused at load rather than run wrong. */

test("every catalog entry has what the picker renders", () => {
  assert.ok(M.CATALOG.length > 0);
  const ids = new Set();
  for (const m of M.CATALOG) {
    assert.match(m.id, /^[a-z0-9.\-]+$/, `${m.name}: id is a slug`);
    assert.ok(!ids.has(m.id), `${m.id}: ids are unique`);
    ids.add(m.id);
    assert.ok(m.name && m.detail && m.note, `${m.id}: has display text`);
    assert.ok(Number.isFinite(m.bytes) && m.bytes > 1e6,
      `${m.id}: a real byte count, so the size shown before a download is true`);
    assert.match(m.url, /^https:\/\//, `${m.id}: https only`);
    assert.match(m.url, /\.(gguf|mwt)$/, `${m.id}: a format the readers handle`);
    assert.ok(m.license, `${m.id}: states a license`);
  }
});

test("humanBytes reads the way a download does", () => {
  assert.strictEqual(M.humanBytes(386_404_992), "386 MB");
  assert.strictEqual(M.humanBytes(1_074_000_000), "1.07 GB");
  assert.strictEqual(M.humanBytes(639_446_688), "639 MB");
});

test("cache helpers degrade to no-cache instead of throwing", async () => {
  // Node has no `caches`; every helper must return the empty answer rather
  // than reject, because the same thing happens in a private window or when
  // storage is full.
  assert.strictEqual(await M.cached("https://example.invalid/x.gguf"), null);
  assert.deepStrictEqual(await M.listCached(), []);
  assert.strictEqual(await M.evict("https://example.invalid/x.gguf"), false);
});

test("example prompts are runnable as written", () => {
  assert.ok(M.EXAMPLES.length > 0);
  for (const ex of M.EXAMPLES) {
    assert.ok(ex.label && ex.note, `${ex.label}: labelled and explained`);
    assert.ok(Array.isArray(ex.prompts) && ex.prompts.length >= 1);
    for (const p of ex.prompts) {
      assert.ok(p.trim().length > 0, `${ex.label}: no blank prompts`);
      assert.ok(!p.includes("\n"),
        `${ex.label}: one prompt per entry — the box splits on newlines`);
    }
  }
});
