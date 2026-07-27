# Mottled web viewer

A self-contained WebGL2 viewer for `.mtj` **scene** files — no dependencies,
no build step, no network access beyond fetching the file itself.

## Run

From the repo root:

```sh
python -m http.server 8000
```

then open <http://localhost:8000/viewer/>. On startup the viewer tries
`samples/scene-abc.mtj`; if that is missing it falls back to a drop prompt.

Loading a file:

- drag a `.mtj` anywhere onto the page, or
- click **Open .mtj…**, or
- pass a page-relative URL: `http://localhost:8000/viewer/?file=samples/single.mtj`.

Files of `kind: "trajectory"` are recognized but not rendered — the viewer
asks you to export a `kind: "scene"` file instead (scenes carry the projected
terrain and draped trajectory points; raw trajectories don't).

## Controls

| input | action |
|---|---|
| left-drag | orbit |
| right-drag or shift-drag | pan |
| wheel | zoom |
| hover near a trajectory point | highlight + inspector (run, layer, token, entropy, neighborhood fidelity, top-k readout) |
| slider / play button | scrub or animate the marbles across layers |
| runs panel checkboxes | show / hide individual runs |
| attention flow toggle | draw top-3 attention edges (weight ≥ 0.1) at the current layer |
| uncertainty toggle | recolor the terrain by the density's bootstrap standard error (amber = less certain); appears only when the scene carries an `se` field |

The uncertainty controls surface two honesty signals the format now carries.
The **uncertainty toggle** washes the terrain toward amber where the density
estimate is least stable across bootstrap resamples — the height there is
bandwidth artifact more than measurement. The inspector's **nbhd preserved**
line reports how much of a state's hidden-space neighborhood survived the 2-D
projection at the point you hover: a low percentage means the flattened
position is not to be trusted.

Run A is drawn solid; runs B, C, … get dash patterns and reduced opacity,
with labels prefixed `B · token` etc. Comparison summaries (`hausdorff`,
`dtw_normalized`, `shared_tokens`) appear under the runs list when present.

## What the viewer expects from the format

Everything is per `docs/mtj-format.md`, version 1:

- container: `MTRJ` magic, u32 LE version `1`, u32 LE manifest length,
  UTF-8 JSON manifest padded so the blob starts 16-byte aligned, then raw
  little-endian arrays at 16-byte-aligned offsets relative to the blob start;
- manifest `kind: "scene"` with `terrain.{x,y,z}` array references
  (`(W,)`, `(H,)`, `(H, W)`; `z[i][j]` is the height at `(x[j], y[i])`) plus
  optional `terrain.{density,se}` `(H, W)` uncertainty layers;
- `runs[]`, each with a required `points` array `(N, L, 3)` and optional
  `entropy` `(L, T)`, `quality` `(L, T)` (projection fidelity),
  `attention` `(L-1, T, T)`, and manifest `topk`
  `[L][T][k]` of `[token, prob]` pairs;
- array references are resolved strictly through `manifest.arrays` — never by
  the `run{i}.` naming convention;
- unknown manifest fields and unknown dtypes are ignored (forward
  compatibility); supported dtypes are `float32`, `int32`, `float16`
  (decoded to `Float32Array`).

Trajectory polylines are densified in the viewer with Catmull-Rom splines
(8 segments per layer span) for smooth lines and marble animation, as the
spec prescribes — fine paths are not stored in the file.

## Files

- `mtj.js` — parser (`MTJ.parse`, `MTJ.loadScene`); works in the browser and
  under plain Node for testing.
- `main.js`, `index.html`, `style.css` — the viewer app.
- `tests/parser.test.js` — run with `node --test viewer/tests/` from the
  repo root.
