# Field notes — the interpretability landscape, dated

**Verified 2026-08-05.** Orientation for anyone — human or agent — picking
this repo up cold. The field moves fast enough that remembered facts go stale
within a year, so:

**Rule of the file: every claim is either (a) dated and re-verifiable with the
commands at the bottom, or (b) marked secondary and cited.** If a claim has
neither, delete it rather than trust it.

---

## What the field is trying to do

Reverse-engineer what a network *computes*, not what it correlates with. The
working ontology, roughly in order of how settled it is:

| idea | what it claims | how settled |
|---|---|---|
| **Residual stream as a highway** | Blocks read from and write to a shared additive stream | Solid — and mechanically checkable (see traps) |
| **Logit lens** | Decoding an intermediate state with the output head shows what the model "would say" there | Useful, known to mislead in early layers |
| **Superposition** | Models pack more features than dimensions, so single neurons are polysemantic | Widely accepted framing |
| **Linear features** | Concepts are directions, recoverable by dictionary learning (SAEs) | **Contested — see below** |
| **Circuits** | Behaviour decomposes into identifiable subgraphs | Real, but expensive to establish per-behaviour |

The honest summary: **instrumentation is solid, decomposition is contested,
explanation is weakest.**

## The stack (verified versions 2026-08-05)

| library | version | use it when |
|---|---|---|
| `transformer-lens` | 3.6.0 | Standard mech-interp on GPT-style models; activation patching, attention analysis |
| `sae-lens` | 6.47.0 | Training/loading SAEs. Inference works with plain PyTorch, not just TL |
| `nnsight` | 0.7.0 | Non-transformer architectures, or remote execution on models too big to host (NDIF) |
| `circuit-tracer` | 0.5.0 | Attribution graphs / cross-layer transcoders — the open implementation of Anthropic's circuit tracing |
| `pyvene` | 0.1.8 | Intervention-focused, architecture-agnostic |
| `transformers` | 5.14.1 | The substrate everything else wraps |

Mottled deliberately depends on **none** of these at runtime. It reads the
residual stream through plain HF forward hooks and treats TL/SAELens as
optional adapters. That is a design choice about not inheriting anyone's
preprocessing (see trap #1) — not a claim they are unnecessary.

## Public SAE artifacts that actually exist

Checked on the Hub, 2026-08-05. **`gated` refers to the SAE repo, not the
base model — they differ, and that catches people.**

| suite | models | gated | last updated |
|---|---|---|---|
| `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50` | Qwen3 1.7B | no | 2026-05-13 |
| `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100` | Qwen3 8B | no | 2026-05-13 |
| `google/gemma-scope-2b-pt-res` | Gemma-2 2B | no | 2025-01-19 |
| `google/gemma-scope-9b-pt-res` | Gemma-2 9B | no | 2024-12-24 |
| `EleutherAI/sae-llama-3.1-8b-64x` | Llama-3.1 8B | no | 2025-07-22 |
| `jbloom/GPT2-Small-SAEs-Reformatted` | GPT-2 small | no | 2024-08-02 |

**Qwen ship SAEs for their own ungated models, per layer.** That makes
`Qwen3-1.7B-Base` + its SAE the only fully-ungated *modern* feature path —
Gemma-2 and Llama-3 base weights are licence-gated even where the SAEs
aren't. If you need a modern feature demo, start there.

## The shape of the frontier gap

It is not one gap, and calling it a "chasm" flattens four very different
things:

1. **Instrumentation — no gap.** Hooks, logit lens, residual decomposition
   are architecture-agnostic. This repo's decomposition reconciles to
   `0.0000` on Qwen2.5-1.5B (GQA/RoPE/SwiGLU/RMSNorm). Scale costs money,
   not method.
2. **Trained dictionaries — ~2 orders of magnitude.** ~8–9B publicly vs
   ~1T frontier (Kimi K2's weights are 1,029 GB, measured). Lagging roughly
   a generation in time. Compute-bound, and closing.
3. **Explanations — the widest gap, least discussed.** Neuronpedia's GPT-2
   labels were written by **gpt-3.5-turbo** (verified in the API response's
   `explanationModelName`). Auto-interp quality is capped by the explainer,
   so the semantic layer runs on models two generations stale.
4. **Closed weights — a wall, not a gap.** No residual stream at any price.
   This is what `models/logprobs.py` exists for: a deliberately degraded
   producer that declares what it cannot see.

Also worth separating: *"the public ecosystem lags frontier"* is a different
claim from *"interpretability lags frontier."* Labs run frontier-scale
interpretability internally; what's thin is the **shared artifacts**.

## Contested ground — do not present as settled

SAEs are the field's dominant decomposition tool and are under real, current
criticism *(secondary — from the literature, not verified here)*:

- **Incomplete coverage.** A fixed latent budget captures high-frequency
  patterns and starves fine-grained ones; even the largest SAEs give an
  incomplete description of a model's representations.
- **Structured residual error.** Much SAE error is *linearly predictable*
  from the input activations — i.e. unlearned features, not noise — and
  reinserting reconstructions causes substantial substitution loss.
- **Low steerability.** Many latents are neither interpretable nor
  steerable; user-desired concepts are often simply absent.
- **Circuit tracing is partial.** Anthropic's attribution-graph work on
  Claude 3.5 Haiku reportedly gave satisfying insight on roughly a *quarter*
  of tested prompts.

Mech interp was named one of MIT Technology Review's 10 Breakthrough
Technologies for 2026 — the field is ascendant *and* internally unsure. Both
are true; say both.

## Traps this project already hit

Hard-won and specific. Each cost real debugging time.

1. **Preprocessing is not provenance.** A trained SAE fits the activation
   distribution it was trained on — including the *preprocessing*.
   TransformerLens folds LayerNorm and centres weights, which changes
   residual *values* while preserving the model's function. The same
   jbloom SAE reads **~24% reconstruction error on a TL capture and ~342% on
   raw HF states.** Never trust an SAE because it names your model —
   measure the fit (`sae.fit_report`).
2. **CKA saturates on residual streams.** A few very-high-variance
   dimensions shared by every layer dominate the inner products, so raw CKA
   scores ~1.0 for *every* layer pair and its argmax is noise (measured:
   middle rows flat to within 0.001 on GPT-2 vs DistilGPT-2). Z-score per
   dimension first — but know the cost: exact rotation invariance is lost
   (~0.96 for a rotated copy).
3. **Auto-interp labels describe correlates, not computation.** Carry
   `explained_by` with the text. A feature name is a lead.
4. **Token *count* is not token *alignment*.** GPT-2 and Pythia-70m both
   split `"The capital of France is"` into 5 pieces with different
   tokenizers. Pair states on the token **strings**.
5. **`np.eye(V)` is a memory bomb.** Representing "each axis is a vocabulary
   entry" as an identity matrix is 7 GB at V=42k. It killed a process here.
6. **Entropy from truncated top-k is a lower bound**, never the true
   predictive entropy. Label it.

## Where Mottled sits

Not a circuit-discovery tool and does not claim to be. It instruments
**latent dynamics** — where states travel and pile up — and measures how much
of that survives projection to 2-D. For verified causal claims, reach for
`circuit-tracer`, TransformerLens, ACDC/EAP. The design bet is that an honest
map you read *before* those tools is worth having, and that a visualization
which cannot state its own error is worse than none.

## Re-verify (run this before trusting the tables above)

```bash
python - <<'PY'
import json, urllib.request
from huggingface_hub import model_info
pypi = lambda n: json.load(urllib.request.urlopen(
    f'https://pypi.org/pypi/{n}/json', timeout=20))['info']['version']
for p in ['transformer-lens','sae-lens','nnsight','circuit-tracer','pyvene','transformers']:
    print(f"{p:20s} {pypi(p)}")
for r in ['Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50','google/gemma-scope-2b-pt-res',
          'EleutherAI/sae-llama-3.1-8b-64x','jbloom/GPT2-Small-SAEs-Reformatted']:
    i = model_info(r); print(f"{r:48s} gated={i.gated} {str(i.last_modified)[:10]}")
PY
```

Trap #1 is re-checkable in-repo with `pytest -m network -k "gpt2_and_distilgpt2 or real_labels"`,
and the decomposition claim with `pytest -k causally_exact`.

## Sources

Secondary claims above (SAE limitations, circuit tracing, field status) come
from:

- [Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders](https://arxiv.org/pdf/2508.09363)
- [Interpretable and Steerable Concept Bottleneck Sparse Autoencoders](https://arxiv.org/abs/2512.10805)
- [Improving Dictionary Learning with Gated Sparse Autoencoders](https://arxiv.org/pdf/2404.16014)
- [The Circuits Research Landscape — Neuronpedia](https://www.neuronpedia.org/graph/info)
- [Circuit Tracing: A Step Closer to Understanding Large Language Models](https://towardsdatascience.com/circuit-tracing-a-step-closer-to-understanding-large-language-models/)
- [Understanding Mechanistic Interpretability in AI Models — IntuitionLabs](https://intuitionlabs.ai/articles/mechanistic-interpretability-ai-llms)
- [SAELens](https://github.com/decoderesearch/SAELens)
- [A Primer on the Inner Workings of Transformer-based Language Models](https://arxiv.org/pdf/2405.00208)
