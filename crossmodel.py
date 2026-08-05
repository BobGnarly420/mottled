"""Compare *models*, not just prompts.

`compare.py` measures two trajectories against each other, but only inside one
model: the runs must share a layer count and hidden dimension for their states
to be commensurable. Two different models share neither — and, with different
tokenizers, they do not even agree on what the tokens are.

They do share one thing: **the text they read out into**. This module builds
the comparison on that, and nothing else.

    readout_space([traj_a, traj_b])   # both models -> one shared coordinate
                                      # system (distribution over shared vocab)
    compare_models(traj_a, traj_b)    # where their dynamics diverge, per layer
    layer_similarity(traj_a, traj_b)  # which layer of B matches layer l of A
    compare_generations(a, b)         # where their *generations* part company
    forced_divergence(a, b)           # both scored on one fixed text

Two independent views, deliberately:

* **Readout space** is a genuine shared coordinate system — each state becomes
  the next-token distribution it predicts, restricted to the vocabulary both
  models have, with the mass on tokens only one of them knows kept in a
  visible `⟨unshared⟩` bucket rather than renormalised away. Because it is a
  real space, both models' trajectories can be joint-projected and drawn on
  one terrain by the existing pipeline. It says what the models *do*.
* **CKA** compares representational geometry directly and is invariant to
  dimension, rotation and isotropic scale, so it needs no alignment at all —
  but it needs paired states, which only exist when the two runs tokenize the
  same text the same way. It says how the models *represent*, and refuses
  loudly when the pairing does not hold.

Neither is a claim about mechanism: identical readouts can come from
different computations, and a high CKA between two layers means their
similarity structures agree, not that they compute the same function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trajectory import StateTrajectory

UNSHARED_TOKEN = "⟨unshared⟩"
_EPS = 1e-12


# ------------------------------------------------------------ shared vocab
def shared_vocab(trajs: list[StateTrajectory]) -> list[str]:
    """Token strings every trajectory's vocabulary contains, sorted.

    Tokenizers disagree on ids, on merges, and on how whitespace attaches, so
    the intersection is taken over the *strings* — the only representation the
    models actually share.
    """
    if not trajs:
        raise ValueError("shared_vocab needs at least one trajectory")
    sets = []
    for t in trajs:
        if not t.vocab:
            raise ValueError("every trajectory needs a vocab to compare models")
        sets.append(set(t.vocab))
    common = set.intersection(*sets)
    common.discard(UNSHARED_TOKEN)
    if not common:
        raise ValueError("these models share no vocabulary; nothing to compare")
    return sorted(common)


def _readout_probs(traj: StateTrajectory) -> np.ndarray:
    """(L, T, V) softmax of the logit lens over the trajectory's own vocab."""
    if traj.logits is None:
        raise ValueError("readout comparison needs logit-lens logits "
                         "(capture with keep_logits=True)")
    x = traj.logits.astype(np.float64)
    x -= x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    return p / np.maximum(p.sum(axis=-1, keepdims=True), _EPS)


def readout_trajectory(traj: StateTrajectory, vocab: list[str]) -> StateTrajectory:
    """Re-express a trajectory in readout space over `vocab`.

    Every state becomes the next-token distribution it predicts, restricted to
    the shared vocabulary, plus a final `⟨unshared⟩` component holding the
    probability this model puts on tokens the others do not have. That bucket
    is the honesty: two models can look identical on shared tokens while
    spending most of their mass elsewhere, and the picture has to be able to
    show that.
    """
    probs = _readout_probs(traj)                       # (L, T, V_own)
    own = {tok: i for i, tok in enumerate(traj.vocab)}
    cols = [own.get(tok) for tok in vocab]
    if any(c is None for c in cols):
        missing = [v for v, c in zip(vocab, cols) if c is None][:3]
        raise ValueError(f"vocab entries missing from this trajectory: {missing}")

    # a token string can map to several ids (different byte-level spellings);
    # sum their mass so the projection onto shared vocabulary is exact
    dup: dict[str, list[int]] = {}
    for i, tok in enumerate(traj.vocab):
        dup.setdefault(tok, []).append(i)
    shared = np.stack([probs[..., dup[tok]].sum(axis=-1) for tok in vocab], axis=-1)

    unshared = np.clip(1.0 - shared.sum(axis=-1), 0.0, 1.0)
    hidden = np.concatenate([shared, unshared[..., None]], axis=-1).astype(np.float32)

    out_vocab = list(vocab) + [UNSHARED_TOKEN]
    logits = np.log(np.maximum(hidden, _EPS)).astype(np.float32)
    from capture import _entropy_topk

    entropy, topk = _entropy_topk(logits, out_vocab, 5)
    meta = dict(traj.meta)
    meta.update({
        "space": "readout",
        "shared_vocab": len(vocab),
        "source_model": traj.meta.get("model"),
        # the mass this model spends outside the shared vocabulary, per layer
        "unshared_mass": [round(float(m), 6) for m in unshared.mean(axis=1)],
    })
    out = StateTrajectory(
        hidden=hidden,
        tokens=list(traj.tokens),
        logits=logits.astype(np.float16),
        entropy=entropy,
        topk=topk,
        vocab=out_vocab,
        # No embedding matrix. The obvious choice — an identity, since each
        # axis of readout space *is* a vocabulary entry — is O(V^2): for two
        # models sharing 42k tokens that is a 7 GB allocation, which killed
        # the process outright on a 1.5B-parameter pair. It also buys
        # nothing: "the nearest vocabulary token to this state" in readout
        # space is just the state's largest component, which `topk` already
        # reports. Consumers must treat the matrix as optional (it always
        # was, for producers that have no embedding table).
        embedding_matrix=None,
        meta=meta,
    )
    out.validate()
    return out


def readout_space(trajs: list[StateTrajectory]) -> tuple[list[StateTrajectory], list[str]]:
    """Put several models into ONE shared coordinate system.

    Returns (readout trajectories, shared vocabulary). The results share a
    hidden dimension, so `projection.project_joint` — and therefore the whole
    scene pipeline — accepts them even though the source models do not share
    an architecture, a width, or a tokenizer.
    """
    vocab = shared_vocab(trajs)
    return [readout_trajectory(t, vocab) for t in trajs], vocab


# ------------------------------------------------------------ readout drift
@dataclass
class ModelComparison:
    """How two models' readouts differ along their depth."""

    divergence: np.ndarray      # (L,) Jensen-Shannon divergence per layer pair
    shared_vocab: int           # size of the shared vocabulary
    unshared_a: np.ndarray      # (L,) mass model A spends outside it
    unshared_b: np.ndarray      # (L,)
    agree_from: int | None      # first layer (in the common depth) where the
    # two models' top-1 shared-vocab predictions agree and keep agreeing
    top_a: str
    top_b: str

    @property
    def final_divergence(self) -> float:
        return float(self.divergence[-1])


def _js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence (nats) between distributions, last axis."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    def _kl(a, b):
        safe = a > 0
        return np.where(safe, a * np.log(np.where(safe, a, 1.0) /
                                         np.maximum(b, _EPS)), 0.0).sum(axis=-1)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def compare_models(traj_a: StateTrajectory, traj_b: StateTrajectory,
                   token: int = -1) -> ModelComparison:
    """Compare two models' readouts at one token position, layer by layer.

    Defaults to the final token — the prediction position, which both models
    reach having consumed the same text however they chopped it up, so the
    comparison needs no token alignment. Depth is compared on a **normalised**
    axis: models of different depth are resampled onto the shorter one's
    layers, since layer 6 of a 12-layer model and layer 6 of a 32-layer model
    are not the same stage of computation.
    """
    (ra, rb), vocab = readout_space([traj_a, traj_b])
    n = min(ra.n_layers, rb.n_layers)

    def _resample(traj: StateTrajectory) -> np.ndarray:
        tok = token % traj.n_tokens
        series = traj.hidden[:, tok, :]                       # (L, V)
        if traj.n_layers == n:
            return series
        src = np.linspace(0.0, 1.0, traj.n_layers)
        dst = np.linspace(0.0, 1.0, n)
        idx = np.abs(dst[:, None] - src[None, :]).argmin(axis=1)
        return series[idx]

    pa, pb = _resample(ra), _resample(rb)
    # divergence is measured on the shared vocabulary alone (the last column
    # is the unshared bucket, reported separately rather than mixed in)
    div = _js_divergence(_renorm(pa[:, :-1]), _renorm(pb[:, :-1]))

    top_a = pa[:, :-1].argmax(axis=-1)
    top_b = pb[:, :-1].argmax(axis=-1)
    agree = top_a == top_b
    agree_from = None
    for l in range(n):
        if agree[l:].all():
            agree_from = l
            break

    return ModelComparison(
        divergence=div.astype(np.float32),
        shared_vocab=len(vocab),
        unshared_a=pa[:, -1].astype(np.float32),
        unshared_b=pb[:, -1].astype(np.float32),
        agree_from=agree_from,
        top_a=vocab[int(top_a[-1])],
        top_b=vocab[int(top_b[-1])],
    )


def _renorm(p: np.ndarray) -> np.ndarray:
    return p / np.maximum(p.sum(axis=-1, keepdims=True), _EPS)


# ------------------------------------------------------------------- CKA
def cka(x: np.ndarray, y: np.ndarray, standardize: bool = True) -> float:
    """Linear centered kernel alignment between two state matrices.

    `x` is (n, D_a) and `y` is (n, D_b): the *same* n states represented by
    two models. CKA compares the similarity structure rather than the
    coordinates, so it is invariant to rotation, to isotropic scaling, and to
    the two dimensions differing — which is exactly why it can compare models
    with no alignment fitted between them. 1.0 means the two layers order
    their states' similarities identically; 0.0 means unrelated.

    ``standardize`` (default on) z-scores each dimension across the states
    first, and it matters more than it sounds. A residual stream carries a
    few very-high-variance dimensions shared by every layer; without
    standardizing, those dominate the inner products and **every** layer of A
    looks ~1.0 similar to **every** layer of B. Measured on GPT-2 vs
    DistilGPT-2 over 45 tokens, raw CKA leaves the middle rows flat to within
    0.001 — an alignment that is pure noise — while standardized CKA
    recovers a monotone, proportional layer correspondence.

    That default is a **trade**, stated rather than buried: z-scoring keeps
    exact invariance to isotropic scaling but only approximate invariance to
    rotation (a rotated copy scores ~0.96 rather than 1.0, since rotating
    mixes dimensions whose scales are then equalised differently), and it
    lifts low-variance dimensions — including near-noise ones — to the same
    weight as the rest. Pass ``standardize=False`` for the strictly
    invariant number, and expect its argmax over layers to be unusable.

    It says nothing about *mechanism*: agreeing similarity structure is not
    the same computation.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("cka expects two (n, D) matrices")
    if len(a) != len(b):
        raise ValueError(f"cka needs paired states: got {len(a)} and {len(b)}")
    if len(a) < 2:
        raise ValueError("cka needs at least 2 states")
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    if standardize:
        a = a / np.maximum(a.std(axis=0, keepdims=True), _EPS)
        b = b / np.maximum(b.std(axis=0, keepdims=True), _EPS)
    # ||A^T B||_F^2 / (||A^T A||_F ||B^T B||_F) — the linear-kernel form,
    # computed without ever forming the n x n Gram matrices
    hsic = np.linalg.norm(a.T @ b, ord="fro") ** 2
    norm = (np.linalg.norm(a.T @ a, ord="fro") * np.linalg.norm(b.T @ b, ord="fro"))
    if norm <= _EPS:
        return 0.0
    return float(hsic / norm)


@dataclass
class LayerAlignment:
    """Which layer of B best matches each layer of A."""

    matrix: np.ndarray          # (L_a, L_b) CKA
    best: np.ndarray            # (L_a,) argmax over B for each layer of A
    best_score: np.ndarray      # (L_a,) that CKA value
    contrast: np.ndarray        # (L_a,) best minus the row's median — how
    # much the winner actually beats the field. Near zero means the row is
    # flat and its argmax is noise, not a correspondence.
    monotone: bool              # does the best-match path advance monotonically?
    n_states: int               # states the CKA was computed over

    def identified(self, threshold: float = 0.02) -> np.ndarray:
        """(L_a,) mask: layers whose best match actually stands out."""
        return self.contrast >= threshold

    def summary(self, threshold: float = 0.02) -> str:
        """One line stating the alignment *and* whether to believe it."""
        ok = int(self.identified(threshold).sum())
        line = (f"layer alignment over {self.n_states} states: "
                f"{ok}/{len(self.best)} layers identified "
                f"(contrast ≥ {threshold:g})")
        if ok < len(self.best):
            line += " — flat rows mean the match is unresolved, not absent"
        return line + ("; path is monotone" if self.monotone
                       else "; path is not monotone")


def layer_similarity(traj_a: StateTrajectory, traj_b: StateTrajectory,
                     standardize: bool = True) -> LayerAlignment:
    """CKA between every layer of A and every layer of B.

    Needs **paired states**: the two runs must have tokenized the same text
    into the same number of tokens, so state (l, t) of one corresponds to
    state (m, t) of the other. That holds for models sharing a tokenizer
    (GPT-2 vs DistilGPT-2, a base model vs its finetune) and fails across
    tokenizer families — where it refuses rather than pairing states that are
    not counterparts. Use `compare_models` there: readout space needs no
    pairing.

    Two readouts decide whether the result means anything, and both are
    reported rather than assumed:

    * ``contrast`` — how far each row's winner beats that row's median. CKA
      over a residual stream saturates easily, and a flat row's argmax is
      noise. Short prompts flatten it further: a handful of token states in
      several hundred dimensions cannot resolve a correspondence at all, so
      prefer a few dozen states.
    * ``monotone`` — whether the best-match path advances through B as it
      advances through A, i.e. the two models pass through comparable stages
      in the same order.
    """
    # Pairing is decided by the token *strings*, not their count: two
    # tokenizers can land on the same number of pieces while cutting the text
    # in different places, and pairing those states would compare things that
    # are not counterparts. Conversely, different tokenizer families that
    # happen to segment this text identically are legitimately comparable.
    if list(traj_a.tokens) != list(traj_b.tokens):
        if traj_a.n_tokens != traj_b.n_tokens:
            why = (f"the runs tokenized to {traj_a.n_tokens} and "
                   f"{traj_b.n_tokens} tokens")
        else:
            first = next(i for i, (x, y) in enumerate(zip(traj_a.tokens, traj_b.tokens))
                         if x != y)
            why = (f"the runs split the text differently from token {first} "
                   f"({traj_a.tokens[first]!r} vs {traj_b.tokens[first]!r})")
        raise ValueError(
            f"CKA needs paired states, but {why} — state (l, t) of one is not "
            f"the counterpart of state (m, t) of the other. Compare these "
            f"models in readout space instead (compare_models / "
            f"readout_space), which needs no pairing.")
    ha, hb = traj_a.hidden, traj_b.hidden
    matrix = np.array([[cka(ha[l], hb[m], standardize=standardize)
                        for m in range(traj_b.n_layers)]
                       for l in range(traj_a.n_layers)], dtype=np.float32)
    best = matrix.argmax(axis=1)
    return LayerAlignment(
        matrix=matrix,
        best=best.astype(np.int32),
        best_score=matrix.max(axis=1),
        contrast=(matrix.max(axis=1) - np.median(matrix, axis=1)).astype(np.float32),
        monotone=bool(np.all(np.diff(best) >= 0)),
        n_states=int(traj_a.n_tokens),
    )


# ------------------------------------------------------- the decode axis
@dataclass
class GenerationComparison:
    """Where two models' generations part company, step by step.

    `comparable_steps` is the honest boundary: once the models choose
    different tokens their contexts differ, so every later step is each model
    answering a *different question*. Steps past that point are reported
    (they are what the models actually did) but must not be read as a
    measurement of the same quantity.
    """

    tokens_a: list[str]
    tokens_b: list[str]
    agree: np.ndarray            # (S,) did both choose the same token at step s
    p_a: np.ndarray              # (S,) probability each model gave its choice
    p_b: np.ndarray
    entropy_a: np.ndarray        # (S,) spread of each model's step distribution
    entropy_b: np.ndarray
    first_divergence: int | None  # first step where the choices differ
    comparable_steps: int        # steps before the contexts diverge

    def summary(self) -> str:
        n = len(self.agree)
        if self.first_divergence is None:
            return (f"identical generations for all {n} steps "
                    f"(contexts never diverge)")
        return (f"diverge at step {self.first_divergence} "
                f"({self.tokens_a[self.first_divergence]!r} vs "
                f"{self.tokens_b[self.first_divergence]!r}); "
                f"steps 0–{self.comparable_steps - 1} are like-for-like, the "
                f"rest are different continuations, not a comparison")


def _generation_steps(traj: StateTrajectory) -> list[dict]:
    gen = traj.meta.get("generation")
    if not isinstance(gen, dict) or not gen.get("steps"):
        raise ValueError("this trajectory carries no decode record; capture it "
                         "with capture.generate_and_capture (or a producer that "
                         "fills meta['generation'])")
    return gen["steps"]


def compare_generations(traj_a: StateTrajectory,
                        traj_b: StateTrajectory) -> GenerationComparison:
    """Compare two models' free-running generations of the same prompt.

    Both arguments come from `capture.generate_and_capture` on the same
    prompt. Free-running generation is only a like-for-like comparison up to
    the first disagreement: after that the models are continuing *different*
    texts, and a step-by-step divergence number would be comparing answers to
    different questions. That boundary is measured and reported rather than
    glossed over — for a comparison that stays valid all the way down, score
    both models on one fixed continuation with `forced_divergence`.
    """
    sa, sb = _generation_steps(traj_a), _generation_steps(traj_b)
    n = min(len(sa), len(sb))
    if not n:
        raise ValueError("both trajectories need at least one decode step")

    tokens_a = [str(s["token"]) for s in sa[:n]]
    tokens_b = [str(s["token"]) for s in sb[:n]]
    agree = np.array([a == b for a, b in zip(tokens_a, tokens_b)], dtype=bool)
    first = int(np.argmin(agree)) if not agree.all() else None

    return GenerationComparison(
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        agree=agree,
        p_a=np.array([float(s["p"]) for s in sa[:n]], dtype=np.float32),
        p_b=np.array([float(s["p"]) for s in sb[:n]], dtype=np.float32),
        entropy_a=np.array([float(s["entropy"]) for s in sa[:n]], dtype=np.float32),
        entropy_b=np.array([float(s["entropy"]) for s in sb[:n]], dtype=np.float32),
        first_divergence=first,
        comparable_steps=n if first is None else first + 1,
    )


@dataclass
class ForcedDivergence:
    """Two models scored on one fixed text, position by position."""

    divergence: np.ndarray       # (T-1,) JS between next-token distributions
    top_agree: np.ndarray        # (T-1,) same top-1 prediction?
    tokens: list[str]            # the text both models were scored on
    shared_vocab: int
    unshared_a: np.ndarray       # (T-1,) mass each spends outside shared vocab
    unshared_b: np.ndarray

    @property
    def worst_position(self) -> int:
        return int(np.argmax(self.divergence))


def forced_divergence(traj_a: StateTrajectory,
                      traj_b: StateTrajectory) -> ForcedDivergence:
    """Score two models on the *same* text and compare every next-token step.

    Teacher forcing: because both trajectories are captures of one fixed
    token sequence, each model's final-layer readout at position t is its
    prediction for t+1 given identical context. Unlike free-running
    generation, that stays a like-for-like comparison for the whole sequence —
    it answers "would B have said what A said, here?" at every position.

    Needs the two runs to have tokenized the text identically (the same
    condition `layer_similarity` requires, and for the same reason: position t
    must mean the same thing to both).
    """
    if list(traj_a.tokens) != list(traj_b.tokens):
        raise ValueError(
            "forced scoring needs both models to have tokenized the same text "
            "identically, so position t is the same context for both; these "
            f"runs differ ({traj_a.n_tokens} vs {traj_b.n_tokens} tokens). "
            "Compare their free-running generations instead "
            "(compare_generations), or use readout space (compare_models).")

    (ra, rb), vocab = readout_space([traj_a, traj_b])
    # final layer = the model's actual output distribution; positions 0..T-2
    # predict the next token (the last position has no successor in the text)
    pa, pb = ra.hidden[-1, :-1, :], rb.hidden[-1, :-1, :]
    div = _js_divergence(_renorm(pa[:, :-1]), _renorm(pb[:, :-1]))
    return ForcedDivergence(
        divergence=div.astype(np.float32),
        top_agree=(pa[:, :-1].argmax(axis=-1) == pb[:, :-1].argmax(axis=-1)),
        tokens=list(traj_a.tokens),
        shared_vocab=len(vocab),
        unshared_a=pa[:, -1].astype(np.float32),
        unshared_b=pb[:, -1].astype(np.float32),
    )
