"""Closed-model producer: API logprobs -> StateTrajectory.

Most models cannot be opened. A hosted API exposes no residual stream, no
attention, no per-layer readout — only, at best, the top-k logprobs of each
generated token. That is one axis of the picture Mottled draws, and this
module carries it into the same interchange the open-weight backends use, so
the projection / density / terrain / metrics / viewer stack works unchanged.

What the trajectory *is* here, precisely: depth is unavailable, so the
animated axis is **decode time**, and the moving point is the model's own
output distribution. Step ``s`` contributes one vector — the probability
assigned to each observed token, plus a residual bucket holding the mass the
API did not report — so the path traces how the next-token distribution
travels through the simplex as the completion unfolds.

What it is *not*: it is not the residual stream, and no claim about internal
computation can be read off it. ``meta["degraded"]`` marks every such
trajectory, ``meta["absent"]`` lists what the API could not provide, and the
top-k truncation makes the reported entropy a **lower bound** on the true
predictive entropy (``meta["entropy_is_lower_bound"]``). Surfaces that draw
these trajectories are expected to say so.

    steps = [{"token": " Paris", "logprobs": {" Paris": -0.11, " the": -3.4}},
             {"token": ".",      "logprobs": {".": -0.02, ",": -4.1}}]
    traj = from_logprobs(steps, model="gpt-4o-mini")

`from_openai_logprobs` adapts an OpenAI-style chat-completion response into
that neutral shape; any other provider can be mapped the same way.
"""
from __future__ import annotations

import math

import numpy as np

from trajectory import StateTrajectory

RESIDUAL_TOKEN = "⟨unreported⟩"
_MIN_P = 1e-12


def from_logprobs(
    steps: list[dict],
    model: str = "api",
    prompt: str = "",
    top_k: int = 5,
    keep_logits: bool = True,
) -> StateTrajectory:
    """Build a StateTrajectory from per-step top-k logprobs.

    steps : one dict per generated token, ``{"token": str, "logprobs":
            {token: logprob}}``. The chosen token need not appear in its own
            logprobs map (some APIs omit it); it is added at the mass left
            over when it does not.
    Returns a trajectory of shape ``(S, 1, V)``: S decode steps along the
    animated axis, one path, V observed tokens plus the residual bucket.
    """
    if len(steps) < 2:
        raise ValueError("need at least 2 decode steps to form a trajectory")

    chosen = [str(s.get("token", "")) for s in steps]
    maps = [{str(k): float(v) for k, v in (s.get("logprobs") or {}).items()}
            for s in steps]
    if any(not m for m in maps):
        raise ValueError("every step needs a non-empty 'logprobs' map")

    vocab = sorted({tok for m in maps for tok in m} | set(chosen))
    vocab.append(RESIDUAL_TOKEN)          # unreported mass, always last
    index = {tok: i for i, tok in enumerate(vocab)}

    S, V = len(steps), len(vocab)
    probs = np.zeros((S, 1, V), dtype=np.float32)
    reported = np.zeros(S, dtype=np.float64)
    for s, m in enumerate(maps):
        for tok, lp in m.items():
            probs[s, 0, index[tok]] += math.exp(lp)
        total = float(probs[s, 0].sum())
        if total > 1.0 + 1e-6:
            # renormalise rather than pretend: a provider's rounding can push
            # the reported mass slightly past 1
            probs[s, 0] /= total
            total = 1.0
        reported[s] = total
        probs[s, 0, index[RESIDUAL_TOKEN]] = max(0.0, 1.0 - total)

    # logits = log p, so the shared softmax/entropy/top-k path recovers p
    logits = np.log(np.maximum(probs, _MIN_P)).astype(np.float32)
    from capture import _entropy_topk

    entropy, topk = _entropy_topk(logits, vocab, top_k)

    traj = StateTrajectory(
        hidden=probs,
        tokens=["completion"],
        logits=logits.astype(np.float16) if keep_logits else None,
        entropy=entropy,
        topk=topk,
        vocab=vocab,
        # each token is its own axis in distribution space: the honest
        # "embedding" for a simplex, so neighbor queries stay well-defined
        embedding_matrix=np.eye(V, dtype=np.float32),
        meta={
            "backend": "logprobs",
            "model": model,
            "prompt": prompt,
            "degraded": True,
            "axis": "decode",
            "absent": ["residual stream", "per-layer readout", "attention",
                       "residual decomposition"],
            "entropy_is_lower_bound": True,
            "reported_mass": [round(float(x), 6) for x in reported],
            "generation": {
                "prompt_tokens": 0,
                "new_tokens": S,
                "mode": "api",
                "temperature": float("nan"),
                "seed": None,
                "steps": [
                    {"token": chosen[s], "id": int(index.get(chosen[s], -1)),
                     "p": float(probs[s, 0, index[chosen[s]]])
                          if chosen[s] in index else float("nan"),
                     "entropy": float(entropy[s, 0])}
                    for s in range(S)
                ],
            },
        },
    )
    traj.validate()
    return traj


def from_openai_logprobs(response, model: str = "api", prompt: str = "",
                         **kwargs) -> StateTrajectory:
    """Adapt an OpenAI-style chat-completion response.

    Reads ``choices[0].logprobs.content[i]``, each carrying ``token`` and
    ``top_logprobs`` (a list of ``{token, logprob}``). Accepts the SDK's
    objects or the equivalent plain dicts. Other providers can be mapped into
    the neutral ``from_logprobs`` shape the same way.
    """
    def _get(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    choices = _get(response, "choices") or []
    if not choices:
        raise ValueError("response has no choices")
    content = _get(_get(choices[0], "logprobs") or {}, "content") or []
    if not content:
        raise ValueError("response carries no per-token logprobs "
                         "(request them with logprobs=True, top_logprobs=N)")

    steps = []
    for item in content:
        top = _get(item, "top_logprobs") or []
        lp = {str(_get(e, "token")): float(_get(e, "logprob")) for e in top}
        token = str(_get(item, "token", ""))
        own = _get(item, "logprob")
        if own is not None:
            lp.setdefault(token, float(own))
        steps.append({"token": token, "logprobs": lp})
    return from_logprobs(steps, model=model, prompt=prompt, **kwargs)
