#!/usr/bin/env python3
"""Convert a pretrained SAE into Mottled's portable ``.npz`` dictionary.

Mottled *applies* SAEs, it never trains them (a non-goal); this is how you
bring a real, trained dictionary in so the feature overlay shows interpretable
features instead of ``demo_sae``'s random directions.

Installed with the package as the ``mottled-convert-sae`` command (from a
source checkout without installing, use ``python -m convert_sae``). Two sources:

    # a SAELens release (needs `pip install "mottled[sae]"`)
    mottled-convert-sae from-saelens \
        gpt2-small-res-jb blocks.8.hook_resid_pre -o gpt2-res-l8.npz

    # a raw state dict (.safetensors or torch .pt) using the standard
    # W_enc / b_enc / W_dec / b_dec keys
    mottled-convert-sae from-file sae.safetensors -o sae.npz

The output round-trips through ``sae.load_npz`` and drops straight into
``feature_trajectory`` / ``feature_field`` and the UI overlay.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sae as S


def _load_state_dict(path: str) -> dict:
    p = Path(path)
    if p.suffix == ".safetensors":
        try:
            from safetensors.numpy import load_file
        except ImportError:
            from safetensors.torch import load_file  # type: ignore
        return dict(load_file(str(p)))
    import torch  # only needed for torch checkpoints

    obj = torch.load(str(p), map_location="cpu")
    return dict(obj.get("state_dict", obj) if isinstance(obj, dict) else obj)


def _from_saelens(args) -> S.SAE:
    try:
        from sae_lens import SAE  # optional dependency
    except ImportError:
        raise SystemExit(
            "sae-lens is not installed. Install it (`pip install sae-lens`) or use "
            "`from-file` with an exported state dict.")
    loaded = SAE.from_pretrained(args.release, args.sae_id, device=args.device)
    obj = loaded[0] if isinstance(loaded, tuple) else loaded
    return S.from_sae_lens(obj)


def _from_file(args) -> S.SAE:
    return S.from_state_dict(_load_state_dict(args.path))


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # -o/--out belongs on each subcommand, not the parent: the subparsers action
    # consumes the subcommand name and everything after it, so a required option
    # added to the parent before add_subparsers can never be given on the command
    # line (it would have to precede the subcommand). A shared parent parser keeps
    # the definition in one place.
    out = argparse.ArgumentParser(add_help=False)
    out.add_argument("-o", "--out", required=True, help="output .npz path")

    sl = sub.add_parser("from-saelens", parents=[out], help="fetch a SAELens release")
    sl.add_argument("release")
    sl.add_argument("sae_id")
    sl.add_argument("--device", default="cpu")
    sl.set_defaults(func=_from_saelens)

    ff = sub.add_parser("from-file", parents=[out], help="convert a local state dict")
    ff.add_argument("path")
    ff.set_defaults(func=_from_file)

    # no sae-lens needed: pulls sae_weights.safetensors + cfg.json straight
    # from a SAELens-format HF repo through huggingface_hub's cache
    fh = sub.add_parser("fetch", parents=[out],
                        help="fetch a SAELens-format repo from the HF hub "
                             "(no sae-lens install needed)")
    fh.add_argument("repo_id", nargs="?", default="jbloom/GPT2-Small-SAEs-Reformatted")
    fh.add_argument("subfolder", nargs="?", default="blocks.8.hook_resid_pre")
    fh.add_argument("--revision", default=None)
    fh.set_defaults(func=lambda a: S.fetch_from_hub(a.repo_id, a.subfolder,
                                                    revision=a.revision))
    return ap


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    sae = args.func(args)
    S.save_npz(sae, args.out)
    print(f"wrote {args.out}: dim={sae.dim}, features={sae.n_features}")


if __name__ == "__main__":
    main()
