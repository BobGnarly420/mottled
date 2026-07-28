#!/usr/bin/env python3
"""Convert a pretrained SAE into Mottled's portable ``.npz`` dictionary.

Mottled *applies* SAEs, it never trains them (a non-goal); this is how you
bring a real, trained dictionary in so the feature overlay shows interpretable
features instead of ``demo_sae``'s random directions.

Two sources:

    # a SAELens release (needs `pip install sae-lens`)
    python scripts/convert_sae.py from-saelens \
        gpt2-small-res-jb blocks.8.hook_resid_pre -o gpt2-res-l8.npz

    # a raw state dict (.safetensors or torch .pt) using the standard
    # W_enc / b_enc / W_dec / b_dec keys
    python scripts/convert_sae.py from-file sae.safetensors -o sae.npz

The output round-trips through ``sae.load_npz`` and drops straight into
``feature_trajectory`` / ``feature_field`` and the UI overlay.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/convert_sae.py ...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sae as S  # noqa: E402


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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", required=True, help="output .npz path")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sl = sub.add_parser("from-saelens", help="fetch a SAELens release")
    sl.add_argument("release")
    sl.add_argument("sae_id")
    sl.add_argument("--device", default="cpu")
    sl.set_defaults(func=_from_saelens)

    ff = sub.add_parser("from-file", help="convert a local state dict")
    ff.add_argument("path")
    ff.set_defaults(func=_from_file)

    args = ap.parse_args(argv)
    sae = args.func(args)
    S.save_npz(sae, args.out)
    print(f"wrote {args.out}: dim={sae.dim}, features={sae.n_features}")


if __name__ == "__main__":
    main()
