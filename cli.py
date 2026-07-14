"""Console entry points: `mottled` launches the explorer.

    mottled                    # Streamlit explorer (default)
    mottled serve              # stdlib web server: viewer + capture API
    mottled export PROMPT ...  # capture prompts -> scene.mtj on stdout/file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _app_path() -> str:
    import ui

    return str(Path(ui.__file__).resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mottled", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("app", help="launch the Streamlit explorer (default)")

    p_serve = sub.add_parser("serve", help="serve the web viewer + capture API")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--model", default="synthetic",
                         help="model for /api captures (default: synthetic)")

    p_export = sub.add_parser("export", help="capture prompts and write a .mtj scene")
    p_export.add_argument("prompts", nargs="+", help="one or more prompts")
    p_export.add_argument("-o", "--output", default="scene.mtj")
    p_export.add_argument("--model", default="synthetic")

    args = parser.parse_args(argv)

    if args.command == "serve":
        from serve import run_server

        run_server(port=args.port, model=args.model)
        return 0

    if args.command == "export":
        import statefile
        from config import MarbleConfig
        from ui import run_scene

        cfg = MarbleConfig(model=args.model, use_cache=False)
        statefile.save_scene(run_scene(cfg, args.prompts), args.output)
        print(f"wrote {args.output}")
        return 0

    from streamlit.web import cli as st_cli

    sys.argv = ["streamlit", "run", _app_path()]
    return st_cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
