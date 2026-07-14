"""Optional capture backend: serve the web viewer + generate scenes on demand.

    python serve.py                      # or: mottled serve
    python serve.py --model gpt2 --port 8000

Standard library only — a ThreadingHTTPServer that serves the repo as static
files (so /viewer/ works exactly like `python -m http.server`) plus a tiny
JSON API the viewer discovers at runtime:

    GET  /api/health          -> {"ok": true, "model": "..."}
    POST /api/scene           -> .mtj scene bytes
         body: {"prompts": ["...", ...]}

The model is chosen server-side at startup (never by the request), captures
run one at a time behind a lock, and scene generation reuses the exact
pipeline the Streamlit explorer uses (`ui.run_scene` -> `statefile`).
Without this backend the viewer is a plain static site; with it, the
browser can generate trajectories directly.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import statefile
from config import MarbleConfig
from ui import run_scene

ROOT = Path(__file__).resolve().parent
_MAX_PROMPTS = 6
_MAX_PROMPT_CHARS = 500


class _Backend:
    """Owns the configured model and serializes capture requests."""

    def __init__(self, model: str = "synthetic"):
        self.model_name = model
        self._lock = threading.Lock()
        self._model = self._tokenizer = None
        if model != "synthetic":
            from capture import load_model

            self._model, self._tokenizer = load_model(model)

    def scene_bytes(self, prompts: list[str]) -> bytes:
        if not prompts:
            raise ValueError("no prompts given")
        if len(prompts) > _MAX_PROMPTS:
            raise ValueError(f"too many prompts (max {_MAX_PROMPTS})")
        if any(len(p) > _MAX_PROMPT_CHARS for p in prompts):
            raise ValueError(f"prompt too long (max {_MAX_PROMPT_CHARS} chars)")
        cfg = MarbleConfig(model=self.model_name, use_cache=False)
        with self._lock:
            result = run_scene(cfg, prompts, model=self._model, tokenizer=self._tokenizer)
        buf = io.BytesIO()
        statefile.save_scene(result, buf)
        return buf.getvalue()


class Handler(SimpleHTTPRequestHandler):
    backend: _Backend  # set by run_server / make_server

    def do_GET(self):
        if self.path.rstrip("/") == "/api/health":
            return self._json(200, {"ok": True, "model": self.backend.model_name})
        return super().do_GET()

    def do_POST(self):
        if self.path.rstrip("/") != "/api/scene":
            return self._json(404, {"error": "unknown endpoint"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            prompts = [str(p).strip() for p in body.get("prompts", []) if str(p).strip()]
            data = self.backend.scene_bytes(prompts)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": str(exc)})
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # quiet: one line per request is plenty
        print(f"{self.address_string()} {fmt % args}")


def make_server(port: int = 0, model: str = "synthetic",
                directory: str | Path = ROOT) -> ThreadingHTTPServer:
    """Build (but don't start) the server; port 0 picks a free port."""
    handler = partial(Handler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    Handler.backend = _Backend(model)
    return server


def run_server(port: int = 8000, model: str = "synthetic") -> None:
    server = make_server(port=port, model=model)
    print(f"mottled serve — http://127.0.0.1:{server.server_address[1]}/viewer/ "
          f"(model: {model})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="synthetic")
    args = parser.parse_args()
    run_server(port=args.port, model=args.model)
