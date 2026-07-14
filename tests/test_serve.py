"""Capture backend (serve.py) and the `mottled export` CLI."""

import json
import threading
import urllib.error
import urllib.request

import pytest

import statefile
from serve import make_server

PROMPT_A = "the capital of france is"
PROMPT_B = "the capital of germany is"


@pytest.fixture(scope="module")
def base_url():
    server = make_server(port=0, model="synthetic")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


def _post(url: str, payload: dict):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    return urllib.request.urlopen(req)


def test_health(base_url):
    with urllib.request.urlopen(f"{base_url}/api/health") as res:
        body = json.loads(res.read())
    assert body == {"ok": True, "model": "synthetic"}


def test_scene_capture_roundtrip(base_url):
    with _post(f"{base_url}/api/scene", {"prompts": [PROMPT_A, PROMPT_B]}) as res:
        assert res.headers["Content-Type"] == "application/octet-stream"
        data = res.read()
    manifest, _ = statefile.read_container(__import__("io").BytesIO(data))
    assert manifest["kind"] == "scene"
    assert len(manifest["runs"]) == 2
    assert manifest["runs"][0]["prompt"] == PROMPT_A
    assert manifest["comparisons"][0]["shared_tokens"] == 3


def test_scene_rejects_bad_requests(base_url):
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(f"{base_url}/api/scene", {"prompts": []})
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(f"{base_url}/api/scene", {"prompts": ["p"] * 99})
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(f"{base_url}/api/scene", {"prompts": ["x" * 10_000]})
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(f"{base_url}/api/nope", {})
    assert e.value.code == 404


def test_serves_the_viewer_statics(base_url):
    with urllib.request.urlopen(f"{base_url}/viewer/index.html") as res:
        html = res.read().decode()
    assert "capture-row" in html  # the backend-discovered form is in the page
    with urllib.request.urlopen(f"{base_url}/viewer/samples/scene-abc.mtj") as res:
        assert res.read(4) == b"MTRJ"


def test_cli_export(tmp_path):
    from cli import main

    out = tmp_path / "scene.mtj"
    assert main(["export", PROMPT_A, PROMPT_B, "-o", str(out)]) == 0
    scene = statefile.load_scene(out)
    assert len(scene["runs"]) == 2
    assert scene["runs"][1]["prompt"] == PROMPT_B
