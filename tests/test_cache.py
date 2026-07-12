import numpy as np

from cache import DiskCache, make_key


def test_key_stability_and_sensitivity():
    a = make_key("prompt", model="synthetic", k=5)
    assert a == make_key("prompt", model="synthetic", k=5)
    assert a != make_key("prompt", model="synthetic", k=6)
    assert len(a) == 32


def test_roundtrip(tmp_path):
    cache = DiskCache(tmp_path)
    key = make_key("x")
    assert key not in cache
    payload = {"arr": np.arange(6).reshape(2, 3), "meta": {"k": 1}}
    cache.put(key, payload)
    assert key in cache
    out = cache.get(key)
    assert np.array_equal(out["arr"], payload["arr"]) and out["meta"] == {"k": 1}
    cache.clear()
    assert key not in cache


def test_corrupt_entry_dropped(tmp_path):
    cache = DiskCache(tmp_path)
    key = make_key("bad")
    (tmp_path / f"{key}.pkl").write_bytes(b"not a pickle")
    assert cache.get(key, default="fallback") == "fallback"
    assert key not in cache
