"""Response cache hits, misses, key formula, and atomic writes."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from promptregress.cache import ResponseCache


def test_miss_then_hit_and_counters(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    assert cache.get("prov", "model", "hi") is None
    assert cache.misses == 1
    assert cache.hits == 0
    cache.put("prov", "model", "hi", "there")
    assert cache.get("prov", "model", "hi") == "there"
    assert cache.hits == 1
    assert cache.misses == 1
    assert cache.get("prov", "model", "other") is None
    assert cache.misses == 2
    assert cache.get("prov", "other", "hi") is None
    assert cache.get("other", "model", "hi") is None


def test_key_is_sha256_of_nul_separated_fields(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    material = b"prov\0model\0prompt"
    expect = hashlib.sha256(material).hexdigest()
    assert cache.key("prov", "model", "prompt") == expect
    assert len(expect) == 64
    assert cache.key("prov", "model", "prompt") != cache.key("prov", "model", "prompt2")
    assert cache.key("ab", "c", "d") != cache.key("a", "bc", "d")
    cache.put("prov", "model", "prompt", "answer")
    assert (tmp_path / f"{expect}.json").is_file()
    stored = json.loads((tmp_path / f"{expect}.json").read_text(encoding="utf-8"))
    assert stored["response"] == "answer"
    assert stored["prompt"] == "prompt"


def test_put_replaces_previous_value(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    cache.put("p", "m", "q", "first")
    cache.put("p", "m", "q", "second")
    assert cache.get("p", "m", "q") == "second"
    assert list(tmp_path.glob("*.json")) == [tmp_path / f"{cache.key('p', 'm', 'q')}.json"]


def test_put_renames_a_complete_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = ResponseCache(tmp_path)
    seen: dict[str, str] = {}
    real_replace = os.replace

    def spy(src: str, dst: str) -> None:
        text = Path(src).read_text(encoding="utf-8")
        data = json.loads(text)
        assert data["response"] == "hello"
        seen["src"] = src
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    cache.put("p", "m", "prompt", "hello")
    assert seen["src"]
    assert cache.get("p", "m", "prompt") == "hello"
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*")) == []


def test_failed_put_leaves_no_final_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = ResponseCache(tmp_path)

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk")

    monkeypatch.setattr("promptregress.cache.json.dump", boom)
    with pytest.raises(RuntimeError, match="disk"):
        cache.put("p", "m", "prompt", "response")
    assert list(tmp_path.glob("*.json")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_corrupt_file_is_a_miss(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    path = tmp_path / f"{cache.key('p', 'm', 'q')}.json"
    path.write_text("{not json", encoding="utf-8")
    assert cache.get("p", "m", "q") is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_invalid_utf8_cache_file_is_a_miss(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    path = tmp_path / f"{cache.key('p', 'm', 'q')}.json"
    path.write_bytes(b"\xff\xfe{not utf-8")
    assert cache.get("p", "m", "q") is None
    assert cache.misses == 1
    assert cache.hits == 0
    cache.put("p", "m", "q", "recovered")
    assert cache.get("p", "m", "q") == "recovered"
    assert cache.hits == 1
