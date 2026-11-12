"""Content-addressed file cache for provider responses."""

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CachedResponse:
    """A cache hit: the stored text and the token counts saved with it."""

    response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseCache:
    """JSON-file cache keyed by provider name, model, and prompt.

    The key is the hex SHA-256 of the UTF-8 bytes of
    ``provider_name + "\\0" + model + "\\0" + prompt``. NUL separators keep
    the three fields from running together (``ab`` + ``c`` would otherwise
    hash the same as ``a`` + ``bc``). A field that contains ``\\`` or NUL is
    escaped first (``\\`` → ``\\\\``, NUL → ``\\0``) so an embedded NUL cannot
    shift those separators. Fields without ``\\`` or NUL are unchanged, and
    their hashes stay the same.

    ``put`` writes a temp file in the cache directory and renames it into
    place so readers never see a partial JSON document. ``get`` increments
    ``hits`` or ``misses``.

    Token counts are stored beside the response. A file written before
    those fields existed still hits; the counts are read as zero. Invalid
    token fields are also read as zero and do not turn a good response
    into a miss.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def key(self, provider_name: str, model: str, prompt: str) -> str:
        """Return the cache key for one provider call."""
        material = (
            f"{_escape_cache_field(provider_name)}\0"
            f"{_escape_cache_field(model)}\0"
            f"{_escape_cache_field(prompt)}"
        ).encode()
        return hashlib.sha256(material).hexdigest()

    def get(self, provider_name: str, model: str, prompt: str) -> str | None:
        """Return the cached response, or ``None`` on a miss or a bad file.

        A missing file, an unreadable file, invalid UTF-8, invalid JSON, or a
        document without a string ``response`` is a miss. Callers then treat
        it like a cache miss and may overwrite the file. Hit and miss counts
        are updated once; this delegates to :meth:`get_record`.
        """
        record = self.get_record(provider_name, model, prompt)
        if record is None:
            return None
        return record.response

    def get_record(self, provider_name: str, model: str, prompt: str) -> CachedResponse | None:
        """Return the cached response and its stored token counts.

        Increments ``hits`` or ``misses`` exactly once. Token fields that
        are missing or not non-negative integers are returned as zero.
        """
        path = self._path(provider_name, model, prompt)
        with self._lock:
            if not path.is_file():
                self.misses += 1
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                response = data["response"]
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
                self.misses += 1
                return None
            if not isinstance(data, dict) or not isinstance(response, str):
                self.misses += 1
                return None
            self.hits += 1
            return CachedResponse(
                response=response,
                prompt_tokens=_cached_tokens(data, "prompt_tokens"),
                completion_tokens=_cached_tokens(data, "completion_tokens"),
                total_tokens=_cached_tokens(data, "total_tokens"),
            )

    def put(
        self,
        provider_name: str,
        model: str,
        prompt: str,
        response: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        """Store ``response`` atomically under the content key.

        Token counts are keyword-only and default to zero so existing
        callers that store only the text keep working.
        """
        prompt_tokens = _require_tokens(prompt_tokens, "prompt_tokens")
        completion_tokens = _require_tokens(completion_tokens, "completion_tokens")
        total_tokens = _require_tokens(total_tokens, "total_tokens")
        digest = self.key(provider_name, model, prompt)
        final_path = self.directory / f"{digest}.json"
        payload = {
            "provider": provider_name,
            "model": model,
            "prompt": prompt,
            "response": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        descriptor, tmp_name = tempfile.mkstemp(
            prefix=f".{digest[:12]}.",
            suffix=".tmp",
            dir=self.directory,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            with self._lock:
                os.replace(tmp_path, final_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _path(self, provider_name: str, model: str, prompt: str) -> Path:
        return self.directory / f"{self.key(provider_name, model, prompt)}.json"


def _escape_cache_field(value: str) -> str:
    """Escape ``\\`` and NUL so only the field separators are raw NULs.

    Strings that contain neither are returned unchanged, which keeps the
    historical hash for ordinary provider names, models, and prompts.
    """
    if "\\" not in value and "\0" not in value:
        return value
    return value.replace("\\", "\\\\").replace("\0", "\\0")


def _cached_tokens(data: dict[Any, Any], key: str) -> int:
    value = data.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _require_tokens(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value
