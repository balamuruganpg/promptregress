"""Model providers. Tests mock urllib; the demo never touches the network."""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o-mini"
_ENV_API_KEY = "PROMPTREGRESS_API_KEY"
_ENV_BASE_URL = "PROMPTREGRESS_BASE_URL"
_ENV_MODEL = "PROMPTREGRESS_MODEL"
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_ENV_ANTHROPIC_KEY = "ANTHROPIC_API_KEY"
_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
_DEFAULT_MAX_TOKENS = 1024


class ProviderError(RuntimeError):
    """A provider call failed after its retries were exhausted."""


@dataclass(frozen=True)
class TokenUsage:
    """Token counts copied from a provider, or derived from text length.

    ``total_tokens`` is stored as reported. It is not rewritten to
    ``prompt_tokens + completion_tokens`` when the server sends a different
    total.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        for label in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")


@dataclass(frozen=True)
class Completion:
    """One provider response: text plus the usage that belongs to it."""

    text: str
    usage: TokenUsage


def usage_from_text_lengths(prompt: str, output: str) -> TokenUsage:
    """Deterministic pseudo-usage from character lengths.

    ``prompt_tokens`` is ``len(prompt)`` and ``completion_tokens`` is
    ``len(output)``. This is not a model tokenizer. The same strings always
    yield the same counts, which is what offline tests need.
    """
    if not isinstance(prompt, str) or not isinstance(output, str):
        raise TypeError("prompt and output must be strings")
    prompt_tokens = len(prompt)
    completion_tokens = len(output)
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


@runtime_checkable
class Provider(Protocol):
    """Something that turns a prompt string into a completion string."""

    name: str

    def complete(self, prompt: str) -> str:
        """Return the model output for ``prompt``."""


def provider_model_name(provider: Provider) -> str:
    """Model id used for cache keys and history, falling back to ``name``."""
    model = getattr(provider, "model", None)
    if isinstance(model, str) and model:
        return model
    return provider.name


class MockProvider:
    """Deterministic provider: first matching substring rule wins.

    ``rules`` is a list of ``(needle, response)`` pairs. ``needle`` is tested
    with ``needle in prompt``. ``response`` is a string, or a list of strings
    when ``vary`` is enabled. If nothing matches, ``default`` is returned.
    ``call_count`` counts every completion call, including defaults.

    ``vary`` is off by default, so repeated calls return the same text. A
    list response then uses its first string (the rule output). With
    ``vary`` on, a list response cycles: sample index ``i`` returns
    ``alternates[i % len(alternates)]``. A string response stays that string,
    which is the fallback when a rule has no alternates. :meth:`complete` and
    :meth:`complete_detailed` take the next index for that prompt.
    :meth:`complete_at` uses the index it is given, so a runner can pin
    sample ``i`` without depending on call order or other threads.
    """

    def __init__(
        self,
        rules: Sequence[tuple[str, Any]] | None = None,
        default: str = "",
        name: str = "mock",
        model: str = "mock",
        vary: bool = False,
    ) -> None:
        if not isinstance(vary, bool):
            raise TypeError("vary must be a bool")
        if not isinstance(default, str):
            raise TypeError("default must be a string")
        self.rules = [_normalize_rule(rule) for rule in (rules or [])]
        self.default = default
        self.name = name
        self.model = model
        self.vary = vary
        self.call_count = 0
        self._vary_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def complete(self, prompt: str) -> str:
        return self.complete_detailed(prompt).text

    def complete_detailed(self, prompt: str) -> Completion:
        """Return the completion and character-length pseudo-usage.

        ``call_count`` increments once per call, including when
        :meth:`complete` is the caller. With ``vary``, repeated calls of the
        same prompt walk the matched rule's alternate list.
        """
        with self._lock:
            index = self._vary_counts.get(prompt, 0)
            self._vary_counts[prompt] = index + 1
            self.call_count += 1
            text = self._select(prompt, index)
        return Completion(text=text, usage=usage_from_text_lengths(prompt, text))

    def complete_at(self, prompt: str, sample_index: int) -> Completion:
        """Return alternate ``sample_index`` for ``prompt``.

        The index is the caller's sample number, not a shared counter.
        ``call_count`` still increments once. ``vary`` off ignores the index
        and returns the rule's primary output.
        """
        if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
            raise ValueError("sample_index must be an integer >= 0")
        with self._lock:
            self.call_count += 1
            text = self._select(prompt, sample_index)
        return Completion(text=text, usage=usage_from_text_lengths(prompt, text))

    def _select(self, prompt: str, sample_index: int) -> str:
        chosen: str | tuple[str, ...] | None = None
        for needle, response in self.rules:
            if needle in prompt:
                chosen = response
                break
        if chosen is None:
            return self.default
        if isinstance(chosen, str) or not self.vary:
            if isinstance(chosen, str):
                return chosen
            return chosen[0]
        return chosen[sample_index % len(chosen)]


class EchoProvider:
    """Return the prompt unchanged. Useful for smoke tests."""

    name = "echo"
    model = "echo"

    def complete(self, prompt: str) -> str:
        return prompt


class OpenAICompatibleProvider:
    """POST ``{base_url}/chat/completions`` with the stdlib HTTP client.

    ``api_key``, ``base_url``, and ``model`` come from the constructor or, when
    the constructor argument is ``None``, from ``PROMPTREGRESS_API_KEY``,
    ``PROMPTREGRESS_BASE_URL``, and ``PROMPTREGRESS_MODEL``. The default base
    URL is OpenAI's and the default model is ``gpt-4o-mini``. The same shape
    works for xAI and for Ollama-style local servers.

    Failed attempts retry with exponential backoff ``backoff_base * 2**attempt``,
    at most 3 attempts. ``timeout`` is passed to ``urlopen``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        name: str = "openai",
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ValueError("max_retries must be an integer")
        if isinstance(backoff_base, bool) or not isinstance(backoff_base, (int, float)):
            raise ValueError("backoff_base must be a number")
        if backoff_base < 0:
            raise ValueError("backoff_base must be non-negative")
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self.timeout = float(timeout)
        self.max_retries = min(3, max(1, max_retries))
        self.backoff_base = float(backoff_base)
        self.name = name

    @property
    def api_key(self) -> str:
        return _pick(self._api_key, _ENV_API_KEY, "")

    @property
    def base_url(self) -> str:
        return _pick(self._base_url, _ENV_BASE_URL, _DEFAULT_BASE_URL).rstrip("/")

    @property
    def model(self) -> str:
        return _pick(self._model, _ENV_MODEL, _DEFAULT_MODEL)

    def complete(self, prompt: str) -> str:
        return self.complete_detailed(prompt).text

    def complete_detailed(self, prompt: str) -> Completion:
        """POST the chat completion and copy its ``usage`` object.

        A missing ``usage`` object becomes zeros. A present object that is
        not a JSON object, or a token field that is not a non-negative
        number, is a :class:`ProviderError` and is retried like any other
        bad payload. ``total_tokens`` defaults to the sum of the two counts
        when the server omits it.
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._request(prompt)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError, ProviderError) as exc:
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                time.sleep(self.backoff_base * (2**attempt))
        if last_error is None:
            raise ProviderError("OpenAI-compatible request failed without an attempt")
        raise ProviderError(
            f"OpenAI-compatible request failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def _request(self, prompt: str) -> Completion:
        base_url = self.base_url
        if not base_url:
            raise ProviderError("base_url is empty")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read()
        parsed = json.loads(raw.decode("utf-8"))
        return Completion(
            text=_content_from_payload(parsed),
            usage=_usage_from_payload(parsed),
        )


class AnthropicProvider:
    """POST the Anthropic Messages API with the stdlib HTTP client.

    The URL is ``https://api.anthropic.com/v1/messages``. Headers are
    ``x-api-key`` (constructor ``api_key``, or ``ANTHROPIC_API_KEY`` when
    that argument is ``None``), ``anthropic-version: 2023-06-01``, and
    ``Content-Type: application/json``. The body is ``model``, ``max_tokens``
    (the API requires it; default 1024), and one user message whose content
    is the prompt. ``model`` defaults to ``claude-sonnet-4-6``.

    A missing or blank key raises :class:`ProviderError` before any request.
    Text blocks in ``content`` are concatenated in order. Other blocks are
    skipped. ``usage.input_tokens`` and ``usage.output_tokens`` become
    ``prompt_tokens`` and ``completion_tokens``. ``total_tokens`` is their
    sum. A missing ``usage`` object becomes zeros.

    Failed attempts retry with exponential backoff ``backoff_base * 2**attempt``,
    at most 3 attempts, same as :class:`OpenAICompatibleProvider`. A non-200
    response is a :class:`ProviderError` that includes the status code and is
    retried like any other failed attempt. ``timeout`` is passed to ``urlopen``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        name: str = "anthropic",
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ValueError("max_retries must be an integer")
        if isinstance(backoff_base, bool) or not isinstance(backoff_base, (int, float)):
            raise ValueError("backoff_base must be a number")
        if backoff_base < 0:
            raise ValueError("backoff_base must be non-negative")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer >= 1")
        if model is not None and not isinstance(model, str):
            raise TypeError("model must be a string")
        if model is not None and model.strip() == "":
            raise ValueError("model must be a non-empty string")
        if api_key is not None and not isinstance(api_key, str):
            raise TypeError("api_key must be a string")
        self._api_key = api_key
        self.model = _DEFAULT_ANTHROPIC_MODEL if model is None else model
        self.max_tokens = max_tokens
        self.timeout = float(timeout)
        self.max_retries = min(3, max(1, max_retries))
        self.backoff_base = float(backoff_base)
        self.name = name

    @property
    def api_key(self) -> str:
        if self._api_key is not None:
            return self._api_key
        return os.environ.get(_ENV_ANTHROPIC_KEY, "")

    def complete(self, prompt: str) -> str:
        return self.complete_detailed(prompt).text

    def complete_detailed(self, prompt: str) -> Completion:
        """POST one user message and copy text blocks plus token usage.

        A blank API key is a :class:`ProviderError` and is not retried.
        Transport errors, non-200 statuses, and bad payloads are retried.
        """
        if self.api_key.strip() == "":
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._request(prompt)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError, ProviderError) as exc:
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                time.sleep(self.backoff_base * (2**attempt))
        if last_error is None:
            raise ProviderError("Anthropic request failed without an attempt")
        raise ProviderError(
            f"Anthropic request failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def _request(self, prompt: str) -> Completion:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        request = urllib.request.Request(_ANTHROPIC_URL, data=data, headers=headers, method="POST")
        try:
            response_cm = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"Anthropic request returned status {exc.code}") from exc
        with response_cm as response:
            status = _http_status(response)
            if status is not None and status != 200:
                raise ProviderError(f"Anthropic request returned status {status}")
            raw = response.read()
        parsed = json.loads(raw.decode("utf-8"))
        return Completion(
            text=_anthropic_text(parsed),
            usage=_anthropic_usage(parsed),
        )


def _normalize_rule(rule: tuple[str, Any]) -> tuple[str, str | tuple[str, ...]]:
    if not isinstance(rule, tuple) or len(rule) != 2:
        raise TypeError("mock rule must be a (needle, response) pair")
    needle, response = rule
    if not isinstance(needle, str):
        raise TypeError("mock rule needle must be a string")
    return needle, _normalize_mock_response(response)


def _normalize_mock_response(response: Any) -> str | tuple[str, ...]:
    """A string, or a non-empty tuple of alternate strings."""
    if isinstance(response, str):
        return response
    if isinstance(response, (list, tuple)):
        if len(response) == 0:
            raise ValueError("mock rule alternates must be a non-empty list of strings")
        texts: list[str] = []
        for item in response:
            if not isinstance(item, str):
                raise TypeError("mock rule alternates must be strings")
            texts.append(item)
        return tuple(texts)
    raise TypeError("mock rule response must be a string or a list of strings")


def _pick(explicit: str | None, env_name: str, default: str) -> str:
    if explicit is not None:
        return explicit
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return env_value
    return default


def _content_from_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ProviderError("Chat completions response is not a JSON object")
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"Chat completions response is missing content: {exc}") from exc
    if not isinstance(content, str):
        raise ProviderError("Chat completions content is not a string")
    return content


def _usage_from_payload(payload: Any) -> TokenUsage:
    if not isinstance(payload, dict):
        raise ProviderError("Chat completions response is not a JSON object")
    raw = payload.get("usage")
    if raw is None:
        return TokenUsage(0, 0, 0)
    if not isinstance(raw, dict):
        raise ProviderError("usage must be a JSON object")
    prompt_tokens = _token_count(raw, "prompt_tokens")
    completion_tokens = _token_count(raw, "completion_tokens")
    if "total_tokens" in raw and raw["total_tokens"] is not None:
        total_tokens = _token_count(raw, "total_tokens")
    else:
        total_tokens = prompt_tokens + completion_tokens
    return TokenUsage(prompt_tokens, completion_tokens, total_tokens)


def _http_status(response: object) -> int | None:
    """Status from ``urlopen``, or ``None`` when the transport did not set one."""
    status: object = getattr(response, "status", None)
    if status is None:
        getcode = getattr(response, "getcode", None)
        if callable(getcode):
            status = getcode()
    if status is None:
        return None
    if isinstance(status, bool) or not isinstance(status, int):
        raise ProviderError("Anthropic response status is not an integer")
    return status


def _anthropic_text(payload: Any) -> str:
    """Concatenate ``content`` blocks whose ``type`` is ``text``."""
    if not isinstance(payload, dict):
        raise ProviderError("Anthropic response is not a JSON object")
    content = payload.get("content")
    if not isinstance(content, list):
        raise ProviderError("Anthropic response content is not a list of blocks")
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            raise ProviderError("Anthropic content block is not an object")
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if not isinstance(text, str):
            raise ProviderError("Anthropic text block is missing text")
        parts.append(text)
    return "".join(parts)


def _anthropic_usage(payload: Any) -> TokenUsage:
    """Map ``input_tokens`` / ``output_tokens`` onto the shared usage shape.

    ``total_tokens`` is the sum. The Messages API does not send a separate
    total. A missing ``usage`` object is zeros. A present value that is not
    an object, or a token field that is not a non-negative integer, is a
    :class:`ProviderError`.
    """
    if not isinstance(payload, dict):
        raise ProviderError("Anthropic response is not a JSON object")
    raw = payload.get("usage")
    if raw is None:
        return TokenUsage(0, 0, 0)
    if not isinstance(raw, dict):
        raise ProviderError("usage must be a JSON object")
    prompt_tokens = _token_count(raw, "input_tokens")
    completion_tokens = _token_count(raw, "output_tokens")
    return TokenUsage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens)


def _token_count(raw: dict[str, Any], key: str) -> int:
    if key not in raw or raw[key] is None:
        return 0
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderError(f"usage.{key} must be a non-negative integer")
    number = float(value)
    if not number.is_integer() or number < 0:
        raise ProviderError(f"usage.{key} must be a non-negative integer")
    return int(number)


__all__ = [
    "AnthropicProvider",
    "Completion",
    "EchoProvider",
    "MockProvider",
    "OpenAICompatibleProvider",
    "Provider",
    "ProviderError",
    "TokenUsage",
    "provider_model_name",
    "usage_from_text_lengths",
]
