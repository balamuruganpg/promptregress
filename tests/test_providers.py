"""Mock, echo, OpenAI-compatible, and Anthropic clients. HTTP is faked."""

import json
import urllib.error
from email.message import EmailMessage
from pathlib import Path

import pytest

from promptregress.cli import main
from promptregress.providers import (
    AnthropicProvider,
    EchoProvider,
    MockProvider,
    OpenAICompatibleProvider,
    Provider,
    ProviderError,
    TokenUsage,
)


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def test_mock_first_rule_wins_and_counts_calls() -> None:
    provider = MockProvider(rules=[("a", "first"), ("ab", "second")], default="fallback")
    assert isinstance(provider, Provider)
    assert provider.complete("ab") == "first"
    assert provider.complete("zzz") == "fallback"
    assert provider.complete("ba") == "first"
    assert provider.call_count == 3
    assert provider.name == "mock"


def test_echo_returns_the_prompt() -> None:
    provider = EchoProvider()
    assert provider.complete("Hello {input}") == "Hello {input}"
    assert provider.name == "echo"


def test_openai_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float | None = None) -> _Response:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode())  # type: ignore[attr-defined]
        captured["headers"] = {
            key.lower(): value
            for key, value in request.header_items()  # type: ignore[attr-defined]
        }
        return _Response(b'{"choices":[{"message":{"content":"pong"}}]}')

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        api_key="sek",
        base_url="https://example.test/v1/",
        model="demo",
        timeout=7,
    )
    assert provider.complete("hello") == "pong"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["timeout"] == 7
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "demo"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["authorization"] == "Bearer sek"
    assert headers["content-type"] == "application/json"


def test_openai_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(request: object, timeout: float | None = None) -> _Response:
        del request, timeout
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("temporary")
        return _Response(b'{"choices":[{"message":{"content":"ok"}}]}')

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "promptregress.providers.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )
    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://example.test/v1",
        model="m",
        backoff_base=0.5,
    )
    assert provider.complete("hi") == "ok"
    assert calls["n"] == 3
    assert sleeps == [0.5, 1.0]


def test_openai_retries_are_capped_at_three(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_urlopen(request: object, timeout: float | None = None) -> _Response:
        del request, timeout
        calls["n"] += 1
        raise urllib.error.URLError("down")

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("promptregress.providers.time.sleep", lambda seconds: None)
    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://example.test/v1",
        model="m",
        max_retries=10,
        backoff_base=0.0,
    )
    with pytest.raises(ProviderError, match="3 attempts"):
        provider.complete("hi")
    assert calls["n"] == 3


def test_openai_reads_env_and_constructor_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMPTREGRESS_API_KEY", "env-key")
    monkeypatch.setenv("PROMPTREGRESS_BASE_URL", "https://llm.test/v1/")
    monkeypatch.setenv("PROMPTREGRESS_MODEL", "grok-test")
    provider = OpenAICompatibleProvider()
    assert provider.api_key == "env-key"
    assert provider.base_url == "https://llm.test/v1"
    assert provider.model == "grok-test"
    override = OpenAICompatibleProvider(
        api_key="a",
        base_url="https://override.test/v1",
        model="m",
    )
    assert override.api_key == "a"
    assert override.base_url == "https://override.test/v1"
    assert override.model == "m"


def test_openai_bad_payload_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_urlopen(request: object, timeout: float | None = None) -> _Response:
        del request, timeout
        calls["n"] += 1
        return _Response(b'{"nope": true}')

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("promptregress.providers.time.sleep", lambda seconds: None)
    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://example.test/v1",
        model="m",
        backoff_base=0,
    )
    with pytest.raises(ProviderError, match="3 attempts"):
        provider.complete("hi")
    assert calls["n"] == 3


class _StatusResponse:
    def __init__(self, payload: bytes, status: int | str | None = 200) -> None:
        self.payload = payload
        self.status = status

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "_StatusResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _anthropic_payload(text: str = "Hello world") -> bytes:
    body: dict[str, object] = {
        "content": [
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
            {"type": "text", "text": " world"},
        ],
        "usage": {"input_tokens": 11, "output_tokens": 4},
    }
    if text != "Hello world":
        body["content"] = [{"type": "text", "text": text}]
    return json.dumps(body).encode("utf-8")


def test_anthropic_request_shape_text_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float | None = None) -> _StatusResponse:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["method"] = request.get_method()  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode())  # type: ignore[attr-defined]
        captured["headers"] = {
            key.lower(): value
            for key, value in request.header_items()  # type: ignore[attr-defined]
        }
        return _StatusResponse(_anthropic_payload())

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", fake_urlopen)
    from_env = AnthropicProvider(max_tokens=256, timeout=9)
    detailed = from_env.complete_detailed("hello")
    assert detailed.text == "Hello world"
    assert detailed.usage == TokenUsage(11, 4, 15)
    assert from_env.complete("hello") == "Hello world"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["method"] == "POST"
    assert captured["timeout"] == 9
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "claude-sonnet-4-6"
    assert body["max_tokens"] == 256
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["x-api-key"] == "env-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"

    def override_urlopen(request: object, timeout: float | None = None) -> _StatusResponse:
        del timeout
        captured["override_key"] = {
            key.lower(): value
            for key, value in request.header_items()  # type: ignore[attr-defined]
        }["x-api-key"]
        captured["override_body"] = json.loads(request.data.decode())  # type: ignore[attr-defined]
        return _StatusResponse(_anthropic_payload("pong"))

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", override_urlopen)
    override = AnthropicProvider(api_key="sek", model="claude-test", max_tokens=32)
    assert override.complete("hello") == "pong"
    assert captured["override_key"] == "sek"
    override_body = captured["override_body"]
    assert isinstance(override_body, dict)
    assert override_body["model"] == "claude-test"
    assert override_body["max_tokens"] == 32


def test_anthropic_missing_api_key_does_not_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def fake_urlopen(request: object, timeout: float | None = None) -> _StatusResponse:
        del request, timeout
        raise AssertionError("urlopen should not be called")

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY is not set"):
        AnthropicProvider().complete("hi")
    monkeypatch.setenv("ANTHROPIC_API_KEY", " \t")
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY is not set"):
        AnthropicProvider().complete_detailed("hi")
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY is not set"):
        AnthropicProvider(api_key="  ").complete("hi")


def test_anthropic_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_http(request: object, timeout: float | None = None) -> _StatusResponse:
        del request, timeout
        raise urllib.error.HTTPError(
            "https://api.anthropic.com/v1/messages",
            401,
            "Unauthorized",
            EmailMessage(),
            None,
        )

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", raise_http)
    monkeypatch.setattr("promptregress.providers.time.sleep", lambda seconds: None)
    provider = AnthropicProvider(api_key="sek", model="m", max_retries=1, backoff_base=0)
    with pytest.raises(ProviderError, match="status 401"):
        provider.complete("hi")

    def status_code(request: object, timeout: float | None = None) -> _StatusResponse:
        del request, timeout
        return _StatusResponse(b"{}", status=418)

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", status_code)
    with pytest.raises(ProviderError, match="status 418"):
        provider.complete_detailed("hi")

    def bad_status_type(request: object, timeout: float | None = None) -> _StatusResponse:
        del request, timeout
        return _StatusResponse(b"{}", status="200")

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", bad_status_type)
    with pytest.raises(ProviderError, match="not an integer"):
        provider.complete("hi")

    class _Coded:
        def __init__(self, code: int) -> None:
            self._code = code

        def read(self) -> bytes:
            return _anthropic_payload("via-code")

        def getcode(self) -> int:
            return self._code

        def __enter__(self) -> "_Coded":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    def coded(request: object, timeout: float | None = None) -> _Coded:
        del request, timeout
        return _Coded(200)

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", coded)
    assert provider.complete("hi") == "via-code"

    def coded_error(request: object, timeout: float | None = None) -> _Coded:
        del request, timeout
        return _Coded(503)

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", coded_error)
    with pytest.raises(ProviderError, match="status 503"):
        provider.complete("hi")

    class _Bare:
        def read(self) -> bytes:
            return _anthropic_payload("bare")

        def __enter__(self) -> "_Bare":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    def bare(request: object, timeout: float | None = None) -> _Bare:
        del request, timeout
        return _Bare()

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", bare)
    assert provider.complete("hi") == "bare"


def test_anthropic_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(request: object, timeout: float | None = None) -> _StatusResponse:
        del request, timeout
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("temporary")
        return _StatusResponse(_anthropic_payload("ok"))

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "promptregress.providers.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )
    provider = AnthropicProvider(api_key="k", model="m", backoff_base=0.5)
    assert provider.complete("hi") == "ok"
    assert calls["n"] == 3
    assert sleeps == [0.5, 1.0]


def test_anthropic_bad_payload_and_missing_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = {"body": b'{"content":[{"type":"text","text":"pong"}]}'}

    def fake_urlopen(request: object, timeout: float | None = None) -> _StatusResponse:
        del request, timeout
        return _StatusResponse(payloads["body"])

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", fake_urlopen)
    provider = AnthropicProvider(api_key="k", model="m", max_retries=1, backoff_base=0)
    assert provider.complete_detailed("hi").usage == TokenUsage(0, 0, 0)
    payloads["body"] = b'{"content":[{"type":"text","text":"pong"}],"usage":[]}'
    with pytest.raises(ProviderError, match="usage"):
        provider.complete("hi")
    payloads["body"] = b'{"content":[{"type":"text","text":"pong"}],"usage":{"input_tokens":-1}}'
    with pytest.raises(ProviderError, match="input_tokens"):
        provider.complete_detailed("hi")
    payloads["body"] = b'{"content":"pong"}'
    with pytest.raises(ProviderError, match="content"):
        provider.complete("hi")
    payloads["body"] = b'{"content":["nope"]}'
    with pytest.raises(ProviderError, match="block"):
        provider.complete("hi")
    payloads["body"] = b'{"content":[{"type":"text","text":1}]}'
    with pytest.raises(ProviderError, match="text"):
        provider.complete("hi")
    payloads["body"] = b"[]"
    with pytest.raises(ProviderError, match="JSON object"):
        provider.complete("hi")
    payloads["body"] = b"\xff"
    with pytest.raises(ProviderError, match="1 attempts"):
        provider.complete("hi")


def test_anthropic_rejects_bad_constructor_values() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        AnthropicProvider(api_key="k", max_tokens=0)
    with pytest.raises(ValueError, match="max_tokens"):
        AnthropicProvider(api_key="k", max_tokens=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model"):
        AnthropicProvider(api_key="k", model="  ")
    with pytest.raises(TypeError, match="model"):
        AnthropicProvider(api_key="k", model=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="api_key"):
        AnthropicProvider(api_key=1)  # type: ignore[arg-type]
    bare = AnthropicProvider(api_key="k")
    assert bare.model == "claude-sonnet-4-6"
    assert bare.max_tokens == 1024
    assert bare.name == "anthropic"


def test_cli_anthropic_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [
                    {
                        "id": "a",
                        "input": "alpha",
                        "reference": "pong",
                        "assertions": [{"kind": "exact_match"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float | None = None) -> _StatusResponse:
        captured["body"] = json.loads(request.data.decode())  # type: ignore[attr-defined]
        captured["key"] = {
            key.lower(): value
            for key, value in request.header_items()  # type: ignore[attr-defined]
        }["x-api-key"]
        captured["timeout"] = timeout
        return _StatusResponse(_anthropic_payload("pong"))

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", fake_urlopen)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "run.json"
    code = main(
        [
            "run",
            str(suite),
            "--provider",
            "anthropic",
            "--api-key",
            "sek",
            "--model",
            "claude-test",
            "--timeout",
            "11",
            "--config-name",
            "anth",
            "--out",
            str(out),
            "--no-history",
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["pass_rate"] == 1.0
    assert payload["results"][0]["output"] == "pong"
    assert payload["results"][0]["prompt_tokens"] == 11
    assert payload["results"][0]["completion_tokens"] == 4
    assert payload["total_tokens"] == 15
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "claude-test"
    assert body["max_tokens"] == 1024
    assert captured["key"] == "sek"
    assert captured["timeout"] == 11.0
