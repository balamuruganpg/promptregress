"""Latency aggregates, provider usage, and cache-hit semantics."""

import json
from pathlib import Path

import pytest

from helpers import run_with
from promptregress.cache import ResponseCache
from promptregress.gate import evaluate_gate
from promptregress.models import AssertionSpec, Case, CaseResult, RunResult, Suite
from promptregress.providers import (
    EchoProvider,
    MockProvider,
    OpenAICompatibleProvider,
    ProviderError,
    TokenUsage,
    usage_from_text_lengths,
)
from promptregress.report import render_html, render_markdown, write_run_json
from promptregress.runner import run_suite
from promptregress.stats import compare_runs


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _suite(text: str = "hello") -> Suite:
    return Suite(
        name="s",
        cases=[
            Case(
                id="a",
                input=text,
                reference="ok",
                assertions=[AssertionSpec("exact_match", value="ok")],
            )
        ],
    )


def test_p95_uses_the_linear_rank_rule() -> None:
    rows = [
        CaseResult(
            case_id=str(index),
            output="x",
            scores={},
            score=1.0,
            passed=True,
            latency_ms=float(latency),
            prompt_tokens=index,
            completion_tokens=1,
            total_tokens=index + 1,
        )
        for index, latency in enumerate([10.0, 20.0, 30.0, 40.0, 50.0], start=1)
    ]
    result = RunResult(suite_name="s", config_name="c", results=rows)
    assert result.mean_latency_ms == pytest.approx(30.0)
    # rank 0.95 * (5 - 1) = 3.8 → 40 * 0.2 + 50 * 0.8
    assert result.p95_latency_ms == pytest.approx(48.0)
    assert result.total_tokens == sum(row.total_tokens for row in rows)
    assert result.total_prompt_tokens == sum(row.prompt_tokens for row in rows)
    assert result.total_latency_ms == pytest.approx(150.0)
    assert result.n_cached == 0
    empty = RunResult(suite_name="s", config_name="c", results=[])
    assert empty.p95_latency_ms == 0.0
    assert empty.mean_latency_ms == 0.0
    assert empty.total_tokens == 0


def test_mock_usage_is_character_length_and_cache_hit_is_zero_latency(tmp_path: Path) -> None:
    root = tmp_path
    provider = MockProvider(rules=[("hello", "ok")], default="no")
    cache = ResponseCache(root)
    prompt = "hello"
    first = run_suite(_suite(), provider, "c", cache=cache)
    row = first.results[0]
    expected = usage_from_text_lengths(prompt, "ok")
    assert row.output == "ok"
    assert row.cached is False
    assert row.latency_ms >= 0.0
    assert row.prompt_tokens == expected.prompt_tokens == len(prompt)
    assert row.completion_tokens == expected.completion_tokens == len("ok")
    assert row.total_tokens == expected.total_tokens
    assert first.total_tokens == expected.total_tokens
    assert provider.call_count == 1
    second = run_suite(_suite(), provider, "c", cache=cache)
    hit = second.results[0]
    assert provider.call_count == 1
    assert cache.hits == 1
    assert hit.cached is True
    assert hit.latency_ms == 0.0
    assert hit.total_tokens == expected.total_tokens
    assert hit.prompt_tokens == expected.prompt_tokens
    assert second.n_cached == 1
    assert second.mean_latency_ms == 0.0
    assert second.p95_latency_ms == 0.0


def test_echo_without_detailed_usage_records_zeros() -> None:
    result = run_suite(_suite("ok"), EchoProvider(), "echo", prompt_template="{input}")
    row = result.results[0]
    assert row.passed is True
    assert row.cached is False
    assert row.prompt_tokens == 0
    assert row.completion_tokens == 0
    assert row.total_tokens == 0
    assert result.total_tokens == 0


def test_provider_error_keeps_zero_usage(tmp_path: Path) -> None:
    root = tmp_path

    class Boom:
        name = "boom"
        model = "boom"

        def complete(self, prompt: str) -> str:
            raise RuntimeError("nope")

    cache = ResponseCache(root)
    result = run_suite(_suite(), Boom(), "c", cache=cache)
    row = result.results[0]
    assert row.cached is False
    assert row.total_tokens == 0
    assert row.latency_ms >= 0.0
    assert row.error is not None
    assert list(root.glob("*.json")) == []


def test_openai_usage_reaches_the_case_and_the_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path

    def fake_urlopen(request: object, timeout: float | None = None) -> _Response:
        del request, timeout
        body = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 9},
        }
        return _Response(json.dumps(body).encode())

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(api_key="k", base_url="https://example.test/v1", model="m")
    detailed = provider.complete_detailed("hello")
    assert detailed.text == "ok"
    assert detailed.usage == TokenUsage(5, 2, 9)
    assert provider.complete("hello") == "ok"
    result = run_suite(_suite(), provider, "openai")
    row = result.results[0]
    assert row.prompt_tokens == 5
    assert row.completion_tokens == 2
    assert row.total_tokens == 9
    assert row.cached is False
    out = root / "run.json"
    write_run_json(result, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["total_tokens"] == 9
    assert payload["total_prompt_tokens"] == 5
    assert payload["mean_latency_ms"] == result.mean_latency_ms
    assert payload["p95_latency_ms"] == result.p95_latency_ms
    assert payload["results"][0]["cached"] is False
    assert payload["total_latency_ms"] == result.total_latency_ms


def test_openai_missing_usage_is_zero_and_bad_usage_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = {"body": b'{"choices":[{"message":{"content":"pong"}}]}'}

    def fake_urlopen(request: object, timeout: float | None = None) -> _Response:
        del request, timeout
        return _Response(payloads["body"])

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://example.test/v1",
        model="m",
        max_retries=1,
        backoff_base=0,
    )
    assert provider.complete_detailed("hi").usage == TokenUsage(0, 0, 0)
    payloads["body"] = b'{"choices":[{"message":{"content":"pong"}}],"usage":{"prompt_tokens":-1}}'
    with pytest.raises(ProviderError, match="prompt_tokens"):
        provider.complete("hi")
    payloads["body"] = b'{"choices":[{"message":{"content":"pong"}}],"usage":{"prompt_tokens":1.5}}'
    with pytest.raises(ProviderError, match="prompt_tokens"):
        provider.complete_detailed("hi")


def test_reports_show_latency_and_tokens() -> None:
    baseline = run_with([("a", 1.0, True)], "baseline")
    candidate = run_with([("a", 1.0, True)], "candidate")
    comparison = compare_runs(baseline, candidate, n_boot=20, seed=1)
    verdict = evaluate_gate(comparison)
    markdown = render_markdown(comparison, verdict)
    html = render_html(comparison, verdict)
    assert "Latency and usage" in markdown
    assert "Mean latency (ms)" in markdown
    assert "P95 latency (ms)" in markdown
    assert "Total tokens" in markdown
    assert "Total latency (ms)" in markdown
    assert "Latency and usage" in html
    assert "P95 latency (ms)" in html
    assert "Total tokens" in html
