"""Regression tests for bugs found in the independent v0.6.15 review cycle #17.

Scope is report / JUnit / annotations / providers / runner / gate: OpenAI
array content, env whitespace on provider config, JUnit time and XML 1.0
C1 controls, and JSON reports that emitted non-RFC NaN/Infinity literals.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import comparison_with
from promptregress.gate import GateVerdict
from promptregress.junit_xml import render_junit_xml
from promptregress.models import CaseResult, RunResult
from promptregress.providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    ProviderError,
    _content_from_payload,
)
from promptregress.report import write_json, write_run_json


def test_openai_content_array_concatenates_text_parts() -> None:
    """Chat Completions ``content`` may be a list of parts, not only a string.

    Anthropic already concatenates text blocks and skips other block types.
    OpenAI-compatible responses (and some proxies) use the same array shape
    for assistant ``message.content``. Treating a list as invalid raised
    ``ProviderError`` after retries, so a successful HTTP body scored as a
    provider failure. Concatenate ``type=text`` parts in order, like Anthropic.
    """
    text = _content_from_payload(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "image_url", "image_url": {"url": "x"}},
                            {"type": "text", "text": " world"},
                        ]
                    }
                }
            ]
        }
    )
    assert text == "Hello world"
    with pytest.raises(ProviderError, match="not a string"):
        _content_from_payload({"choices": [{"message": {"content": {"text": "x"}}}]})
    with pytest.raises(ProviderError, match="text"):
        _content_from_payload(
            {"choices": [{"message": {"content": [{"type": "text", "text": 1}]}}]}
        )


def test_provider_config_strips_env_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trailing newlines on provider env values must not reach the wire.

    ``.env`` loaders and shell exports often leave a trailing newline on
    ``PROMPTREGRESS_BASE_URL`` / ``API_KEY`` / ``MODEL`` and
    ``ANTHROPIC_API_KEY``. The OpenAI client built ``…/v1\\n/chat/completions``
    and sent ``Bearer`` / model ids with the newline; Anthropic checked
    blankness with ``strip()`` but still forwarded the raw key. Strip
    resolved config strings so whitespace-only base URLs are empty (and
    error) and keys/models match what the server expects.
    """
    monkeypatch.setenv("PROMPTREGRESS_BASE_URL", "https://example.test/v1/\n")
    monkeypatch.setenv("PROMPTREGRESS_API_KEY", "sek\n")
    monkeypatch.setenv("PROMPTREGRESS_MODEL", "demo\n")
    openai = OpenAICompatibleProvider()
    assert openai.base_url == "https://example.test/v1"
    assert openai.api_key == "sek"
    assert openai.model == "demo"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-key\n")
    anthropic = AnthropicProvider()
    assert anthropic.api_key == "anth-key"

    blank = OpenAICompatibleProvider(api_key="k", base_url="  \n", model="m")
    assert blank.base_url == ""


def test_junit_clamps_negative_latency_time_to_zero() -> None:
    """JUnit ``time`` must not stay negative when case latency is negative.

    Non-finite latency already mapped to ``0.000000`` seconds so a poisoned
    p95 could not emit ``time="nan"``. Negative latency (accepted by
    ``load_run_report``, rejected by the gate budget) still produced
    ``time="-0.005000"``, which many CI consumers treat as corrupt. Clamp
    negatives the same way as non-finite values.
    """
    result = RunResult(
        suite_name="s",
        config_name="c",
        results=[
            CaseResult(
                case_id="a",
                output="x",
                scores={"contains": 0.0},
                score=0.0,
                passed=False,
                latency_ms=-5.0,
            )
        ],
    )
    xml = render_junit_xml([result])
    assert 'time="-0.005000"' not in xml
    assert 'time="0.000000"' in xml


def test_junit_strips_xml10_c1_controls() -> None:
    """XML 1.0 C1 controls (except NEL) must be stripped like C0 controls.

    The exporter already removed C0 controls and surrogates so completions
    could not embed illegal Char values. DEL and C1 controls
    ``U+007F``–``U+0084`` / ``U+0086``–``U+009F`` are also outside the XML
    1.0 Char production and were left in case ids and ``system-out``.
    ``U+0085`` (NEL) stays; it is a legal Char.
    """
    result = RunResult(
        suite_name="s",
        config_name="c",
        results=[
            CaseResult(
                case_id="a\x7fb",
                output="x\u0080y\u0085z\u009f",
                scores={"contains": 0.0},
                score=0.0,
                passed=False,
                latency_ms=1.0,
            )
        ],
    )
    root = __import__("xml.etree.ElementTree", fromlist=["ET"]).fromstring(
        render_junit_xml([result])
    )
    case = root.find("testsuite/testcase")
    assert case is not None
    assert case.get("name") == "ab"
    system_out = case.find("system-out")
    assert system_out is not None
    assert system_out.text == "xy\u0085z"


def test_json_reports_reject_nonfinite_numbers(tmp_path: Path) -> None:
    """Compare/run JSON must not emit RFC-8259-illegal ``NaN`` / ``Infinity``.

    ``_json_number`` already nulls non-finite Wilcoxon/t/Cohen fields, but
    ``json.dumps`` still allowed ``NaN`` on ``mean_diff`` / scores /
    latency. Python's loader accepts those literals; strict consumers and
    this project's own ``load_run_report`` do not. Refuse non-finite numbers
    at write time with ``allow_nan=False``.
    """
    comparison = comparison_with(
        ci_low=float("nan"),
        ci_high=0.0,
        mean_diff=0.0,
        candidate_rows=[("a", 1.0, True)],
    )
    with pytest.raises(ValueError, match="not JSON compliant|Out of range"):
        write_json(comparison, tmp_path / "cmp.json", GateVerdict(False, ["FAIL"]))

    run = RunResult(
        suite_name="s",
        config_name="c",
        results=[
            CaseResult(
                case_id="a",
                output="x",
                scores={"contains": 1.0},
                score=1.0,
                passed=True,
                latency_ms=float("inf"),
            )
        ],
    )
    with pytest.raises(ValueError, match="not JSON compliant|Out of range"):
        write_run_json(run, tmp_path / "run.json")
