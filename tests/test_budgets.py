"""Cost and p95 latency budgets on the gate, for run and compare."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from helpers import comparison_with
from promptregress.cli import main
from promptregress.gate import GateRule, evaluate_gate, evaluate_run_budgets
from promptregress.models import (
    AssertionSpec,
    Case,
    CaseResult,
    CostEstimate,
    Suite,
)
from promptregress.pricing import ModelPrice, estimate_cost
from promptregress.providers import MockProvider
from promptregress.runner import run_suite


def _priced(usd: float | None, *, note: str | None = None, latency_ms: float = 1.0):
    comparison = comparison_with(
        ci_low=0.0,
        ci_high=0.0,
        mean_diff=0.0,
        candidate_rows=[("a", 1.0, True)],
    )
    results = [replace(row, latency_ms=latency_ms) for row in comparison.candidate.results]
    cost = None if usd is None and note is None else CostEstimate(model="mock", usd=usd, note=note)
    candidate = replace(comparison.candidate, results=results, cost=cost)
    return replace(comparison, candidate=candidate)


def test_cost_budget_passes_at_equality_and_fails_above() -> None:
    comparison = _priced(1.0)
    failed = evaluate_gate(comparison, max_regression=0.0, max_cost_usd=0.5)
    assert failed.passed is False
    assert any("cost $1.000000 exceeds budget $0.500000" in reason for reason in failed.reasons)
    held = evaluate_gate(comparison, max_regression=0.0, max_cost_usd=1.0)
    assert held.passed is True
    assert any(
        "PASS cost: cost $1.000000 is within budget $1.000000." in reason for reason in held.reasons
    )


def test_latency_budget_passes_at_equality_and_fails_above() -> None:
    comparison = _priced(None, latency_ms=40.0)
    failed = evaluate_gate(comparison, max_regression=0.0, max_p95_latency_ms=39.0)
    assert failed.passed is False
    assert any(
        "p95 latency 40.000000ms exceeds budget 39.000000ms" in reason for reason in failed.reasons
    )
    held = evaluate_gate(comparison, max_regression=0.0, max_p95_latency_ms=40.0)
    assert held.passed is True
    assert any(
        "PASS latency: p95 latency 40.000000ms is within budget 40.000000ms." in reason
        for reason in held.reasons
    )


def test_both_budgets_together() -> None:
    comparison = _priced(2.0, latency_ms=10.0)
    both_fail = evaluate_gate(
        comparison, max_regression=0.0, max_cost_usd=1.0, max_p95_latency_ms=5.0
    )
    assert both_fail.passed is False
    assert any("cost $2.000000 exceeds budget $1.000000" in reason for reason in both_fail.reasons)
    assert any(
        "p95 latency 10.000000ms exceeds budget 5.000000ms" in reason
        for reason in both_fail.reasons
    )
    cost_only = evaluate_gate(
        comparison, max_regression=0.0, max_cost_usd=1.0, max_p95_latency_ms=10.0
    )
    assert cost_only.passed is False
    assert any(reason.startswith("FAIL cost:") for reason in cost_only.reasons)
    assert any(reason.startswith("PASS latency:") for reason in cost_only.reasons)
    both_pass = evaluate_gate(
        comparison, max_regression=0.0, max_cost_usd=2.0, max_p95_latency_ms=10.0
    )
    assert both_pass.passed is True
    assert all(not reason.startswith("FAIL ") for reason in both_pass.reasons)


def test_unset_budgets_do_not_change_the_gate() -> None:
    comparison = _priced(9.0, latency_ms=100.0)
    plain = evaluate_gate(comparison, max_regression=0.0)
    explicit = evaluate_gate(
        comparison, max_regression=0.0, max_cost_usd=None, max_p95_latency_ms=None
    )
    assert plain == explicit
    assert plain.passed is True
    assert all("cost" not in reason and "p95 latency" not in reason for reason in plain.reasons)
    run_only = evaluate_run_budgets(comparison.candidate)
    assert run_only.passed is True
    assert run_only.reasons == []


def test_missing_or_unknown_cost_fails_a_set_budget() -> None:
    missing = evaluate_gate(_priced(None), max_regression=0.0, max_cost_usd=1.0)
    assert missing.passed is False
    assert any(
        "cost was not estimated" in reason and "$1.000000" in reason for reason in missing.reasons
    )
    unknown = evaluate_gate(
        _priced(None, note="unknown model 'nope'"),
        max_regression=0.0,
        max_cost_usd=1.0,
    )
    assert unknown.passed is False
    assert any("cost is unknown (unknown model 'nope')" in reason for reason in unknown.reasons)


def test_gate_rule_supplies_budgets_and_rejects_non_finite_limits() -> None:
    comparison = _priced(5.0, latency_ms=5.0)
    verdict = evaluate_gate(
        comparison,
        max_cost_usd=0.0,
        max_p95_latency_ms=0.0,
        rule=GateRule(max_regression=0.0, max_cost_usd=5.0, max_p95_latency_ms=5.0),
    )
    assert verdict.passed is True
    assert any("within budget $5.000000" in reason for reason in verdict.reasons)
    with pytest.raises(ValueError, match="max_cost_usd"):
        evaluate_gate(comparison, max_cost_usd=float("nan"))
    with pytest.raises(ValueError, match="max_p95_latency_ms"):
        evaluate_gate(comparison, max_p95_latency_ms=float("inf"))
    with pytest.raises(ValueError, match="max_cost_usd"):
        evaluate_gate(comparison, max_cost_usd=True)  # type: ignore[arg-type]


def test_mock_latency_is_fixed_and_cost_matches_estimate_cost() -> None:
    suite = Suite(
        name="s",
        cases=[
            Case(
                id="a",
                input="hi",
                reference=None,
                assertions=[AssertionSpec(kind="contains", value="yes")],
            )
        ],
    )
    default = MockProvider(default="yes")
    result = run_suite(suite, default, "mock")
    assert result.results[0].latency_ms == 1.0
    assert result.p95_latency_ms == 1.0
    custom = run_suite(suite, MockProvider(default="yes", latency_ms=12.5), "mock")
    assert custom.results[0].latency_ms == 12.5
    prices = {"mock": ModelPrice(prompt_per_1k=500.0, completion_per_1k=0.0)}
    estimate = estimate_cost(result, "mock", prices)
    assert estimate.usd == pytest.approx(result.total_prompt_tokens / 1000 * 500.0)
    priced = replace(result, cost=estimate)
    verdict = evaluate_run_budgets(priced, max_cost_usd=0.5, max_p95_latency_ms=1.0)
    assert verdict.passed is False
    assert any(
        f"cost ${estimate.usd:.6f} exceeds budget $0.500000" in reason for reason in verdict.reasons
    )
    assert any(reason.startswith("PASS latency:") for reason in verdict.reasons)
    with pytest.raises(ValueError, match="latency_ms"):
        MockProvider(latency_ms=float("nan"))
    with pytest.raises(TypeError, match="latency_ms"):
        MockProvider(latency_ms=True)  # type: ignore[arg-type]


def _suite(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [
                    {
                        "id": "a",
                        "input": "hi",
                        "assertions": [{"kind": "contains", "value": "yes"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _prices(path: Path) -> Path:
    path.write_text(
        json.dumps({"mock": {"prompt_per_1k": 500, "completion_per_1k": 0}}),
        encoding="utf-8",
    )
    return path


def test_run_and_compare_text_show_budget_reasons(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    suite = _suite(tmp_path / "suite.json")
    prices = _prices(tmp_path / "prices.json")
    out = tmp_path / "run.json"
    failed = main(
        [
            "run",
            str(suite),
            "--provider",
            "mock",
            "--mock-default",
            "yes",
            "--config-name",
            "c",
            "--out",
            str(out),
            "--pricing",
            str(prices),
            "--max-cost-usd",
            "0.5",
            "--max-p95-latency-ms",
            "0.5",
            "--no-history",
        ]
    )
    assert failed == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    usd = payload["cost"]["usd"]
    assert usd == pytest.approx(payload["total_prompt_tokens"] / 1000 * 500)
    assert payload["p95_latency_ms"] == 1.0
    text = capsys.readouterr().out
    assert f"cost ${usd:.6f} exceeds budget $0.500000" in text
    assert "p95 latency 1.000000ms exceeds budget 0.500000ms" in text

    passed = main(
        [
            "compare",
            str(suite),
            "--provider",
            "mock",
            "--mock-default",
            "yes",
            "--baseline-template",
            "{input}",
            "--candidate-template",
            "{input}",
            "--pricing",
            str(prices),
            "--max-cost-usd",
            "100",
            "--max-p95-latency-ms",
            "1",
            "--max-regression",
            "0",
            "--no-history",
        ]
    )
    assert passed == 0
    compared = capsys.readouterr().out
    assert "PASS cost:" in compared
    assert "is within budget $100.000000" in compared
    assert "PASS latency: p95 latency 1.000000ms is within budget 1.000000ms." in compared
    assert "gate_passed=true" in compared

    unset = main(
        [
            "compare",
            str(suite),
            "--provider",
            "mock",
            "--mock-default",
            "yes",
            "--baseline-template",
            "{input}",
            "--candidate-template",
            "{input}",
            "--max-regression",
            "0",
            "--no-history",
        ]
    )
    assert unset == 0
    quiet = capsys.readouterr().out
    assert "exceeds budget" not in quiet
    assert "within budget" not in quiet
    assert "gate_passed=true" in quiet


def test_run_without_budgets_still_exits_zero_when_a_case_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    suite = tmp_path / "bad.json"
    suite.write_text(
        json.dumps(
            {
                "name": "bad",
                "cases": [
                    {
                        "id": "a",
                        "input": "hi",
                        "assertions": [{"kind": "contains", "value": "missing"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "run",
            str(suite),
            "--provider",
            "mock",
            "--mock-default",
            "no",
            "--config-name",
            "c",
            "--out",
            str(tmp_path / "out.json"),
            "--no-history",
        ]
    )
    assert code == 0


def test_constructed_case_result_keeps_explicit_latency() -> None:
    """Hand-built rows still own their latency. The mock default does not apply."""
    row = CaseResult(case_id="a", output="x", scores={}, score=1.0, passed=True, latency_ms=7.0)
    assert row.latency_ms == 7.0
    assert row.source_line is None
