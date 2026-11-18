"""Pricing files, hand-computed cost, unknown models, and compare deltas."""

import json
from pathlib import Path

import pytest

from promptregress.cli import main
from promptregress.gate import evaluate_gate
from promptregress.models import AssertionSpec, Case, CaseResult, RunResult, Suite
from promptregress.pricing import PricingError, estimate_cost, load_pricing
from promptregress.providers import Completion, EchoProvider, TokenUsage
from promptregress.report import render_html, render_markdown, write_json
from promptregress.runner import run_suite
from promptregress.stats import compare_runs


class _Priced:
    name = "priced"
    model = "priced-model"

    def complete_detailed(self, prompt: str) -> Completion:
        del prompt
        return Completion(text="ok", usage=TokenUsage(1000, 2000, 3000))


class _Unknown:
    name = "priced"
    model = "missing-model"

    def complete_detailed(self, prompt: str) -> Completion:
        del prompt
        return Completion(text="ok", usage=TokenUsage(10, 10, 20))


def _suite() -> Suite:
    return Suite(
        name="s",
        cases=[
            Case(
                id="a",
                input="hi",
                reference="ok",
                assertions=[AssertionSpec("exact_match", value="ok")],
            )
        ],
    )


def _prices(path: Path, body: object, suffix: str = ".json") -> Path:
    target = path if path.suffix else path.with_suffix(suffix)
    if suffix == ".json":
        target.write_text(json.dumps(body) + "\n", encoding="utf-8")
    else:
        target.write_text(str(body), encoding="utf-8")
    return target


def test_hand_computed_cost_from_known_token_counts(tmp_path: Path) -> None:
    table = load_pricing(
        _prices(
            tmp_path / "prices.json",
            {"priced-model": {"prompt_per_1k": 0.5, "completion_per_1k": 0.25}},
        )
    )
    result = run_suite(_suite(), _Priced(), "priced")
    estimate = estimate_cost(result, "priced-model", table)
    # 1000/1000 * 0.5 + 2000/1000 * 0.25 = 0.5 + 0.5 = 1
    assert result.total_prompt_tokens == 1000
    assert result.total_completion_tokens == 2000
    assert estimate.usd == 1.0
    assert estimate.note is None
    assert estimate.model == "priced-model"


def test_unknown_model_is_null_with_a_note(tmp_path: Path) -> None:
    table = load_pricing(
        _prices(tmp_path / "prices.json", {"other": {"prompt_per_1k": 1, "completion_per_1k": 1}})
    )
    result = run_suite(_suite(), _Unknown(), "missing")
    estimate = estimate_cost(result, "missing-model", table)
    assert estimate.usd is None
    assert estimate.note == "unknown model 'missing-model'"


def test_echo_zero_usage_costs_zero(tmp_path: Path) -> None:
    table = load_pricing(
        _prices(tmp_path / "prices.json", {"other": {"prompt_per_1k": 3, "completion_per_1k": 4}})
    )
    result = run_suite(_suite(), EchoProvider(), "echo", prompt_template="{input}")
    assert result.total_tokens == 0
    estimate = estimate_cost(result, "echo", table)
    assert estimate.usd == 0.0
    assert estimate.note is None


def test_yaml_and_json_pricing_match(tmp_path: Path) -> None:
    yaml_path = tmp_path / "prices.yaml"
    yaml_path.write_text(
        "priced-model:\n  prompt_per_1k: 0.5\n  completion_per_1k: 0.25\n",
        encoding="utf-8",
    )
    json_table = load_pricing(
        _prices(
            tmp_path / "prices.json",
            {"priced-model": {"prompt_per_1k": 0.5, "completion_per_1k": 0.25}},
        )
    )
    yaml_table = load_pricing(yaml_path)
    assert yaml_table["priced-model"] == json_table["priced-model"]


def test_malformed_pricing_file_is_a_config_error(tmp_path: Path) -> None:
    broken = tmp_path / "bad.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(PricingError, match="Malformed pricing file"):
        load_pricing(broken)
    listed = tmp_path / "list.json"
    listed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(PricingError, match="expected a mapping"):
        load_pricing(listed)
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"m": {"prompt_per_1k": 1}}) + "\n", encoding="utf-8")
    with pytest.raises(PricingError, match="missing completion_per_1k"):
        load_pricing(missing)
    negative = tmp_path / "negative.yaml"
    negative.write_text("m:\n  prompt_per_1k: -1\n  completion_per_1k: 1\n", encoding="utf-8")
    with pytest.raises(PricingError, match="non-negative"):
        load_pricing(negative)
    flagged = tmp_path / "bool.json"
    flagged.write_text(
        json.dumps({"m": {"prompt_per_1k": True, "completion_per_1k": 1}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PricingError, match="non-negative"):
        load_pricing(flagged)
    wrong_suffix = tmp_path / "prices.txt"
    wrong_suffix.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PricingError, match="Malformed pricing file"):
        load_pricing(wrong_suffix)
    absent = tmp_path / "nope.json"
    with pytest.raises(PricingError, match="Pricing file not found"):
        load_pricing(absent)


def test_compare_report_shows_the_cost_delta(tmp_path: Path) -> None:
    table = load_pricing(
        _prices(
            tmp_path / "prices.json",
            {"priced-model": {"prompt_per_1k": 0.5, "completion_per_1k": 0.25}},
        )
    )
    baseline_run = run_suite(_suite(), _Priced(), "baseline")
    candidate_run = RunResult(
        suite_name="s",
        config_name="candidate",
        results=[
            CaseResult(
                case_id="a",
                output="ok",
                scores={"exact_match": 1.0},
                score=1.0,
                passed=True,
                latency_ms=0.0,
                prompt_tokens=2000,
                completion_tokens=0,
                total_tokens=2000,
            )
        ],
    )
    baseline = RunResult(
        suite_name=baseline_run.suite_name,
        config_name=baseline_run.config_name,
        results=list(baseline_run.results),
        cost=estimate_cost(baseline_run, "priced-model", table),
    )
    candidate = RunResult(
        suite_name=candidate_run.suite_name,
        config_name=candidate_run.config_name,
        results=list(candidate_run.results),
        cost=estimate_cost(candidate_run, "priced-model", table),
    )
    assert baseline.cost is not None and baseline.cost.usd == 1.0
    # 2000 prompt tokens * 0.5 per 1k = 1.0, same as the baseline, so the delta is 0.
    assert candidate.cost is not None and candidate.cost.usd == 1.0
    comparison = compare_runs(baseline, candidate, n_boot=20, seed=1)
    path = tmp_path / "cmp.json"
    write_json(comparison, path, evaluate_gate(comparison))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["cost"]["baseline"]["usd"] == 1.0
    assert payload["cost"]["candidate"]["usd"] == 1.0
    assert payload["cost"]["delta_usd"] == 0.0
    # A second candidate with a different token mix makes a non-zero delta.
    richer = RunResult(
        suite_name="s",
        config_name="candidate",
        results=[
            CaseResult(
                case_id="a",
                output="ok",
                scores={"exact_match": 1.0},
                score=1.0,
                passed=True,
                latency_ms=0.0,
                prompt_tokens=1000,
                completion_tokens=4000,
                total_tokens=5000,
            )
        ],
        cost=estimate_cost(
            RunResult(
                suite_name="s",
                config_name="candidate",
                results=[
                    CaseResult(
                        case_id="a",
                        output="ok",
                        scores={"exact_match": 1.0},
                        score=1.0,
                        passed=True,
                        latency_ms=0.0,
                        prompt_tokens=1000,
                        completion_tokens=4000,
                        total_tokens=5000,
                    )
                ],
            ),
            "priced-model",
            table,
        ),
    )
    assert richer.cost is not None
    # 1000/1000 * 0.5 + 4000/1000 * 0.25 = 1.5, so the delta is 0.5.
    assert richer.cost.usd == 1.5
    drifted = compare_runs(baseline, richer, n_boot=20, seed=1)
    markdown = render_markdown(drifted, evaluate_gate(drifted, max_regression=1.0))
    html = render_html(drifted, evaluate_gate(drifted, max_regression=1.0))
    assert "Estimated cost" in markdown
    assert "| baseline | priced-model | 1.00000000 |  |" in markdown
    assert "| candidate | priced-model | 1.50000000 |  |" in markdown
    assert "| delta (candidate - baseline) |  | 0.50000000 |  |" in markdown
    assert "Estimated cost" in html
    assert "1.50000000" in html
    assert "0.50000000" in html


def test_cli_pricing_unknown_model_and_malformed_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [
                    {
                        "id": "a",
                        "input": "hi",
                        "reference": "yo",
                        "assertions": [{"kind": "exact_match", "value": "yo"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    pricing = tmp_path / "prices.json"
    pricing.write_text(
        json.dumps({"mock": {"prompt_per_1k": 2.0, "completion_per_1k": 4.0}}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "run.json"
    code = main(
        [
            "run",
            str(suite),
            "--provider",
            "mock",
            "--mock-rules",
            json.dumps([["hi", "yo"]]),
            "--config-name",
            "mock",
            "--pricing",
            str(pricing),
            "--out",
            str(out),
            "--no-history",
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    # prompt "hi" is 2 chars, output "yo" is 2 chars.
    expected = (2 / 1000) * 2.0 + (2 / 1000) * 4.0
    assert payload["results"][0]["prompt_tokens"] == 2
    assert payload["results"][0]["completion_tokens"] == 2
    assert payload["cost"]["usd"] == pytest.approx(expected)
    assert f"estimated_cost_usd={expected:.8f}" in capsys.readouterr().out

    unknown_prices = tmp_path / "other.json"
    unknown_prices.write_text(
        json.dumps({"other": {"prompt_per_1k": 1, "completion_per_1k": 1}}) + "\n",
        encoding="utf-8",
    )
    unknown = main(
        [
            "run",
            str(suite),
            "--provider",
            "mock",
            "--mock-rules",
            json.dumps([["hi", "yo"]]),
            "--config-name",
            "mock",
            "--pricing",
            str(unknown_prices),
            "--out",
            str(out),
            "--no-history",
        ]
    )
    assert unknown == 0
    unknown_payload = json.loads(out.read_text(encoding="utf-8"))
    assert unknown_payload["cost"]["usd"] is None
    assert unknown_payload["cost"]["note"] == "unknown model 'mock'"
    assert "estimated_cost_usd=unknown" in capsys.readouterr().out

    echo_out = tmp_path / "echo.json"
    echo = main(
        [
            "run",
            str(suite),
            "--provider",
            "echo",
            "--config-name",
            "echo",
            "--pricing",
            str(unknown_prices),
            "--out",
            str(echo_out),
            "--no-history",
        ]
    )
    assert echo == 0
    echo_payload = json.loads(echo_out.read_text(encoding="utf-8"))
    assert echo_payload["total_tokens"] == 0
    assert echo_payload["cost"]["usd"] == 0.0

    report = tmp_path / "cmp.json"
    compared = main(
        [
            "compare",
            str(suite),
            "--provider",
            "mock",
            "--mock-rules",
            json.dumps([["hi", "yo"]]),
            "--baseline-template",
            "{input}",
            "--candidate-template",
            "X{input}",
            "--pricing",
            str(pricing),
            "--max-regression",
            "1",
            "--report-json",
            str(report),
            "--no-history",
        ]
    )
    assert compared == 0
    compared_payload = json.loads(report.read_text(encoding="utf-8"))
    baseline_cost = compared_payload["cost"]["baseline"]["usd"]
    candidate_cost = compared_payload["cost"]["candidate"]["usd"]
    # Baseline prompt "hi" (2), candidate prompt "Xhi" (3), both outputs "yo" (2).
    assert baseline_cost == pytest.approx((2 / 1000) * 2.0 + (2 / 1000) * 4.0)
    assert candidate_cost == pytest.approx((3 / 1000) * 2.0 + (2 / 1000) * 4.0)
    assert compared_payload["cost"]["delta_usd"] == pytest.approx(candidate_cost - baseline_cost)

    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    failed = main(
        [
            "run",
            str(suite),
            "--provider",
            "echo",
            "--config-name",
            "echo",
            "--pricing",
            str(bad),
            "--out",
            str(out),
            "--no-history",
        ]
    )
    assert failed == 2
    assert "Malformed pricing file" in capsys.readouterr().err
