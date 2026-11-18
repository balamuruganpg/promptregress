"""Semantic drift: raw 1 - cosine, demo losses, gate, and reports."""

import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from promptregress.cli import (
    DEMO_BASELINE_TEMPLATE,
    DEMO_CANDIDATE_TEMPLATE,
    bundled_suite_path,
    demo_rules,
)
from promptregress.cluster import HashEmbedder, TfidfEmbedder
from promptregress.drift import DRIFT_DEFINITION, cosine_similarity, measure_drift, with_drift
from promptregress.gate import GateRule, evaluate_gate
from promptregress.loading import load_suite
from promptregress.models import CaseResult, Comparison, RunResult
from promptregress.providers import MockProvider
from promptregress.report import format_text, render_html, render_markdown, write_json
from promptregress.runner import run_suite
from promptregress.stats import compare_runs

LOSING = {"password_reset", "refund_policy", "shipping_time"}


class _FixedEmbedder:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [list(vector) for vector in self.vectors]


class _MapEmbedder:
    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(self.table[text]) for text in texts]


def _result(case_id: str, output: str, score: float) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        output=output,
        scores={"exact_match": score},
        score=score,
        passed=score >= 1.0,
        latency_ms=0.0,
    )


def _comparison(pairs: list[tuple[str, str, str, float, float]]) -> Comparison:
    baseline = RunResult(
        suite_name="suite",
        config_name="baseline",
        results=[
            _result(case_id, left, left_score) for case_id, left, _right, left_score, _ in pairs
        ],
    )
    candidate = RunResult(
        suite_name="suite",
        config_name="candidate",
        results=[
            _result(case_id, right, right_score) for case_id, _left, right, _, right_score in pairs
        ],
    )
    return compare_runs(baseline, candidate, n_boot=40, seed=1)


def test_identical_outputs_have_zero_drift_and_skip_the_embedder() -> None:
    embedder = _FixedEmbedder([[1.0, 0.0], [-1.0, 0.0]])
    comparison = _comparison(
        [
            ("same", "hello", "hello", 1.0, 1.0),
            ("empty", "", "", 1.0, 1.0),
        ]
    )
    summary = measure_drift(comparison, embedder, embedder_name="hash")
    assert embedder.calls == 0
    assert [item.drift for item in summary.per_case] == [0.0, 0.0]
    assert summary.mean_drift == 0.0
    assert summary.max_drift == 0.0


def test_raw_drift_is_one_minus_cosine_in_zero_to_two() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 1.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    comparison = _comparison([("opposite", "left", "right", 1.0, 0.0)])
    summary = measure_drift(
        comparison,
        _FixedEmbedder([[1.0, 0.0], [-1.0, 0.0]]),
        embedder_name="hash",
    )
    assert summary.per_case[0].drift == 2.0
    assert summary.max_drift == 2.0
    assert summary.mean_drift == 2.0
    assert "raw" in DRIFT_DEFINITION
    assert "[0, 2]" in DRIFT_DEFINITION


def test_zero_vector_against_text_has_drift_one() -> None:
    comparison = _comparison([("blank", "", "hello", 1.0, 0.0)])
    summary = measure_drift(comparison, HashEmbedder(), embedder_name="hash")
    assert summary.per_case[0].drift == 1.0


def test_tfidf_identical_pair_is_zero_and_disjoint_text_drifts() -> None:
    same = _comparison([("a", "alpha alpha", "alpha alpha", 1.0, 1.0)])
    assert measure_drift(same, TfidfEmbedder(), embedder_name="tfidf").mean_drift == 0.0
    split = _comparison([("a", "alpha alpha", "beta beta", 1.0, 0.0)])
    drifted = measure_drift(split, TfidfEmbedder(), embedder_name="tfidf")
    assert drifted.per_case[0].drift > 0.0


def test_top_five_orders_by_drift_then_case_id() -> None:
    unit = [1.0, 0.0]
    orthogonal = [0.0, 1.0]
    half = [0.5, math.sqrt(1.0 - 0.25)]
    quarter = [0.75, math.sqrt(1.0 - 0.75**2)]
    table = {
        "b1": unit,
        "b2": half,
        "c1": unit,
        "c2": [-1.0, 0.0],
        "d1": unit,
        "d2": orthogonal,
        "e1": unit,
        "e2": orthogonal,
        "f1": unit,
        "f2": quarter,
    }
    comparison = _comparison(
        [
            ("a", "same", "same", 1.0, 1.0),
            ("b", "b1", "b2", 1.0, 1.0),
            ("c", "c1", "c2", 1.0, 1.0),
            ("d", "d1", "d2", 1.0, 1.0),
            ("e", "e1", "e2", 1.0, 1.0),
            ("f", "f1", "f2", 1.0, 1.0),
        ]
    )
    summary = measure_drift(comparison, _MapEmbedder(table), embedder_name="map")
    assert [item.case_id for item in summary.top] == ["c", "d", "e", "b", "f"]
    assert summary.top[0].drift == 2.0
    assert summary.max_drift == 2.0
    assert len(summary.top) == 5
    assert summary.mean_drift == pytest.approx((0.0 + 0.5 + 2.0 + 1.0 + 1.0 + 0.25) / 6)


def test_demo_brief_template_drifts_on_the_losing_cases() -> None:
    suite = load_suite(bundled_suite_path())
    provider = MockProvider(rules=demo_rules(suite), default="I do not know.", model="demo")
    baseline = run_suite(suite, provider, "baseline", prompt_template=DEMO_BASELINE_TEMPLATE)
    candidate = run_suite(suite, provider, "candidate", prompt_template=DEMO_CANDIDATE_TEMPLATE)
    comparison = compare_runs(baseline, candidate, n_boot=40, seed=13)
    summary = measure_drift(comparison, HashEmbedder(), embedder_name="hash")
    by_id = {item.case_id: item.drift for item in summary.per_case}
    assert set(by_id) >= LOSING
    for case_id in LOSING:
        row = next(item for item in comparison.per_case if item.case_id == case_id)
        assert row.baseline_output != row.candidate_output
        assert by_id[case_id] > 0.0
    unchanged = [case_id for case_id, drift in by_id.items() if case_id not in LOSING]
    assert unchanged
    assert all(by_id[case_id] == 0.0 for case_id in unchanged)
    assert summary.mean_drift > 0.0
    assert {item.case_id for item in summary.top if item.drift > 0.0} == LOSING


def test_drift_gate_pass_fail_and_unset() -> None:
    comparison = with_drift(
        _comparison([("a", "left", "right", 1.0, 0.0)]),
        measure_drift(
            _comparison([("a", "left", "right", 1.0, 0.0)]),
            _FixedEmbedder([[1.0, 0.0], [-1.0, 0.0]]),
            embedder_name="hash",
        ),
    )
    # The paired scores above are 1 and 0, so the regression check also fails at budget 0.
    both = evaluate_gate(comparison, max_regression=0.0, max_mean_drift=0.0)
    assert both.passed is False
    assert any(reason.startswith("FAIL regression") for reason in both.reasons)
    assert any(
        reason.startswith("FAIL drift") and "mean_drift 2.000000" in reason
        for reason in both.reasons
    )
    ruled = evaluate_gate(
        comparison,
        max_mean_drift=2.0,
        rule=GateRule(max_regression=1.0, max_mean_drift=0.0),
    )
    assert ruled.passed is False
    assert any(reason.startswith("FAIL drift") for reason in ruled.reasons)

    held = evaluate_gate(comparison, max_regression=1.0, max_mean_drift=2.0)
    assert held.passed is True
    assert any(
        reason.startswith("PASS drift") and "max_mean_drift 2.000000" in reason
        for reason in held.reasons
    )

    unset = evaluate_gate(comparison, max_regression=1.0)
    assert unset.passed is True
    assert all("drift" not in reason for reason in unset.reasons)

    missing = evaluate_gate(
        _comparison([("a", "left", "left", 1.0, 1.0)]),
        max_mean_drift=0.0,
    )
    assert missing.passed is False
    assert any("not computed" in reason for reason in missing.reasons)


def test_reports_include_mean_max_and_top_cases(tmp_path: Path) -> None:
    comparison = with_drift(
        _comparison(
            [
                ("a", "same", "same", 1.0, 1.0),
                ("c", "c1", "c2", 1.0, 0.0),
            ]
        ),
        measure_drift(
            _comparison(
                [
                    ("a", "same", "same", 1.0, 1.0),
                    ("c", "c1", "c2", 1.0, 0.0),
                ]
            ),
            _MapEmbedder({"c1": [1.0, 0.0], "c2": [-1.0, 0.0]}),
            embedder_name="hash",
        ),
    )
    path = tmp_path / "drift.json"
    write_json(comparison, path, evaluate_gate(comparison, max_regression=1.0))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["drift"]["embedder"] == "hash"
    assert payload["drift"]["definition"] == DRIFT_DEFINITION
    assert payload["drift"]["mean_drift"] == 1.0
    assert payload["drift"]["max_drift"] == 2.0
    assert payload["drift"]["top"][0] == {"case_id": "c", "drift": 2.0}
    assert [item["case_id"] for item in payload["drift"]["per_case"]] == ["a", "c"]
    verdict = evaluate_gate(comparison, max_regression=1.0)
    markdown = render_markdown(comparison, verdict)
    html = render_html(comparison, verdict)
    plain = format_text(comparison, verdict)
    for text in (markdown, html, plain):
        assert "Semantic drift" in text or "mean_drift=" in text
        assert "c" in text
    assert "Mean drift" in markdown
    assert "Max drift" in markdown
    assert "## Semantic drift" in markdown
    assert "Semantic drift" in html
    assert "mean_drift=1.000000" in plain
    assert "max_drift=2.000000" in plain
    assert "- c 2.000000" in plain
