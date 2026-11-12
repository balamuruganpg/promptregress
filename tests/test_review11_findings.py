"""Regression tests for bugs found in the independent v0.6.10 review cycle.

Scope is the gate's IEEE comparisons for the always-on regression and
pass-rate checks (the same false-PASS class as the 0.6.7–0.6.10 cost,
latency, and drift budget guards).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptregress.cli import main
from promptregress.gate import evaluate_gate
from promptregress.models import (
    CaseResult,
    Comparison,
    PairedRow,
    RunResult,
)


def _comparison(*, ci_low: float = 0.0, mean_diff: float = 0.0) -> Comparison:
    row = CaseResult(
        case_id="a",
        output="x",
        scores={"contains": 1.0},
        score=1.0,
        passed=True,
        latency_ms=1.0,
    )
    run = RunResult(suite_name="s", config_name="c", results=[row])
    paired = PairedRow(
        case_id="a",
        baseline_output="x",
        candidate_output="x",
        baseline_score=1.0,
        candidate_score=1.0,
        diff=0.0,
        baseline_passed=True,
        candidate_passed=True,
        outcome="tie",
    )
    return Comparison(
        baseline=run,
        candidate=run,
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_low,
        mcnemar_p=1.0,
        wins=0,
        losses=0,
        ties=1,
        per_case=[paired],
    )


def test_nonfinite_max_regression_is_rejected_not_a_false_pass() -> None:
    """A non-finite ``max_regression`` must raise, not silently pass.

    Cost, latency, and drift budgets already reject non-finite ceilings
    with ``ValueError``. ``max_regression`` only did ``upper > budget``, so
    ``nan`` (and ``inf``) made every regression look within budget
    (``1.0 > nan`` is false). Argparse ``type=float`` accepts ``nan``, so
    ``compare --max-regression nan`` exited 0 on a clear score drop.
    """
    # Real regression: upper bound 1.0
    comparison = _comparison(ci_low=-1.0, mean_diff=-1.0)
    for budget in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="max_regression"):
            evaluate_gate(comparison, max_regression=budget)


def test_nonfinite_min_pass_rate_is_rejected_not_a_false_pass() -> None:
    """A non-finite ``min_candidate_pass_rate`` must raise, not silently pass.

    ``pass_rate < minimum`` is false for ``minimum=nan``, so a candidate
    with pass rate 0 would PASS the floor check. Match the other budget
    guards: a non-finite threshold is a config error.
    """
    row = CaseResult(
        case_id="a",
        output="x",
        scores={"contains": 0.0},
        score=0.0,
        passed=False,
        latency_ms=1.0,
    )
    run = RunResult(suite_name="s", config_name="c", results=[row])
    paired = PairedRow(
        case_id="a",
        baseline_output="ok",
        candidate_output="x",
        baseline_score=1.0,
        candidate_score=0.0,
        diff=-1.0,
        baseline_passed=True,
        candidate_passed=False,
        outcome="loss",
    )
    comparison = Comparison(
        baseline=run,
        candidate=run,
        mean_diff=0.0,
        ci_low=0.0,
        ci_high=0.0,
        mcnemar_p=1.0,
        wins=0,
        losses=1,
        ties=0,
        per_case=[paired],
    )
    with pytest.raises(ValueError, match="min_candidate_pass_rate"):
        evaluate_gate(comparison, max_regression=10.0, min_candidate_pass_rate=float("nan"))


def test_nonfinite_regression_upper_bound_fails_the_regression_check() -> None:
    """A non-finite ``-ci_low`` must FAIL regression, like invalid mean drift.

    ``compare_runs`` rejects non-finite diffs, but ``evaluate_gate`` is a
    public API and accepts a crafted or corrupted ``Comparison``. With
    ``ci_low=nan``, ``regression_upper`` is ``nan`` and ``nan > 0`` is false,
    so the gate falsely PASSed. ``ci_low=+inf`` makes upper ``-inf``, which
    also never exceeds a finite ceiling. Invalid bounds are not evidence
    the budget held.
    """
    for ci_low in (float("nan"), float("inf")):
        verdict = evaluate_gate(_comparison(ci_low=ci_low), max_regression=0.0)
        assert verdict.passed is False, ci_low
        assert any(reason.startswith("FAIL regression:") for reason in verdict.reasons)


def test_cli_rejects_nan_max_regression_instead_of_passing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``compare --max-regression nan`` must exit 2, not gate-pass a drop.

    Before the fix, argparse accepted ``nan`` and the gate treated every
    finite upper bound as within budget, so a mock candidate that scored 0
    against a perfect baseline exited 0 with ``gate_passed=true``.
    """
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "s",
                "cases": [
                    {
                        "id": "a",
                        "input": "need detail about refund",
                        "reference": "refund within 30 days",
                        "assertions": [{"kind": "contains", "value": "30 days"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "compare",
            str(suite),
            "--provider",
            "mock",
            "--mock-default",
            "refund within 30 days",
            "--baseline-template",
            "{input}",
            "--candidate-template",
            "brief {input}",
            "--mock-rules",
            json.dumps([["brief", "no"]]),
            "--max-regression",
            "nan",
            "--no-history",
        ]
    )
    assert code == 2
