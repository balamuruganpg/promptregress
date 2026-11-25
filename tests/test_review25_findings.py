"""Gate reasons, risk scaling, and comparison JSON from the v0.15.3 review.

Each test fails on v0.15.3 and locks the reporting or numeric contract.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from helpers import comparison_with, run_with
from promptregress.cli import main
from promptregress.gate import evaluate_gate
from promptregress.models import CaseResult, RunResult
from promptregress.report import format_text, write_json
from promptregress.risk import _population_std, load_comparison
from promptregress.stats import compare_runs

_EDGE = 1e308
_REGRESSION_PAIR = re.compile(
    r"regression upper bound (?P<upper>-?\d+\.\d+), max_regression (?P<budget>-?\d+\.\d+)"
)
_PASS_PAIR = re.compile(
    r"candidate pass rate (?P<rate>-?\d+\.\d+), minimum (?P<minimum>-?\d+\.\d+)"
)


def _regression_pair(reason: str) -> tuple[str, str]:
    match = _REGRESSION_PAIR.search(reason)
    assert match is not None, reason
    return match.group("upper"), match.group("budget")


def _pass_pair(reason: str) -> tuple[str, str]:
    match = _PASS_PAIR.search(reason)
    assert match is not None, reason
    return match.group("rate"), match.group("minimum")


def _case(case_id: str, score: float, passed: bool) -> CaseResult:
    return CaseResult(case_id, "out", {"exact_match": score}, score, passed, 1.0)


def _runs(
    rows: list[tuple[str, float, bool, float, bool]],
) -> tuple[RunResult, RunResult]:
    baseline = RunResult(
        "suite",
        "baseline",
        [_case(case_id, base, base_passed) for case_id, base, base_passed, _, _ in rows],
    )
    candidate = RunResult(
        "suite",
        "candidate",
        [_case(case_id, cand, cand_passed) for case_id, _, _, cand, cand_passed in rows],
    )
    return baseline, candidate


def test_regression_reason_shows_the_rounded_budget_gap() -> None:
    """A FAIL must not quote the same six decimals as the passing exact budget.

    The demo upper bound is ``6/14``. ``f"{upper:.6f}"`` is ``0.428571``, and
    ``float("0.428571")`` is smaller, so the full-precision check fails while
    both sides print ``0.428571``. The exact budget passes. Printed numerals
    have to show that, and copying either printed side has to reproduce it.
    """
    upper = 6 / 14
    comparison = comparison_with(
        ci_low=-upper,
        ci_high=0.0,
        mean_diff=-3 / 14,
        candidate_rows=[("a", 1.0, True)],
    )
    copied = float("0.428571")
    failed = evaluate_gate(comparison, max_regression=copied)
    passed = evaluate_gate(comparison, max_regression=upper)
    assert failed.passed is False
    assert passed.passed is True

    fail_upper, fail_budget = _regression_pair(failed.reasons[0])
    pass_upper, pass_budget = _regression_pair(passed.reasons[0])
    assert float(fail_upper) == upper
    assert float(fail_budget) == copied
    assert float(fail_upper) > float(fail_budget)
    assert fail_upper != fail_budget
    assert float(pass_upper) == upper
    assert float(pass_budget) == upper
    assert not (float(pass_upper) > float(pass_budget))
    assert evaluate_gate(comparison, max_regression=float(pass_budget)).passed is True
    assert evaluate_gate(comparison, max_regression=float(fail_budget)).passed is False


def test_pass_rate_reason_shows_two_thirds_below_the_printed_minimum() -> None:
    """``2/3 < float("0.666667")`` must not print both sides as ``0.666667``."""
    comparison = comparison_with(
        ci_low=0.0,
        ci_high=0.0,
        mean_diff=0.0,
        candidate_rows=[("a", 1.0, True), ("b", 1.0, True), ("c", 0.0, False)],
    )
    minimum = float("0.666667")
    verdict = evaluate_gate(comparison, max_regression=0.0, min_candidate_pass_rate=minimum)
    assert verdict.passed is False
    regression = next(reason for reason in verdict.reasons if reason.startswith("PASS regression"))
    assert "regression upper bound -0.000000, max_regression 0.000000" in regression
    rate_text, minimum_text = _pass_pair(
        next(reason for reason in verdict.reasons if reason.startswith("FAIL pass rate"))
    )
    assert float(rate_text) == pytest.approx(2 / 3)
    assert float(rate_text) == 2 / 3
    assert float(minimum_text) == minimum
    assert float(rate_text) < float(minimum_text)
    assert rate_text != minimum_text


def test_submicro_regression_fail_shows_the_tiny_values() -> None:
    """A default-gate FAIL whose six-decimal figures are all zero shows the cause."""
    tiny = 1e-7
    comparison = comparison_with(
        ci_low=-tiny,
        ci_high=-tiny,
        mean_diff=-tiny,
        candidate_rows=[("a", 0.0, False)],
        baseline_rows=[("a", 0.0, True)],
    )
    verdict = evaluate_gate(comparison, max_regression=0.0)
    assert verdict.passed is False
    reason = verdict.reasons[0]
    assert reason.startswith("FAIL regression")
    assert "regression upper bound 0.000000, max_regression 0.000000" not in reason
    upper_text, budget_text = _regression_pair(reason)
    assert float(upper_text) == tiny
    assert float(budget_text) == 0.0
    assert float(upper_text) > float(budget_text)
    assert re.search(r"mean_diff -0\.0000001\d*,", reason)
    assert re.search(r"CI \[-0\.0000001\d*, -0\.0000001\d*\]", reason)
    text = format_text(comparison, verdict)
    assert "regression upper bound 0.000000, max_regression 0.000000" not in text
    assert upper_text in text


def test_distinct_six_decimal_gate_lines_stay_six_places() -> None:
    """Ordinary gaps, including the documented signed-zero pass, stay at six places."""
    wide = comparison_with(
        ci_low=-0.4,
        ci_high=-0.1,
        mean_diff=-0.2,
        candidate_rows=[("a", 1.0, True)],
    )
    failed = evaluate_gate(wide, max_regression=0.0)
    assert "regression upper bound 0.400000, max_regression 0.000000" in failed.reasons[0]
    assert "mean_diff -0.200000," in failed.reasons[0]

    zero = comparison_with(
        ci_low=0.0,
        ci_high=0.0,
        mean_diff=0.0,
        candidate_rows=[("a", 1.0, True)],
    )
    passed = evaluate_gate(zero, max_regression=0.0)
    assert passed.passed is True
    assert "regression upper bound -0.000000, max_regression 0.000000" in passed.reasons[0]


def test_population_std_stays_finite_on_extreme_scores() -> None:
    """Finite scores stay finite: shift overflow and opposite signs included.

    ``(1e200, 2e200)`` is ``5e199``, identical values are 0, and
    ``0.5 +/- 1e-13`` keeps its ``1e-13`` deviation. ``[0.0, 1e308, 1e308]``
    is the population standard deviation ``1e308 * sqrt(2/9)``.
    ``[1e308, -1e308]`` is ``1e308``.
    """
    assert _population_std([1e200, 2e200]) == 5e199
    assert _population_std([1e308, 1e308]) == 0.0
    assert _population_std([1e308, 1e308, 1e308]) == 0.0
    assert _population_std([0.5 + 1e-13, 0.5 - 1e-13]) == 1.0000333894311098e-13

    near_zero_first = _population_std([0.0, _EDGE, _EDGE])
    other_order = _population_std([_EDGE, _EDGE, 0.0])
    expected = _EDGE * math.sqrt(2.0 / 9.0)
    assert near_zero_first == pytest.approx(expected)
    assert other_order == pytest.approx(expected)
    assert math.isfinite(near_zero_first)
    assert _population_std([_EDGE, -_EDGE]) == _EDGE
    mixed = _population_std([-_EDGE, _EDGE, 0.0])
    assert math.isfinite(mixed)
    assert mixed == pytest.approx(_EDGE * math.sqrt(2.0 / 3.0))


def test_risk_trains_on_huge_finite_scores(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The review's two ``risk`` files train instead of overflowing or returning NaN."""
    near_zero = [("c0", 0.0, True, 0.0, True)]
    near_zero.extend((f"c{index}", _EDGE, True, _EDGE, True) for index in range(2, 16))
    near_zero.insert(1, ("c1", _EDGE, True, 0.0, False))
    near_zero.extend(
        [
            ("h0", 1.0, True, 0.0, False),
            ("h1", 1.0, False, 1.0, False),
            ("h2", 1.0, True, 1.0, True),
            ("h3", 1.0, True, 1.0, True),
        ]
    )
    assert len(near_zero) == 20
    path = tmp_path / "near-zero.json"
    write_json(compare_runs(*_runs(near_zero), n_boot=20, seed=1), path)
    assert main(["risk", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_cases"] == 20
    assert payload["holdout_split"] == "last_cases"
    assert math.isfinite(payload["holdout_auc"])
    assert all(math.isfinite(value) for value in payload["scales"])
    assert "Traceback" not in capsys.readouterr().err

    alternating: list[tuple[str, float, bool, float, bool]] = []
    for index in range(20):
        baseline = _EDGE if index % 2 == 0 else -_EDGE
        if index in {0, 16}:
            alternating.append((f"c{index}", baseline, True, 0.0, False))
        else:
            alternating.append((f"c{index}", baseline, True, baseline, True))
    other = tmp_path / "opposite.json"
    write_json(compare_runs(*_runs(alternating), n_boot=20, seed=1), other)
    assert main(["risk", str(other), "--json"]) == 0
    second = json.loads(capsys.readouterr().out)
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "standardized features must be finite" not in err
    assert math.isfinite(second["holdout_auc"])
    assert all(math.isfinite(value) for value in second["scales"])


def test_risk_training_overflow_is_an_error_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An OverflowError from training is exit 2 with an ``error:`` line, not a traceback."""
    path = tmp_path / "cmp.json"
    comparison = compare_runs(
        run_with([("a", 1.0, True)], "baseline"),
        run_with([("a", 0.0, False)], "candidate"),
        n_boot=5,
        seed=1,
    )
    write_json(comparison, path)

    def boom(_comparisons: object, **_kwargs: object) -> object:
        raise OverflowError("intermediate overflow in fsum")

    monkeypatch.setattr("promptregress.cli.train", boom)
    assert main(["risk", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: intermediate overflow in fsum\n"
    assert "Traceback" not in captured.err


def test_write_json_roundtrips_signed_infinity(tmp_path: Path) -> None:
    """``compare_runs`` stores signed infinity; ``write_json`` emits it and loaders return it."""
    baseline = run_with([("a", _EDGE, True)], "baseline")
    candidate = run_with([("a", -_EDGE, False)], "candidate")
    comparison = compare_runs(baseline, candidate, n_boot=20, seed=1)
    assert comparison.per_case[0].diff == -math.inf
    assert comparison.mean_diff == -math.inf
    assert comparison.ci_low == -math.inf
    assert comparison.ci_high == -math.inf

    path = tmp_path / "cmp.json"
    write_json(comparison, path)
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "-Infinity" in text
    payload = json.loads(text)
    assert payload["mean_diff"] == comparison.mean_diff
    assert payload["ci_low"] == comparison.ci_low
    assert payload["ci_high"] == comparison.ci_high
    assert payload["per_case"][0]["diff"] == -math.inf
    assert payload["baseline"]["results"][0]["score"] == _EDGE
    assert payload["candidate"]["results"][0]["score"] == -_EDGE

    loaded = load_comparison(path)
    assert loaded.mean_diff == comparison.mean_diff
    assert loaded.ci_low == comparison.ci_low
    assert loaded.ci_high == comparison.ci_high
    assert loaded.per_case[0].diff == comparison.per_case[0].diff
    assert loaded.per_case[0].baseline_score == _EDGE
    assert loaded.per_case[0].candidate_score == -_EDGE
    assert loaded.per_case[0].baseline_passed is True
    assert loaded.per_case[0].candidate_passed is False
    assert loaded.per_case[0].outcome == "loss"


def test_write_json_roundtrips_positive_infinity_and_slice(tmp_path: Path) -> None:
    """A positive overflow and a slice mean of ``+inf`` survive the same round-trip."""
    baseline = run_with(
        [("a", -_EDGE, False), ("b", 0.0, False)],
        "baseline",
        tags={"a": ["extreme"]},
    )
    candidate = run_with(
        [("a", _EDGE, True), ("b", 0.0, False)],
        "candidate",
    )
    comparison = compare_runs(baseline, candidate, n_boot=20, seed=1)
    assert comparison.per_case[0].diff == math.inf
    assert comparison.slices[0].mean_diff == math.inf

    path = tmp_path / "cmp.json"
    write_json(comparison, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["per_case"][0]["diff"] == math.inf
    assert payload["slices"][0]["mean_diff"] == math.inf
    assert payload["slices"][0]["ci_low"] == comparison.slices[0].ci_low
    assert payload["mean_diff"] == comparison.mean_diff

    loaded = load_comparison(path)
    assert loaded.per_case[0].diff == math.inf
    assert loaded.per_case[0].outcome == "win"
    assert loaded.slices[0].tag == "extreme"
    assert loaded.slices[0].mean_diff == math.inf
    assert loaded.slices[0].ci_low == comparison.slices[0].ci_low
    assert loaded.slices[0].ci_high == comparison.slices[0].ci_high
    assert loaded.mean_diff == comparison.mean_diff


def test_finite_comparison_json_omits_infinity_tokens(tmp_path: Path) -> None:
    """Ordinary reports stay strict JSON: no ``NaN`` or ``Infinity`` tokens."""
    comparison = compare_runs(
        run_with([("a", 1.0, True)], "baseline"),
        run_with([("a", 0.0, False)], "candidate"),
        n_boot=20,
        seed=1,
    )
    path = tmp_path / "cmp.json"
    write_json(comparison, path)
    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    payload = json.loads(text)
    assert payload["mean_diff"] == comparison.mean_diff
