"""Defects found reviewing bayes.py and the suite-audit half of audit.py.

Each test quotes the README or module sentence it enforces.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from promptregress.audit import audit_runs, suite_audit_to_dict
from promptregress.bayes import (
    bayesian_comparison,
    regularized_incomplete_beta,
)
from promptregress.cli import main
from promptregress.models import CaseResult, Comparison, RunResult

HUGE_INT = 10**400
BIG = 1e200
EDGE = 1e308


def _result(case_id: str, score: float, passed: bool) -> CaseResult:
    return CaseResult(case_id, "out", {"exact_match": score}, score, passed, 1.0)


def _run(name: str, rows: list[tuple[str, float, bool]]) -> RunResult:
    return RunResult(
        "suite", name, [_result(case_id, score, passed) for case_id, score, passed in rows]
    )


def _matrix(
    scores: dict[str, list[float]],
    passes: dict[str, list[int]],
    names: tuple[str, ...] = ("c0", "c1", "c2", "c3"),
) -> list[RunResult]:
    case_ids = list(scores)
    runs: list[RunResult] = []
    for index, name in enumerate(names):
        rows = [
            (case_id, scores[case_id][index], bool(passes[case_id][index])) for case_id in case_ids
        ]
        runs.append(_run(name, rows))
    return runs


def _comparison(baseline: RunResult, candidate: RunResult) -> Comparison:
    return Comparison(
        baseline=baseline,
        candidate=candidate,
        mean_diff=0.0,
        ci_low=0.0,
        ci_high=0.0,
        mcnemar_p=1.0,
        wins=0,
        losses=0,
        ties=0,
        per_case=[],
    )


def _write_comparison(path: Path, baseline: RunResult, candidate: RunResult) -> None:
    def side(run: RunResult) -> dict[str, object]:
        return {
            "config_name": run.config_name,
            "suite_name": run.suite_name,
            "results": [
                {"case_id": item.case_id, "score": item.score, "passed": item.passed}
                for item in run.results
            ],
        }

    path.write_text(
        json.dumps({"baseline": side(baseline), "candidate": side(candidate)}),
        encoding="utf-8",
    )


def test_large_finite_totals_keep_the_negative_point_biserial() -> None:
    """A finite total whose squares overflow is still the Pearson correlation.

    README: "Pearson divides the centered sum of products by the square root
    of the two sums of squares." "A misfit is a defined point-biserial at or
    below -0.2, which is strongly negative." "Flags are advisory — the
    command exits 0 with a report and never changes the gate."

    Totals ``[1e200, 1e200, 0, 0]`` are finite, and the pass vector
    ``[0, 0, 1, 1]`` has correlation -1. Squaring the centered totals
    overflows, the ratio collapses to -0, and the case was not flagged.
    """
    scores = {
        "driver": [BIG, BIG, 0.0, 0.0],
        "misfit": [0.0, 0.0, 0.0, 0.0],
        "a": [0.0, 0.0, 0.0, 0.0],
        "b": [0.0, 0.0, 0.0, 0.0],
        "c": [0.0, 0.0, 0.0, 0.0],
    }
    passes = {
        "driver": [1, 1, 0, 0],
        "misfit": [0, 0, 1, 1],
        "a": [1, 0, 1, 0],
        "b": [0, 1, 0, 1],
        "c": [1, 1, 0, 1],
    }
    report = audit_runs(_matrix(scores, passes))
    misfit = next(case for case in report.cases if case.case_id == "misfit")
    assert misfit.point_biserial == pytest.approx(-1.0)
    assert "misfit" in misfit.flags
    assert report.config_fits[0].total_score == pytest.approx(BIG)
    encoded = json.dumps(suite_audit_to_dict(report), allow_nan=False)
    assert "Infinity" not in encoded
    assert "NaN" not in encoded


def test_tiny_finite_totals_keep_the_negative_point_biserial() -> None:
    """Underflowed squares are still the Pearson correlation.

    README: "Pearson divides the centered sum of products by the square root
    of the two sums of squares." "A misfit is a defined point-biserial at or
    below -0.2, which is strongly negative."

    Totals ``[1e-200, 1e-200, 0, 0]`` differ, so the correlation with
    ``[0, 0, 1, 1]`` is -1. Each centered square underflows to 0, the sum
    of squares is 0, and the case was treated as undefined instead of a
    misfit.
    """
    tiny = 1e-200
    scores = {
        "driver": [tiny, tiny, 0.0, 0.0],
        "misfit": [0.0, 0.0, 0.0, 0.0],
        "a": [0.0, 0.0, 0.0, 0.0],
        "b": [0.0, 0.0, 0.0, 0.0],
        "c": [0.0, 0.0, 0.0, 0.0],
    }
    passes = {
        "driver": [1, 1, 0, 0],
        "misfit": [0, 0, 1, 1],
        "a": [1, 0, 1, 0],
        "b": [0, 1, 0, 1],
        "c": [1, 1, 0, 1],
    }
    report = audit_runs(_matrix(scores, passes))
    misfit = next(case for case in report.cases if case.case_id == "misfit")
    assert misfit.point_biserial == pytest.approx(-1.0)
    assert "misfit" in misfit.flags


def test_variance_product_overflow_still_flags_the_misfit() -> None:
    """A finite pair of sums of squares can still overflow their product.

    README: "Pearson divides the centered sum of products by the square root
    of the two sums of squares." "A misfit is a defined point-biserial at or
    below -0.2, which is strongly negative."

    Eight totals ``[8e153] * 4 + [0] * 4`` have a finite sum of squared
    deviations, and ``[0, 0, 0, 0, 1, 1, 1, 1]`` has sum of squares 2. The
    product of those sums is not a finite float, so the ratio collapsed to
    0. The correlation is -1.
    """
    high = 8e153
    names = tuple(f"c{index}" for index in range(8))
    zeros = [0.0] * 8
    scores = {
        "driver": [high, high, high, high, 0.0, 0.0, 0.0, 0.0],
        "misfit": list(zeros),
        "a": list(zeros),
        "b": list(zeros),
        "c": list(zeros),
    }
    passes = {
        "driver": [1, 1, 1, 1, 0, 0, 0, 0],
        "misfit": [0, 0, 0, 0, 1, 1, 1, 1],
        "a": [1, 0, 1, 0, 1, 0, 1, 0],
        "b": [0, 1, 0, 1, 0, 1, 0, 1],
        "c": [1, 1, 0, 0, 1, 1, 0, 0],
    }
    report = audit_runs(_matrix(scores, passes, names))
    misfit = next(case for case in report.cases if case.case_id == "misfit")
    assert misfit.point_biserial == pytest.approx(-1.0)
    assert "misfit" in misfit.flags
    assert report.config_fits[0].total_score == pytest.approx(high)


def test_total_is_the_exact_sum_when_fsum_overflows_midway() -> None:
    """The config total is the accurate sum, not an ``OverflowError``.

    README: "A config's total score is math.fsum of its case scores."
    ``math.fsum`` raises ``OverflowError`` on ``[1e308, 1e308, -1e308]``
    even though the exact sum is the finite value ``1e308``. The audit
    crashed before it could report that total.
    """
    scores = {
        "wide": [EDGE, EDGE, -EDGE, 0.0, 0.0],
        "flat": [1.0, 1.0, 1.0, 1.0, 1.0],
        "a": [0.0, 1.0, 0.0, 1.0, 0.0],
        "b": [1.0, 0.0, 1.0, 0.0, 1.0],
        "c": [0.0, 0.0, 1.0, 1.0, 0.0],
    }
    passes = {
        "wide": [1, 1, 0, 0, 1],
        "flat": [1, 0, 1, 0, 1],
        "a": [1, 0, 0, 1, 0],
        "b": [0, 1, 1, 0, 1],
        "c": [0, 1, 0, 1, 0],
    }
    # Five configs: the fifth keeps the matrix at MIN_CONFIGS after the four
    # above, and its scores cancel the same way.
    names = ("c0", "c1", "c2", "c3", "c4")
    widened = {case_id: values + [values[-1]] for case_id, values in scores.items()}
    extended = {case_id: bits + [bits[-1]] for case_id, bits in passes.items()}
    # c0 carries the cancelling series. Other configs use the last column of
    # ``widened``, which is finite and small except ``wide``'s last 0.
    per_config = []
    for index, name in enumerate(names):
        if index == 0:
            rows = [
                (case_id, widened[case_id][0], bool(extended[case_id][0])) for case_id in widened
            ]
        else:
            rows = [
                (
                    case_id,
                    0.0 if case_id == "wide" else widened[case_id][index],
                    bool(extended[case_id][index]),
                )
                for case_id in widened
            ]
        per_config.append(_run(name, rows))
    # Rebuild c0 explicitly so its five scores are the cancelling series.
    per_config[0] = _run(
        "c0",
        [
            ("wide", EDGE, True),
            ("flat", EDGE, True),
            ("a", -EDGE, False),
            ("b", 0.0, False),
            ("c", 0.0, True),
        ],
    )
    report = audit_runs(per_config)
    totals = {fit.config: fit.total_score for fit in report.config_fits}
    assert totals["c0"] == EDGE
    assert math.isfinite(totals["c0"])
    encoded = json.dumps(suite_audit_to_dict(report), allow_nan=False)
    assert "Infinity" not in encoded


def test_nonrepresentable_total_raises_valueerror(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A total outside the float range is ``ValueError``, not a crash.

    README: "Fewer than 4 configs or fewer than 5 cases raises ValueError,
    as do non-finite scores and unpairable inputs." The command exits 2.
    Five scores of ``1e308`` are finite, and ``math.fsum`` raises
    ``OverflowError`` because ``5e308`` is not a finite float. The audit
    must refuse with ``ValueError`` and must not write ``Infinity``.
    """
    scores = {case_id: [EDGE, EDGE, EDGE, EDGE] for case_id in "abcde"}
    passes = {
        "a": [1, 1, 0, 0],
        "b": [1, 0, 1, 0],
        "c": [0, 1, 0, 1],
        "d": [1, 0, 0, 1],
        "e": [0, 1, 1, 0],
    }
    with pytest.raises(ValueError, match="not a finite number"):
        audit_runs(_matrix(scores, passes))

    runs = _matrix(scores, passes)
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_comparison(left, runs[0], runs[1])
    _write_comparison(right, runs[2], runs[3])
    code = main(["audit", str(left), str(right)])
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "not a finite number" in err
    assert "OverflowError" not in err


def test_huge_integer_score_and_delta_raise_valueerror(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A number that does not fit in float raises ``ValueError`` (exit 2).

    README (bayes): "An empty comparison, a comparison whose cases cannot
    be paired, a non-finite score, or a delta outside [0, 1) raises
    ValueError (exit 2)." README (suite audit): "Fewer than 4 configs or
    fewer than 5 cases raises ValueError, as do non-finite scores and
    unpairable inputs."

    ``float`` of a 400-digit integer raises ``OverflowError``, which the
    CLI does not catch, so the process crashed instead of exiting 2.
    """
    rows = [("a", 1.0, True)]
    base = _run("baseline", rows)
    cand = _run("candidate", [("a", 0.0, False)])
    huge_base = RunResult("suite", "baseline", [_result("a", HUGE_INT, True)])
    with pytest.raises(ValueError, match="finite number"):
        bayesian_comparison(_comparison(huge_base, cand))
    with pytest.raises(ValueError, match="delta"):
        bayesian_comparison(_comparison(base, cand), delta=HUGE_INT)
    with pytest.raises(ValueError, match="finite number"):
        regularized_incomplete_beta(0.5, HUGE_INT, 2)

    with pytest.raises(ValueError, match="finite number"):
        audit_runs(
            _matrix(
                {case_id: [HUGE_INT] * 4 for case_id in "abcde"},
                {case_id: [1, 0, 1, 0] for case_id in "abcde"},
            )
        )

    path = tmp_path / "comparison.json"
    payload = {
        "baseline": {
            "config_name": "baseline",
            "results": [
                {"case_id": "a", "score": HUGE_INT, "passed": True},
                *[{"case_id": cid, "score": 1, "passed": True} for cid in "bcde"],
            ],
        },
        "candidate": {
            "config_name": "candidate",
            "results": [{"case_id": cid, "score": 0, "passed": False} for cid in "abcde"],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["bayes", str(path)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "OverflowError" not in err

    code = main(["audit", str(path), str(path)])
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "finite" in err
