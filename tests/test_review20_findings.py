"""Reproductions for numerical bugs in risk training and conformal calibration."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from promptregress.conformal import calibrate_threshold, conformal_to_dict
from promptregress.models import CaseResult, Comparison, RunResult
from promptregress.risk import FEATURE_NAMES, predict_proba, rank_cases, risk_to_dict, train


def _comparison(
    latencies: tuple[float, float] = (10.0, 10.0),
    scores: tuple[float, float] = (0.5, 0.5),
) -> Comparison:
    """Twelve alternating labels; only the requested features can separate them."""
    baseline = [
        CaseResult(
            case_id=f"c{index}",
            output="same",
            scores={},
            score=scores[index % 2],
            passed=True,
            latency_ms=latencies[index % 2],
        )
        for index in range(12)
    ]
    candidate = [
        replace(case, score=math.nextafter(case.score, -math.inf) if index % 2 == 0 else case.score)
        for index, case in enumerate(baseline)
    ]
    return Comparison(
        baseline=RunResult("numeric", "baseline", baseline),
        candidate=RunResult("numeric", "candidate", candidate),
        mean_diff=0.0,
        ci_low=0.0,
        ci_high=0.0,
        mcnemar_p=1.0,
        wins=0,
        losses=6,
        ties=6,
        per_case=[],
    )


@pytest.mark.parametrize(
    ("labels", "alpha", "expected_threshold", "expected_bound"),
    [
        ([0], 0.5 - 1e-13, math.inf, 0.0),
        ([0], math.nextafter(0.5, 0.0), math.inf, 0.0),
        ([0], 0.5, 0.4, 0.5),
        ([1], math.nextafter(1.0, 0.0), math.inf, 0.0),
    ],
)
def test_conformal_epsilon_cannot_promote_an_extra_tail_rank(
    labels: list[int], alpha: float, expected_threshold: float, expected_bound: float
) -> None:
    """Epsilon rounded alpha upward across a rank boundary, violating bound <= alpha.

    Just below 1/2, one fail needs the padded infinity; with no fails the
    tail count is always zero. Exact boundary arithmetic preserves both rules.
    """
    result = calibrate_threshold([0.4], labels, alpha)
    assert result.threshold == expected_threshold
    assert result.guaranteed_bound == expected_bound
    assert result.guaranteed_bound <= alpha
    json.dumps(conformal_to_dict(result), allow_nan=False)


def test_risk_small_nonzero_variance_is_standardized() -> None:
    """The 1e-12 scale cutoff erased a separating baseline-score feature.

    Population standardization must preserve every nonconstant feature,
    even when its units make the standard deviation smaller than 1e-12.
    """
    comparison = _comparison(scores=(0.5 + 1e-13, 0.5 - 1e-13))
    model = train([comparison, comparison])
    index = FEATURE_NAMES.index("baseline_score")
    expected_sd = (comparison.baseline.results[0].score - comparison.baseline.results[1].score) / 2
    assert model.scales[index] == pytest.approx(expected_sd, rel=1e-12, abs=0.0)
    assert model.holdout_auc == 1.0


@pytest.mark.parametrize("latencies", [(1e200, 2e200), (1e308, 1e308)])
def test_risk_finite_training_features_do_not_overflow_scaling(
    latencies: tuple[float, float],
) -> None:
    """Naive sums/squared deviations overflowed on finite columns.

    A stable population mean and deviation keep both varying and constant
    large features trainable and the resulting model valid for strict JSON.
    """
    comparison = _comparison(latencies=latencies)
    model = train([comparison, comparison])
    index = FEATURE_NAMES.index("baseline_latency_ms")
    assert math.isfinite(model.means[index])
    assert math.isfinite(model.scales[index])
    if latencies[0] == latencies[1]:
        assert model.means[index] == latencies[0]
        assert model.scales[index] == 1.0
        assert model.coefficients[index] == 0.0
        assert model.holdout_auc == 0.5
    else:
        assert model.scales[index] == pytest.approx(5e199)
        assert model.holdout_auc == 1.0
    json.dumps(risk_to_dict(model), allow_nan=False)


@pytest.mark.parametrize("latencies", [(1e-4, 2e-4), (1.0, 2.0)])
def test_risk_extreme_target_cannot_emit_nonfinite_contributions(
    latencies: tuple[float, float],
) -> None:
    """Finite targets overflowed standardization or coefficient products into JSON.

    An unrepresentable feature/contribution must raise a clear ValueError
    before returning a ranking containing Infinity or NaN.
    """
    comparison = _comparison(latencies=latencies)
    model = train([comparison, comparison])
    target = replace(comparison.baseline.results[0], latency_ms=1e308)
    with pytest.raises(ValueError, match="finite"):
        rank_cases(model, [target])


def test_risk_sigmoid_clipping_does_not_turn_holdout_ordering_into_ties() -> None:
    """Clipping logits at -30 made distinct representable probabilities both zero.

    A stable logistic function retains the ordering of shifted holdout cases,
    so Mann-Whitney AUC and case ranking reflect the learned linear score.
    """
    training = _comparison(latencies=(1.0, 2.0))
    holdout = _comparison(latencies=(10.0, 11.0))
    model = train([training, holdout])
    probabilities = predict_proba(model, holdout.baseline.results)
    assert probabilities[0] > probabilities[1] > 0.0
    assert model.holdout_auc == 1.0


def test_write_json_rejects_non_finite_values(tmp_path: Path) -> None:
    """_write_json serialized NaN/Infinity as bare tokens (invalid JSON).

    Every other report writer in the project dumps with allow_nan=False;
    the shared helper now does the same, so a non-finite payload raises
    ValueError instead of writing a file strict parsers reject.
    """
    from promptregress.cli import _write_json

    with pytest.raises(ValueError):
        _write_json(str(tmp_path / "bad.json"), {"score": float("nan")})
    with pytest.raises(ValueError):
        _write_json(str(tmp_path / "bad2.json"), {"score": float("inf")})
