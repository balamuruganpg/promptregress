"""Pass/fail gate for a paired comparison. The exit code in CI comes from this."""

import math
from dataclasses import dataclass

from promptregress.models import Comparison


@dataclass(frozen=True)
class GateRule:
    """Thresholds for :func:`evaluate_gate`.

    ``max_regression`` is the largest allowed upper bound on the score drop.
    ``min_candidate_pass_rate`` is a floor on the candidate's pass rate.
    ``require_significant`` demands that the bootstrap interval for
    (candidate - baseline) lie entirely above 0.
    ``max_mean_drift`` is an optional ceiling on mean semantic drift.
    ``None`` leaves that check off.
    """

    max_regression: float = 0.0
    min_candidate_pass_rate: float = 0.0
    require_significant: bool = False
    max_mean_drift: float | None = None


@dataclass(frozen=True)
class GateVerdict:
    """Whether the gate passed, with one human-readable reason per check."""

    passed: bool
    reasons: list[str]


def evaluate_gate(
    comparison: Comparison,
    max_regression: float = 0.0,
    min_candidate_pass_rate: float = 0.0,
    require_significant: bool = False,
    *,
    max_mean_drift: float | None = None,
    rule: GateRule | None = None,
) -> GateVerdict:
    """Decide whether a candidate is allowed to replace the baseline.

    Per-case differences are ``candidate_score - baseline_score``. The
    regression magnitude's upper bound is ``-ci_low`` (the worst end of the
    percentile interval). The gate fails when that upper bound is greater
    than ``max_regression``. Using the interval, not the point estimate, is
    the conservative check: a change fails if the interval still allows a
    drop larger than the budget.

    The gate also fails when the candidate pass rate is below
    ``min_candidate_pass_rate``. When ``require_significant`` is true, the
    gate fails unless ``ci_low > 0`` (the improvement interval is entirely
    positive). A ``GateRule`` passed as ``rule`` replaces the scalar arguments,
    including ``max_mean_drift``.

    ``max_mean_drift`` is optional. When it is ``None`` the drift check is
    skipped and no drift reason is recorded, which is the historical gate.
    When it is set, the gate fails if semantic drift was not computed or if
    ``comparison.drift.mean_drift`` is greater than the budget. An equal
    mean passes. Drift is ``1 - cosine`` in ``[0, 2]``.

    Every check that runs adds a reason that includes the numbers it used.
    ``passed`` is false when any reason starts with ``FAIL``.
    """
    if rule is not None:
        max_regression = rule.max_regression
        min_candidate_pass_rate = rule.min_candidate_pass_rate
        require_significant = rule.require_significant
        max_mean_drift = rule.max_mean_drift

    reasons: list[str] = []
    regression_upper = -comparison.ci_low
    level = int(round((1.0 - comparison.alpha) * 100))
    detail = (
        f"mean_diff {_fmt(comparison.mean_diff)}, "
        f"{level}% CI [{_fmt(comparison.ci_low)}, {_fmt(comparison.ci_high)}], "
        f"regression upper bound {_fmt(regression_upper)}, "
        f"max_regression {_fmt(max_regression)}"
    )
    if regression_upper > max_regression:
        reasons.append(f"FAIL regression: upper bound exceeds max_regression ({detail}).")
    else:
        reasons.append(f"PASS regression: upper bound is within max_regression ({detail}).")

    pass_rate = comparison.candidate.pass_rate
    rate_detail = f"candidate pass rate {_fmt(pass_rate)}, minimum {_fmt(min_candidate_pass_rate)}"
    if pass_rate < min_candidate_pass_rate:
        reasons.append(f"FAIL pass rate: candidate is below the minimum ({rate_detail}).")
    else:
        reasons.append(f"PASS pass rate: candidate meets the minimum ({rate_detail}).")

    if require_significant:
        if comparison.ci_low > 0:
            reasons.append(
                "PASS significance: improvement CI is entirely above 0 "
                f"(ci_low {_fmt(comparison.ci_low)})."
            )
        else:
            reasons.append(
                "FAIL significance: require_significant is set and ci_low "
                f"{_fmt(comparison.ci_low)} is not above 0."
            )
    else:
        reasons.append("PASS significance: require_significant is off.")

    if max_mean_drift is not None:
        reasons.append(_drift_reason(comparison, max_mean_drift))

    passed = not any(reason.startswith("FAIL ") for reason in reasons)
    return GateVerdict(passed=passed, reasons=reasons)


def _drift_reason(comparison: Comparison, max_mean_drift: float) -> str:
    if isinstance(max_mean_drift, bool) or not isinstance(max_mean_drift, (int, float)):
        raise ValueError("max_mean_drift must be a finite number")
    budget = float(max_mean_drift)
    if not math.isfinite(budget):
        raise ValueError("max_mean_drift must be a finite number")
    summary = comparison.drift
    if summary is None:
        return (
            "FAIL drift: max_mean_drift is set but semantic drift was not computed "
            f"(max_mean_drift {_fmt(budget)})."
        )
    detail = (
        f"mean_drift {_fmt(summary.mean_drift)}, "
        f"max_drift {_fmt(summary.max_drift)}, "
        f"max_mean_drift {_fmt(budget)}"
    )
    if summary.mean_drift > budget:
        return f"FAIL drift: mean drift exceeds max_mean_drift ({detail})."
    return f"PASS drift: mean drift is within max_mean_drift ({detail})."


def _fmt(value: float) -> str:
    return f"{value:.6f}"
