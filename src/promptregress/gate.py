"""Pass/fail gate for a paired comparison. The exit code in CI comes from this."""

import math
from dataclasses import dataclass

from promptregress.models import Comparison, RunResult


@dataclass(frozen=True)
class GateRule:
    """Thresholds for :func:`evaluate_gate`.

    ``max_regression`` is the largest allowed upper bound on the score drop.
    ``min_candidate_pass_rate`` is a floor on the candidate's pass rate.
    ``require_significant`` demands that the bootstrap interval for
    (candidate - baseline) lie entirely above 0.
    ``max_mean_drift`` is an optional ceiling on mean semantic drift.
    ``None`` leaves that check off.
    ``max_cost_usd`` is an optional ceiling on the candidate's estimated
    cost. ``max_p95_latency_ms`` is an optional ceiling on the candidate's
    p95 case latency. ``None`` leaves each of those checks off.
    """

    max_regression: float = 0.0
    min_candidate_pass_rate: float = 0.0
    require_significant: bool = False
    max_mean_drift: float | None = None
    max_cost_usd: float | None = None
    max_p95_latency_ms: float | None = None


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
    max_cost_usd: float | None = None,
    max_p95_latency_ms: float | None = None,
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
    including ``max_mean_drift``, ``max_cost_usd``, and ``max_p95_latency_ms``.

    ``max_mean_drift`` is optional. When it is ``None`` the drift check is
    skipped and no drift reason is recorded, which is the historical gate.
    When it is set, the gate fails if semantic drift was not computed, if
    ``comparison.drift.mean_drift`` is non-finite or negative, or if the
    mean is greater than the budget. An equal finite non-negative mean
    passes. Drift is ``1 - cosine`` in ``[0, 2]``.

    ``max_cost_usd`` and ``max_p95_latency_ms`` are optional and apply to
    the candidate run. ``None`` skips that check and records no reason.
    Cost uses the estimate already stored on the candidate. A missing
    estimate, an estimate whose ``usd`` is ``None``, or a non-finite or
    negative ``usd`` fails a set cost budget. Latency uses
    ``RunResult.p95_latency_ms``. A non-finite or negative p95 fails a set
    latency budget. An equal finite non-negative cost or an equal finite
    non-negative p95 passes. A greater value fails. The reason names the
    numbers, for example ``cost $1.500000 exceeds budget $1.000000`` and
    ``p95 latency 120.000000ms exceeds budget 100.000000ms``.

    ``max_regression`` and ``min_candidate_pass_rate`` must be finite numbers.
    A non-finite value raises ``ValueError`` (the same rule as a non-finite
    cost, latency, or drift budget). A non-finite regression upper bound
    (``-ci_low``) fails the regression check: ``nan > budget`` and
    ``-inf > budget`` are false in IEEE compare and would otherwise PASS.

    Every check that runs adds a reason that includes the numbers it used.
    ``passed`` is false when any reason starts with ``FAIL``.
    """
    if rule is not None:
        max_regression = rule.max_regression
        min_candidate_pass_rate = rule.min_candidate_pass_rate
        require_significant = rule.require_significant
        max_mean_drift = rule.max_mean_drift
        max_cost_usd = rule.max_cost_usd
        max_p95_latency_ms = rule.max_p95_latency_ms

    max_regression = _budget_number(max_regression, "max_regression")
    min_candidate_pass_rate = _budget_number(min_candidate_pass_rate, "min_candidate_pass_rate")

    reasons: list[str] = []
    regression_upper = -comparison.ci_low
    level = int(round((1.0 - comparison.alpha) * 100))
    detail = (
        f"mean_diff {_fmt(comparison.mean_diff)}, "
        f"{level}% CI [{_fmt(comparison.ci_low)}, {_fmt(comparison.ci_high)}], "
        f"regression upper bound {_fmt(regression_upper)}, "
        f"max_regression {_fmt(max_regression)}"
    )
    # Same IEEE trap as an invalid cost / latency / drift observation:
    # ``nan > budget`` and ``-inf > budget`` never hold, so a non-finite
    # upper bound would falsely PASS any finite ceiling.
    if not math.isfinite(regression_upper):
        reasons.append(f"FAIL regression: regression upper bound is invalid ({detail}).")
    elif regression_upper > max_regression:
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
    reasons.extend(
        budget_reasons(
            comparison.candidate,
            max_cost_usd=max_cost_usd,
            max_p95_latency_ms=max_p95_latency_ms,
        )
    )

    passed = not any(reason.startswith("FAIL ") for reason in reasons)
    return GateVerdict(passed=passed, reasons=reasons)


def evaluate_run_budgets(
    result: RunResult,
    *,
    max_cost_usd: float | None = None,
    max_p95_latency_ms: float | None = None,
) -> GateVerdict:
    """Cost and latency checks for a single run.

    Both limits are optional. ``None`` records no reason, so an unset
    budget does not change a run that otherwise passes. The same checks
    are what :func:`evaluate_gate` applies to the candidate.
    """
    reasons = budget_reasons(
        result,
        max_cost_usd=max_cost_usd,
        max_p95_latency_ms=max_p95_latency_ms,
    )
    passed = not any(reason.startswith("FAIL ") for reason in reasons)
    return GateVerdict(passed=passed, reasons=reasons)


def budget_reasons(
    result: RunResult,
    *,
    max_cost_usd: float | None,
    max_p95_latency_ms: float | None,
) -> list[str]:
    """Reasons for the cost and latency budgets that are set.

    An unset budget contributes nothing. Each set budget contributes one
    ``PASS`` or ``FAIL`` reason.
    """
    reasons: list[str] = []
    if max_cost_usd is not None:
        reasons.append(_cost_reason(result, max_cost_usd))
    if max_p95_latency_ms is not None:
        reasons.append(_latency_reason(result, max_p95_latency_ms))
    return reasons


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
    mean_drift = summary.mean_drift
    detail = (
        f"mean_drift {_fmt(mean_drift)}, "
        f"max_drift {_fmt(summary.max_drift)}, "
        f"max_mean_drift {_fmt(budget)}"
    )
    # Same rule as an invalid cost or latency observation: ``nan > budget``
    # and a negative mean never compare greater, so they would falsely PASS.
    # Documented drift is in ``[0, 2]``.
    if not math.isfinite(mean_drift) or mean_drift < 0:
        return f"FAIL drift: mean drift is invalid ({detail})."
    if mean_drift > budget:
        return f"FAIL drift: mean drift exceeds max_mean_drift ({detail})."
    return f"PASS drift: mean drift is within max_mean_drift ({detail})."


def _cost_reason(result: RunResult, max_cost_usd: float) -> str:
    budget = _budget_number(max_cost_usd, "max_cost_usd")
    estimate = result.cost
    if estimate is None or estimate.usd is None:
        if estimate is None:
            detail = "cost was not estimated"
        else:
            note = estimate.note or "unknown model"
            detail = f"cost is unknown ({note})"
        return f"FAIL cost: {detail} (budget {_usd(budget)})."
    usd = estimate.usd
    # Non-finite or negative USD is not a valid estimate. A raw ``usd > budget``
    # compare would treat NaN / negatives as within budget.
    if not math.isfinite(usd) or usd < 0:
        return f"FAIL cost: cost is invalid ({_usd(usd)}) (budget {_usd(budget)})."
    if usd > budget:
        return f"FAIL cost: cost {_usd(usd)} exceeds budget {_usd(budget)}."
    return f"PASS cost: cost {_usd(usd)} is within budget {_usd(budget)}."


def _latency_reason(result: RunResult, max_p95_latency_ms: float) -> str:
    budget = _budget_number(max_p95_latency_ms, "max_p95_latency_ms")
    observed = result.p95_latency_ms
    # Same rule as an invalid cost estimate: a non-finite or negative p95
    # cannot pass a set budget. ``nan > budget`` and ``-1 > budget`` are both
    # false in IEEE compare and would PASS.
    if not math.isfinite(observed) or observed < 0:
        return (
            f"FAIL latency: p95 latency is invalid ({_fmt(observed)}ms) (budget {_fmt(budget)}ms)."
        )
    if observed > budget:
        return f"FAIL latency: p95 latency {_fmt(observed)}ms exceeds budget {_fmt(budget)}ms."
    return f"PASS latency: p95 latency {_fmt(observed)}ms is within budget {_fmt(budget)}ms."


def _budget_number(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _usd(value: float) -> str:
    return f"${_fmt(value)}"


def _fmt(value: float) -> str:
    return f"{value:.6f}"
