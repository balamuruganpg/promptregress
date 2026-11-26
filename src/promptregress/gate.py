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

    Regression, pass-rate, significance, drift, cost, and latency reasons
    print six decimal places. When that rendering makes the two compared
    values look identical, or rounds a nonzero value to a signed zero and
    hides the comparison, those reasons use more places. The printed pair
    then compares the same way as the full-precision check, and each
    numeral parses back to the value that was compared. Significance
    compares ``ci_low`` with 0. Drift, cost, and latency compare the
    observation with its budget. The decision is unchanged.

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
    places = _decision_places(regression_upper, max_regression)
    detail = (
        f"mean_diff {_fmt_at(comparison.mean_diff, places)}, "
        f"{level}% CI ["
        f"{_fmt_at(comparison.ci_low, places)}, {_fmt_at(comparison.ci_high, places)}], "
        f"regression upper bound {_fmt_at(regression_upper, places)}, "
        f"max_regression {_fmt_at(max_regression, places)}"
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
    rate_places = _decision_places(pass_rate, min_candidate_pass_rate)
    rate_detail = (
        f"candidate pass rate {_fmt_at(pass_rate, rate_places)}, "
        f"minimum {_fmt_at(min_candidate_pass_rate, rate_places)}"
    )
    if pass_rate < min_candidate_pass_rate:
        reasons.append(f"FAIL pass rate: candidate is below the minimum ({rate_detail}).")
    else:
        reasons.append(f"PASS pass rate: candidate meets the minimum ({rate_detail}).")

    if require_significant:
        shown = _fmt_at(comparison.ci_low, _decision_places(comparison.ci_low, 0.0))
        if comparison.ci_low > 0:
            reasons.append(
                f"PASS significance: improvement CI is entirely above 0 (ci_low {shown})."
            )
        else:
            reasons.append(
                f"FAIL significance: require_significant is set and ci_low {shown} is not above 0."
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
    places = _decision_places(mean_drift, budget)
    detail = (
        f"mean_drift {_fmt_at(mean_drift, places)}, "
        f"max_drift {_fmt_at(summary.max_drift, places)}, "
        f"max_mean_drift {_fmt_at(budget, places)}"
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
    places = _decision_places(usd, budget)
    shown = _usd_at(usd, places)
    limit = _usd_at(budget, places)
    if not math.isfinite(usd) or usd < 0:
        return f"FAIL cost: cost is invalid ({shown}) (budget {limit})."
    if usd > budget:
        return f"FAIL cost: cost {shown} exceeds budget {limit}."
    return f"PASS cost: cost {shown} is within budget {limit}."


def _latency_reason(result: RunResult, max_p95_latency_ms: float) -> str:
    budget = _budget_number(max_p95_latency_ms, "max_p95_latency_ms")
    observed = result.p95_latency_ms
    # Same rule as an invalid cost estimate: a non-finite or negative p95
    # cannot pass a set budget. ``nan > budget`` and ``-1 > budget`` are both
    # false in IEEE compare and would PASS.
    places = _decision_places(observed, budget)
    shown = _fmt_at(observed, places)
    limit = _fmt_at(budget, places)
    if not math.isfinite(observed) or observed < 0:
        return f"FAIL latency: p95 latency is invalid ({shown}ms) (budget {limit}ms)."
    if observed > budget:
        return f"FAIL latency: p95 latency {shown}ms exceeds budget {limit}ms."
    return f"PASS latency: p95 latency {shown}ms is within budget {limit}ms."


def _budget_number(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _usd(value: float) -> str:
    return _usd_at(value, 6)


def _usd_at(value: float, places: int) -> str:
    return f"${_fmt_at(value, places)}"


def _fmt(value: float) -> str:
    return _fmt_at(value, 6)


def _fmt_at(value: float, places: int) -> str:
    return f"{value:.{places}f}"


def _fmt_vs_zero(value: float) -> str:
    """Format a number with enough decimal places to distinguish it from zero."""
    return _fmt_at(value, _decision_places(value, 0.0))


def _fmt_pair(left: float, right: float) -> tuple[str, str]:
    """Format two numbers at the same precision, preserving their order."""
    places = 6 if left == right else _decision_places(left, right)
    return _fmt_at(left, places), _fmt_at(right, places)


def _decision_places(left: float, right: float) -> int:
    """Decimal places that keep a printed comparison faithful.

    Six places is the default. More places are used when that rendering
    makes the two strings identical, or rounds a nonzero value to a signed
    zero and hides the comparison. The extra places are the fewest at which
    the printed numerals compare the same way as the full-precision values
    and each numeral parses back to the value it came from.
    """
    if not math.isfinite(left) or not math.isfinite(right):
        return 6
    left_text = f"{left:.6f}"
    right_text = f"{right:.6f}"
    if left_text != right_text and not _signed_zero_hides(left, right, left_text, right_text):
        return 6
    agreed: int | None = None
    for places in range(6, 360):
        left_text = f"{left:.{places}f}"
        right_text = f"{right:.{places}f}"
        if not _printed_agrees(left, right, left_text, right_text):
            continue
        if float(left_text) == left and float(right_text) == right:
            return places
        if agreed is None:
            agreed = places
    return 6 if agreed is None else agreed


def _signed_zero_hides(left: float, right: float, left_text: str, right_text: str) -> bool:
    hidden = _nonzero_as_zero(left, left_text) or _nonzero_as_zero(right, right_text)
    return hidden and not _printed_agrees(left, right, left_text, right_text)


def _nonzero_as_zero(value: float, text: str) -> bool:
    if value == 0.0 or not math.isfinite(value):
        return False
    body = text[1:] if text.startswith("-") else text
    whole, dot, frac = body.partition(".")
    return dot == "." and whole in {"0", ""} and frac != "" and set(frac) <= {"0"}


def _printed_agrees(left: float, right: float, left_text: str, right_text: str) -> bool:
    printed_left = float(left_text)
    printed_right = float(right_text)
    return (
        (left > right) == (printed_left > printed_right)
        and (left < right) == (printed_left < printed_right)
        and (left == right) == (printed_left == printed_right)
    )
