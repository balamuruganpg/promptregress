"""Split-conformal threshold for LLM-judge pass/fail decisions.

Given calibration pairs ``(score, label)`` — a judge score in ``[0, 1]`` and a
gold pass/fail bit — :func:`calibrate_threshold` returns the score cut that
controls the false-pass rate at a user-chosen ``alpha`` under exchangeability.

A false pass is a gold-fail example the judge still passes. The nonconformity
score of a gold-fail example is the judge score itself (higher is more
pass-like). Pass-class rows do not enter the quantile; they are Mondrian /
class-conditional conformal on the fail class.

The cut is the standard split-conformal quantile: the
``k = ceil((n + 1) (1 - alpha))``-th order statistic of the ``n`` gold-fail
scores (the empirical quantile at level ``k / n``). When ``k > n`` the
quantile is ``+inf`` and the judge must never pass. That is the usual
``+inf`` padding of the calibration bag.

Simplifications, stated here so the numbers are not read as more than they
are:

- The finite-sample guarantee is ``P(s_new > threshold) <= guaranteed_bound``
  for a new gold-fail score exchangeable with the calibration fail scores.
  ``guaranteed_bound`` is ``floor(alpha * (n + 1)) / (n + 1)``, which is at
  most ``alpha``. The guarantee needs exchangeability (or i.i.d. as a
  special case). It does not need a well-calibrated judge, a Gaussian
  score, or a large ``n``.
- Prediction uses ``score >= threshold``. A score tied at the threshold is
  a pass. Equal scores sort as equal, so the ``k``-th order statistic is
  the unique value at that rank. The exchangeability bound is for the
  strict tail ``s_new > threshold``; atoms at the threshold can raise the
  observed false-pass rate by that point mass.
- ``n`` is the number of gold-fail calibration rows, not the number of
  pairs. All-pass input yields ``n = 0``, ``threshold = +inf``, and
  empirical error 0.
- The bound is for a new fail example, not for the empirical rate on a
  finite test set, and not for false-fails (gold pass, judge fail).
- Scores outside ``[0, 1]`` or non-finite scores are rejected. Labels must
  be the integers ``0`` and ``1``; ``True`` / ``False`` are not accepted.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Snap (n+1)*alpha that lands a few ulps above an integer back onto it, so
# ceil((n+1)(1-alpha)) is 9 when (n+1)(1-alpha) is 9.000000000000002.
_RANK_EPS = 1e-12


@dataclass(frozen=True)
class ConformalResult:
    """Split-conformal false-pass threshold fit on gold-fail scores.

    ``n`` is the number of gold-fail calibration scores. ``threshold`` is
    ``+inf`` when no finite cut meets the bound (never pass).
    ``guaranteed_bound`` is ``floor(alpha * (n + 1)) / (n + 1)``.
    ``empirical_error_rate`` is the false-pass rate of this threshold on
    the same calibration fail scores (ties at the threshold count as pass).
    """

    threshold: float
    n: int
    alpha: float
    guaranteed_bound: float
    empirical_error_rate: float


@dataclass(frozen=True)
class ConformalEvaluation:
    """Empirical false-pass rate and fail-class coverage at a threshold.

    ``coverage`` is ``1 - false_pass_rate`` on gold-fail rows. When there
    are no gold-fail rows both rates are 0 and 1 respectively.
    """

    n: int
    n_fail: int
    n_false_pass: int
    n_false_fail: int
    false_pass_rate: float
    coverage: float


def calibrate_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    alpha: float,
) -> ConformalResult:
    """Fit a split-conformal score threshold that bounds the false-pass rate.

    ``scores`` are judge scores in ``[0, 1]``. ``labels`` are gold bits
    (``1`` pass, ``0`` fail). ``alpha`` is the target false-pass level in
    ``(0, 1)``.

    The threshold is the ``ceil((n + 1) (1 - alpha))``-th order statistic of
    the gold-fail scores, or ``+inf`` when that index is ``n + 1``.
    """
    values, bits = _pairs(scores, labels)
    level = _alpha(alpha)
    fail_scores = [score for score, label in zip(values, bits, strict=True) if label == 0]
    n = len(fail_scores)
    threshold = _fail_threshold(fail_scores, level)
    bound = _guaranteed_bound(n, level)
    return ConformalResult(
        threshold=threshold,
        n=n,
        alpha=level,
        guaranteed_bound=bound,
        empirical_error_rate=_false_pass_rate(values, bits, threshold),
    )


def predict_pass(score: float, threshold: float) -> bool:
    """Return True when ``score >= threshold``.

    A ``+inf`` threshold never passes a finite score. Ties at a finite
    threshold count as pass. Non-finite scores raise ``ValueError``.
    ``threshold`` must be finite or ``+inf``.
    """
    value = _score(score)
    cut = _predict_threshold(threshold)
    return value >= cut


def evaluate_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> ConformalEvaluation:
    """Empirical coverage and false-pass error of ``threshold`` on held-out pairs.

    Pass is ``score >= threshold``, the same rule as :func:`predict_pass`.
    """
    values, bits = _pairs(scores, labels)
    cut = _predict_threshold(threshold)
    n_fail = 0
    n_false_pass = 0
    n_false_fail = 0
    for score, label in zip(values, bits, strict=True):
        passed = score >= cut
        if label == 0:
            n_fail += 1
            if passed:
                n_false_pass += 1
        elif not passed:
            n_false_fail += 1
    if n_fail == 0:
        false_pass_rate = 0.0
        coverage = 1.0
    else:
        false_pass_rate = n_false_pass / n_fail
        coverage = 1.0 - false_pass_rate
    return ConformalEvaluation(
        n=len(values),
        n_fail=n_fail,
        n_false_pass=n_false_pass,
        n_false_fail=n_false_fail,
        false_pass_rate=false_pass_rate,
        coverage=coverage,
    )


def conformal_to_dict(result: ConformalResult) -> dict[str, object]:
    """JSON-ready conformal result. An infinite threshold is ``None``."""
    threshold: float | None
    if math.isfinite(result.threshold):
        threshold = result.threshold
    else:
        threshold = None
    return {
        "alpha": result.alpha,
        "n": result.n,
        "threshold": threshold,
        "guaranteed_bound": result.guaranteed_bound,
        "empirical_error_rate": result.empirical_error_rate,
    }


def format_conformal(result: ConformalResult) -> list[str]:
    """Human lines for a conformal block (no trailing blank)."""
    return [
        "conformal",
        f"alpha={result.alpha:.6f}",
        f"n={result.n}",
        f"threshold={_fmt_threshold(result.threshold)}",
        f"guaranteed_bound={result.guaranteed_bound:.6f}",
        f"empirical_error={result.empirical_error_rate:.6f}",
    ]


def _fail_threshold(fail_scores: list[float], alpha: float) -> float:
    n = len(fail_scores)
    rank = _split_conformal_rank(n, alpha)
    if rank > n:
        return math.inf
    ordered = sorted(fail_scores)
    return ordered[rank - 1]


def _split_conformal_rank(n: int, alpha: float) -> int:
    """1-based index ``k = ceil((n + 1) (1 - alpha))``.

    ``k == n + 1`` means the quantile is ``+inf``. ``n == 0`` returns 1.
    The ``_RANK_EPS`` shift keeps a product that overshoots an integer by a
    few ulps (``10 * 0.9`` → ``9.000000000000002``) from jumping up a rank.
    """
    target = (n + 1) * (1.0 - alpha)
    rank = math.ceil(target - _RANK_EPS)
    if rank < 1:
        return 1
    if rank > n + 1:
        return n + 1
    return rank


def _guaranteed_bound(n: int, alpha: float) -> float:
    allowed = math.floor((n + 1) * alpha + _RANK_EPS)
    return allowed / (n + 1)


def _false_pass_rate(scores: Sequence[float], labels: Sequence[int], threshold: float) -> float:
    n_fail = 0
    n_false_pass = 0
    for score, label in zip(scores, labels, strict=True):
        if label == 0:
            n_fail += 1
            if score >= threshold:
                n_false_pass += 1
    if n_fail == 0:
        return 0.0
    return n_false_pass / n_fail


def _pairs(scores: Sequence[float], labels: Sequence[int]) -> tuple[list[float], list[int]]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if len(scores) == 0:
        raise ValueError("conformal calibration requires at least one pair")
    values = [_score(score) for score in scores]
    bits = [_label(label) for label in labels]
    return values, bits


def _score(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("scores must be finite numbers in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("scores must be finite numbers in [0, 1]")
    return number


def _label(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError("labels must be 0 or 1")
    return value


def _alpha(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("alpha must be in (0, 1)")
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return number


def _predict_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("threshold must be a finite number or +inf")
    number = float(value)
    if math.isnan(number) or number == -math.inf:
        raise ValueError("threshold must be a finite number or +inf")
    return number


def _fmt_threshold(value: float) -> str:
    if not math.isfinite(value):
        return "+inf"
    return f"{value:.6f}"
