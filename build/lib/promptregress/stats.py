"""Paired bootstrap intervals, hypothesis tests, and run comparison."""

import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence

from promptregress.models import (
    CaseResult,
    Comparison,
    Outcome,
    PairedRow,
    RunResult,
    SliceSummary,
)

_TIE_EPS = 1e-12
_WILCOXON_EXACT_MAX_N = 20
_CI_METHODS = {"percentile", "bca"}
_EFFECT_SMALL = 0.2
_EFFECT_MEDIUM = 0.5
_EFFECT_LARGE = 0.8


class StatsError(ValueError):
    """Baseline and candidate runs cannot be paired."""


def paired_bootstrap_ci(
    diffs: Sequence[float],
    n_boot: int = 2000,
    seed: int = 13,
    alpha: float = 0.05,
    *,
    method: str = "percentile",
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean of ``diffs``.

    Resamples case-level differences with replacement using ``random.Random(seed)``
    and returns the ``alpha/2`` and ``1 - alpha/2`` interval endpoints.

    ``method='percentile'`` (default) uses linear interpolation at rank
    ``p/100 * (n_boot - 1)``, the same rule as NumPy's legacy linear
    percentile. ``method='bca'`` is the bias-corrected and accelerated
    interval with a jackknife acceleration estimate. Constant diffs, ``n < 3``,
    and a degenerate jackknife or bias correction fall back to the percentile
    interval. An empty ``diffs`` returns ``(0.0, 0.0)``.
    """
    low, high, _notes = _mean_confidence_interval(
        diffs, n_boot=n_boot, seed=seed, alpha=alpha, method=method
    )
    return (low, high)


def wilcoxon_signed_rank(diffs: Sequence[float]) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Returns ``(W, p)``. ``W`` is the sum of the ranks of the positive
    differences after zeros are dropped. Ties in absolute value take midranks.

    For ``n <= 20`` non-zero differences the p-value is exact: the null
    distribution of the rank sum is enumerated by dynamic programming over
    subsets of the ranks. For larger ``n`` the p-value is a normal
    approximation with continuity correction and the usual tie correction
    ``sum(t^3 - t) / 48``. All-zero input returns ``(0.0, 1.0)``.
    """
    values = _finite_values(diffs, "diffs")
    nonzero = [item for item in values if abs(item) > _TIE_EPS]
    n = len(nonzero)
    if n == 0:
        return (0.0, 1.0)
    ranks = _signed_rank_values(nonzero)
    w_plus = sum(rank for rank, value in zip(ranks, nonzero, strict=True) if value > 0)
    if n <= _WILCOXON_EXACT_MAX_N:
        p_value = _wilcoxon_exact_p(ranks, w_plus)
    else:
        p_value = _wilcoxon_normal_p(ranks, w_plus, nonzero)
    return (w_plus, p_value)


def paired_ttest(diffs: Sequence[float]) -> tuple[float, float, float]:
    """Two-sided paired t-test on ``diffs``.

    Returns ``(t, df, p)``. ``p`` is the two-sided tail of Student's t with
    ``n - 1`` degrees of freedom, computed from the regularised incomplete
    beta function. ``n < 2`` returns ``(0.0, 0.0, 1.0)``. Zero-variance
    diffs that are identically zero return ``t = 0``, ``p = 1``; non-zero
    constant diffs return an infinite ``t`` and ``p = 0``.
    """
    values = _finite_values(diffs, "diffs")
    n = len(values)
    if n < 2:
        return (0.0, 0.0, 1.0)
    df = float(n - 1)
    mean = sum(values) / n
    sum_sq = sum((item - mean) ** 2 for item in values)
    variance = sum_sq / df
    if variance <= 0.0:
        if abs(mean) <= _TIE_EPS:
            return (0.0, df, 1.0)
        t_stat = math.copysign(math.inf, mean)
        return (t_stat, df, 0.0)
    se = math.sqrt(variance / n)
    t_stat = mean / se
    p_value = _student_t_two_sided_p(t_stat, df)
    return (t_stat, df, p_value)


def cohens_dz(diffs: Sequence[float]) -> float:
    """Cohen's ``d_z`` for paired diffs: mean divided by the sample standard deviation.

    ``n < 2`` returns ``0.0``. Zero variance with a zero mean returns ``0.0``;
    a non-zero constant mean returns signed infinity.
    """
    values = _finite_values(diffs, "diffs")
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    sum_sq = sum((item - mean) ** 2 for item in values)
    variance = sum_sq / (n - 1)
    if variance <= 0.0:
        if abs(mean) <= _TIE_EPS:
            return 0.0
        return math.copysign(math.inf, mean)
    return mean / math.sqrt(variance)


def effect_size_label(dz: float) -> str:
    """Plain-language magnitude for |d_z| using Cohen's 0.2 / 0.5 / 0.8 cuts.

    ``|d| < 0.2`` is negligible, ``< 0.5`` small, ``< 0.8`` medium, otherwise
    large. Non-finite values (zero-variance non-zero diffs) are large.
    """
    if not math.isfinite(dz):
        return "large"
    magnitude = abs(dz)
    if magnitude < _EFFECT_SMALL:
        return "negligible"
    if magnitude < _EFFECT_MEDIUM:
        return "small"
    if magnitude < _EFFECT_LARGE:
        return "medium"
    return "large"


def mcnemar_p_value(b: int, c: int) -> float:
    """Two-sided p-value for McNemar's test.

    ``b`` is the number of cases the baseline passed and the candidate failed.
    ``c`` is the number the baseline failed and the candidate passed.

    When ``b + c < 25`` the p-value is the exact two-sided binomial test of
    ``X ~ Binomial(b + c, 0.5)``: ``min(1, 2 * P(X <= min(b, c)))``. When
    ``b == c`` the p-value is 1.

    Otherwise the p-value is the chi-square test with continuity correction,

        chi2 = (|b - c| - 1)^2 / (b + c)

    and survival ``math.erfc(sqrt(chi2 / 2))`` for chi-square with 1 degree
    of freedom.
    """
    if (
        isinstance(b, bool)
        or isinstance(c, bool)
        or not isinstance(b, int)
        or not isinstance(c, int)
    ):
        raise TypeError("b and c must be integers")
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")
    if b == c:
        return 1.0
    n = b + c
    if n < 25:
        tail = min(b, c)
        total = sum(math.comb(n, index) for index in range(tail + 1))
        return min(1.0, 2.0 * total / float(2**n))
    stat = (abs(b - c) - 1) ** 2 / n
    return math.erfc(math.sqrt(stat / 2.0))


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    n_boot: int = 2000,
    seed: int = 13,
    alpha: float = 0.05,
    *,
    ci_method: str = "percentile",
) -> Comparison:
    """Pair cases by id and summarize the candidate-minus-baseline score diff.

    Raises ``StatsError`` when either run repeats a case id or the two id
    sets differ. Per-case rows follow baseline order. A score difference
    within ``1e-12`` is a tie and is stored as ``0.0``. Wins are higher
    candidate scores. McNemar's test uses pass/fail flips, not score ties.

    Wilcoxon, the paired t-test, and Cohen's ``d_z`` are computed on the
    same diffs and stored on the result. They do not change the gate, which
    still reads the bootstrap interval. ``ci_method`` is ``'percentile'`` or
    ``'bca'``. Cases that carry tags produce one :class:`SliceSummary` per
    distinct tag.
    """
    if ci_method not in _CI_METHODS:
        raise ValueError("ci_method must be 'percentile' or 'bca'")
    baseline_map = _index_results(baseline, "baseline")
    candidate_map = _index_results(candidate, "candidate")
    if set(baseline_map) != set(candidate_map):
        missing_from_candidate = sorted(set(baseline_map) - set(candidate_map))
        missing_from_baseline = sorted(set(candidate_map) - set(baseline_map))
        raise StatsError(
            "Baseline and candidate case ids do not match: "
            f"missing from candidate {missing_from_candidate}, "
            f"missing from baseline {missing_from_baseline}"
        )
    rows: list[PairedRow] = []
    for result in baseline.results:
        other = candidate_map[result.case_id]
        raw_diff = other.score - result.score
        if abs(raw_diff) <= _TIE_EPS:
            diff = 0.0
            outcome: Outcome = "tie"
        elif raw_diff > 0:
            diff = raw_diff
            outcome = "win"
        else:
            diff = raw_diff
            outcome = "loss"
        tags = _row_tags(result, other)
        rows.append(
            PairedRow(
                case_id=result.case_id,
                baseline_output=result.output,
                candidate_output=other.output,
                baseline_score=result.score,
                candidate_score=other.score,
                diff=diff,
                baseline_passed=result.passed,
                candidate_passed=other.passed,
                outcome=outcome,
                tags=tags,
            )
        )
    diffs = [row.diff for row in rows]
    if rows:
        mean_diff = sum(diffs) / len(rows)
    else:
        mean_diff = 0.0
    ci_low, ci_high, details = _mean_confidence_interval(
        diffs, n_boot=n_boot, seed=seed, alpha=alpha, method=ci_method
    )
    discordant_b = sum(1 for row in rows if row.baseline_passed and not row.candidate_passed)
    discordant_c = sum(1 for row in rows if row.candidate_passed and not row.baseline_passed)
    wins, losses, ties = _outcome_counts(rows)
    wilcoxon_stat, wilcoxon_p = wilcoxon_signed_rank(diffs)
    ttest_stat, ttest_df, ttest_p = paired_ttest(diffs)
    dz = cohens_dz(diffs)
    slices = _slice_summaries(rows, n_boot=n_boot, seed=seed, alpha=alpha, method=ci_method)
    return Comparison(
        baseline=baseline,
        candidate=candidate,
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        mcnemar_p=mcnemar_p_value(discordant_b, discordant_c),
        wins=wins,
        losses=losses,
        ties=ties,
        per_case=rows,
        mcnemar_b=discordant_b,
        mcnemar_c=discordant_c,
        alpha=float(alpha),
        ci_method=ci_method,
        wilcoxon_stat=wilcoxon_stat,
        wilcoxon_p=wilcoxon_p,
        ttest_stat=ttest_stat,
        ttest_df=ttest_df,
        ttest_p=ttest_p,
        cohens_dz=dz,
        effect_size_label=effect_size_label(dz),
        slices=slices,
        details=details,
    )


def _mean_confidence_interval(
    diffs: Sequence[float],
    n_boot: int,
    seed: int,
    alpha: float,
    method: str,
) -> tuple[float, float, list[str]]:
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    alpha_value = float(alpha)
    if not 0.0 <= alpha_value < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    if method not in _CI_METHODS:
        raise ValueError("method must be 'percentile' or 'bca'")
    values = _finite_values(diffs, "diffs")
    if not values:
        return (0.0, 0.0, [])
    means = _bootstrap_means(values, n_boot=n_boot, seed=seed)
    means.sort()
    low_pct = 100.0 * (alpha_value / 2.0)
    high_pct = 100.0 * (1.0 - alpha_value / 2.0)
    percentile = (_percentile(means, low_pct), _percentile(means, high_pct))
    if method == "percentile":
        return (percentile[0], percentile[1], [])
    bca = _bca_interval(values, means, alpha_value)
    if bca is None:
        return (percentile[0], percentile[1], [_bca_fallback_note(values, means)])
    return (bca[0], bca[1], [])


def _bootstrap_means(values: list[float], n_boot: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    width = len(values)
    means = [0.0] * n_boot
    for boot_index in range(n_boot):
        total = 0.0
        for _ in range(width):
            total += values[rng.randrange(width)]
        means[boot_index] = total / width
    return means


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile. ``pct`` is in ``[0, 100]``."""
    if not sorted_values:
        raise ValueError("percentile of an empty sample")
    count = len(sorted_values)
    if count == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (count - 1)
    low_index = math.floor(rank)
    high_index = math.ceil(rank)
    if low_index == high_index:
        return sorted_values[low_index]
    weight = rank - low_index
    return sorted_values[low_index] * (1.0 - weight) + sorted_values[high_index] * weight


def _bca_interval(
    values: list[float],
    sorted_means: list[float],
    alpha: float,
) -> tuple[float, float] | None:
    n = len(values)
    if n < 3:
        return None
    if max(values) - min(values) <= _TIE_EPS:
        return None
    acceleration = _jackknife_acceleration(values)
    if acceleration is None:
        return None
    hat = sum(values) / n
    proportion = _bias_correction_proportion(sorted_means, hat)
    if proportion <= 0.0 or proportion >= 1.0:
        return None
    z0 = _norm_ppf(proportion)
    z_lo = _norm_ppf(alpha / 2.0)
    z_hi = _norm_ppf(1.0 - alpha / 2.0)
    a1 = _bca_adjusted_alpha(z0, acceleration, z_lo)
    a2 = _bca_adjusted_alpha(z0, acceleration, z_hi)
    if a1 is None or a2 is None:
        return None
    a1 = min(1.0, max(0.0, a1))
    a2 = min(1.0, max(0.0, a2))
    if a1 > a2:
        return None
    low = _percentile(sorted_means, 100.0 * a1)
    high = _percentile(sorted_means, 100.0 * a2)
    if low > high:
        return None
    return (low, high)


def _jackknife_acceleration(values: list[float]) -> float | None:
    n = len(values)
    if n < 3:
        return None
    total = sum(values)
    jack = [(total - values[index]) / (n - 1) for index in range(n)]
    jack_mean = sum(jack) / n
    deviations = [jack_mean - item for item in jack]
    sum_sq = sum(item * item for item in deviations)
    if sum_sq <= 0.0:
        return None
    sum_cubed = sum(item * item * item for item in deviations)
    return sum_cubed / (6.0 * math.pow(sum_sq, 1.5))


def _bca_adjusted_alpha(z0: float, acceleration: float, z_crit: float) -> float | None:
    numer = z0 + z_crit
    denom = 1.0 - acceleration * numer
    if denom == 0.0 or not math.isfinite(denom):
        return None
    adjusted = _norm_cdf(z0 + numer / denom)
    if not math.isfinite(adjusted):
        return None
    return adjusted


def _bca_fallback_note(values: list[float], sorted_means: list[float]) -> str:
    n = len(values)
    if n < 3:
        return "BCa fell back to percentile: n<3."
    if max(values) - min(values) <= _TIE_EPS:
        return "BCa fell back to percentile: constant diffs."
    if _jackknife_acceleration(values) is None:
        return "BCa fell back to percentile: degenerate jackknife."
    hat = sum(values) / n
    proportion = _bias_correction_proportion(sorted_means, hat)
    if proportion <= 0.0 or proportion >= 1.0:
        return "BCa fell back to percentile: degenerate bias correction."
    return "BCa fell back to percentile: degenerate acceleration mapping."


def _bias_correction_proportion(sorted_means: list[float], hat: float) -> float:
    """Share of bootstrap means below ``hat``, with a half-count on ties.

    A raw ``#{θ* < θ̂} / B`` is pulled away from 1/2 when many replicates land
    on the sample mean, which is common for small discrete score diffs. Counting
    ties as half keeps z0 near 0 on symmetric data.
    """
    n_less = 0
    n_equal = 0
    for mean in sorted_means:
        delta = mean - hat
        if delta < -_TIE_EPS:
            n_less += 1
        elif abs(delta) <= _TIE_EPS:
            n_equal += 1
    return (n_less + 0.5 * n_equal) / len(sorted_means)


def _signed_rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: abs(values[index]))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        magnitude = abs(values[order[index]])
        while end < len(order) and abs(values[order[end]]) == magnitude:
            end += 1
        midrank = 0.5 * ((index + 1) + end)
        for position in range(index, end):
            ranks[order[position]] = midrank
        index = end
    return ranks


def _wilcoxon_exact_p(ranks: list[float], w_plus: float) -> float:
    scaled_ranks = [int(round(rank * 2.0)) for rank in ranks]
    observed = int(round(w_plus * 2.0))
    total = sum(scaled_ranks)
    ways = [0] * (total + 1)
    ways[0] = 1
    running = 0
    for rank in scaled_ranks:
        running += rank
        for score in range(running, rank - 1, -1):
            ways[score] += ways[score - rank]
    assignments = 1 << len(ranks)
    mu = total / 2.0
    deviation = abs(observed - mu)
    extreme = 0
    for score, count in enumerate(ways):
        if abs(score - mu) + 1e-12 >= deviation:
            extreme += count
    return min(1.0, extreme / assignments)


def _wilcoxon_normal_p(ranks: list[float], w_plus: float, values: list[float]) -> float:
    n = len(values)
    mu = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    for tie_size in Counter(abs(item) for item in values).values():
        if tie_size > 1:
            variance -= (tie_size**3 - tie_size) / 48.0
    if variance <= 0.0:
        return 1.0 if abs(w_plus - mu) <= _TIE_EPS else 0.0
    z = (abs(w_plus - mu) - 0.5) / math.sqrt(variance)
    if z <= 0.0:
        return 1.0
    return min(1.0, math.erfc(z / math.sqrt(2.0)))


def _student_t_two_sided_p(t_stat: float, df: float) -> float:
    if df <= 0.0:
        return 1.0
    if not math.isfinite(t_stat):
        return 0.0
    if t_stat == 0.0:
        return 1.0
    x = df / (df + t_stat * t_stat)
    return _regularised_incomplete_beta(x, df / 2.0, 0.5)


def _regularised_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta I_x(a, b) via a continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if a <= 0.0 or b <= 0.0:
        raise ValueError("incomplete beta parameters must be positive")
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, max_iter: int = 200, epsilon: float = 3e-14) -> float:
    """Lentz continued fraction for the incomplete beta (Numerical Recipes)."""
    tiny = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            return h
    return h


def _norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _norm_ppf(p: float) -> float:
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    lo = -20.0
    hi = 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _finite_values(diffs: Sequence[float], label: str) -> list[float]:
    values = [float(item) for item in diffs]
    if any(math.isnan(item) or math.isinf(item) for item in values):
        raise ValueError(f"{label} must be finite")
    return values


def _index_results(run: RunResult, label: str) -> dict[str, CaseResult]:
    found: dict[str, CaseResult] = {}
    for result in run.results:
        if result.case_id in found:
            raise StatsError(f"Duplicate case id in {label} results: {result.case_id}")
        found[result.case_id] = result
    return found


def _row_tags(baseline: CaseResult, candidate: CaseResult) -> list[str]:
    source = baseline.tags if baseline.tags else candidate.tags
    seen: set[str] = set()
    tags: list[str] = []
    for tag in source:
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def _outcome_counts(rows: Sequence[PairedRow]) -> tuple[int, int, int]:
    wins = sum(1 for row in rows if row.outcome == "win")
    losses = sum(1 for row in rows if row.outcome == "loss")
    ties = sum(1 for row in rows if row.outcome == "tie")
    return wins, losses, ties


def _slice_summaries(
    rows: Sequence[PairedRow],
    n_boot: int,
    seed: int,
    alpha: float,
    method: str,
) -> list[SliceSummary]:
    grouped: dict[str, list[PairedRow]] = defaultdict(list)
    for row in rows:
        for tag in row.tags:
            grouped[tag].append(row)
    slices: list[SliceSummary] = []
    for tag in sorted(grouped):
        group = grouped[tag]
        diffs = [row.diff for row in group]
        mean_diff = sum(diffs) / len(group)
        ci_low, ci_high, _notes = _mean_confidence_interval(
            diffs, n_boot=n_boot, seed=seed, alpha=alpha, method=method
        )
        wins, losses, ties = _outcome_counts(group)
        slices.append(
            SliceSummary(
                tag=tag,
                n=len(group),
                mean_diff=mean_diff,
                ci_low=ci_low,
                ci_high=ci_high,
                wins=wins,
                losses=losses,
                ties=ties,
            )
        )
    return slices
