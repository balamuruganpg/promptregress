"""Bayesian paired comparison of pass/fail cells.

The frequentist companion is McNemar in :mod:`promptregress.stats`. This
module puts a uniform Dirichlet prior on the same four cells and reports the
posterior of the accuracy difference. Nothing here is a Monte Carlo draw.
``seed`` is accepted so the CLI matches the other commands, and the math
does not read it.

Cells, from the stored ``passed`` bit on each paired case (the bit
``compare_runs`` copies onto the paired row and the bit McNemar counts):

- ``n11`` both pass
- ``n10`` baseline pass, candidate fail (McNemar's ``b``)
- ``n01`` baseline fail, candidate pass (McNemar's ``c``)
- ``n00`` both fail

``n = n11 + n10 + n01 + n00``. A case is a pass when that bit is true. This
module does not apply a second score cut.

Prior: Dirichlet(1, 1, 1, 1) on ``(p11, p10, p01, p00)``.
Posterior: Dirichlet(1 + n11, 1 + n10, 1 + n01, 1 + n00).

The quantity of interest is the accuracy difference. The plug-in on this
suite is ``(n01 - n10) / n``, which is the candidate pass rate minus the
baseline pass rate. Under the posterior the same contrast is

    d = p01 - p10

A negative ``d`` means the candidate is worse on this suite's pairs.

Let ``α01 = 1 + n01``, ``α10 = 1 + n10``, ``α11 = 1 + n11``,
``α00 = 1 + n00``, and ``S = α01 + α10 + α11 + α00 = n + 4``.
The posterior mean is

    E[d] = (α01 - α10) / S = (n01 - n10) / (n + 4)

Group the discordant cells. ``θ = p01 + p10`` and
``φ = p01 / (p01 + p10)`` are independent, with

    φ ~ Beta(α01, α10) = Beta(1 + n01, 1 + n10)
    θ ~ Beta(α01 + α10, α11 + α00) = Beta(n01 + n10 + 2, n00 + n11 + 2)

and ``d = (2φ - 1) θ``. ``θ > 0`` almost surely, so

    P(d < 0) = P(φ < 1/2) = I_{1/2}(α01, α10)

where ``I_x`` is the regularized incomplete beta function.

For ``x`` in ``(0, 1)`` let ``φ0 = (1 + x) / 2``. Then ``d < x`` for every
``φ <= φ0``, and for ``φ > φ0`` it holds when ``θ < x / (2φ - 1)``:

    P(d < x) = I_{φ0}(α01, α10)
        + ∫_{φ0}^{1} f_φ(φ) I_{x / (2φ - 1)}(α01 + α10, α00 + α11) dφ

For ``x`` in ``(-1, 0)`` only ``φ < φ0`` can contribute, and then only when
``θ`` is large enough:

    P(d < x) = ∫_{0}^{φ0} f_φ(φ) (1 - I_{x / (2φ - 1)}(α01 + α10, α00 + α11)) dφ

The survival is computed as ``I_{1 - y}(α00 + α11, α01 + α10)``. Quadrature
is Gauss-Legendre of order 64 on subintervals about one standard deviation
of ``φ`` wide, centered on the beta mode (the mean, when a parameter is 1).
A panel whose 32-point and 64-point rules disagree by more than ``1e-12``
is split. No random numbers are used. Swapping ``n01`` with ``n10`` negates
``d``; that identity is a check, not a second model.

The 95% credible interval is the equal-tailed central interval: the 0.025
and 0.975 quantiles of this CDF, by bisection. It is not a highest-posterior
density interval.

``I_x(a, b)`` is the Lentz continued fraction (Numerical Recipes
``betacf``), with the log-gamma front factor. ``x <= 0`` returns 0 and
``x >= 1`` returns 1. ``I_x(1, b) = 1 - (1 - x)^b`` and ``I_x(a, 1) = x^a``
are used when a parameter is exactly 1, and ``I_{1/2}(a, a)`` returns 1/2.
Those are the same function, not a different approximation.

The posterior is over this suite's paired outcomes under that prior. It is
not a guarantee about unseen cases, and it does not change the gate.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from promptregress.junit_xml import strip_unpaired_surrogates
from promptregress.models import CaseResult, Comparison, RunResult
from promptregress.risk import load_comparison

DEFAULT_DELTA = 0.05
DEFAULT_SEED = 13
CI_LEVEL = 0.95
_PRIOR_PER_CELL = 1
_N_CELLS = 4
_PRIOR_STRENGTH = _PRIOR_PER_CELL * _N_CELLS
_TAIL = (1.0 - CI_LEVEL) / 2.0
_QUAD_TOL = 1e-12
_BETA_ITERS = 300
_BETA_EPS = 3e-14
_BETA_TINY = 1e-30

BAYES_LIMIT = (
    "The posterior is over this suite's paired outcomes under a uniform "
    "Dirichlet prior. It is not a guarantee about unseen cases, and it does "
    "not change the gate."
)

BAYES_NOTE = (
    "The cells are counted from the stored passed bit on each paired case, "
    "the same bit McNemar uses. n11 is both pass, n10 is baseline only "
    "(McNemar b), n01 is candidate only (McNemar c), n00 is both fail. "
    "The prior is Dirichlet(1, 1, 1, 1) on those four cell probabilities. "
    "The posterior is Dirichlet(1+n11, 1+n10, 1+n01, 1+n00). "
    "d = p01 - p10, which is the candidate pass probability minus the "
    "baseline pass probability. A negative d means the candidate is worse. "
    "observed_diff is (n01 - n10) / n. posterior_mean is (n01 - n10) / (n + 4). "
    "ci_low and ci_high are the equal-tailed central quantiles at ci_level "
    "0.95 (0.025 and 0.975), not a highest-posterior-density interval. "
    "p_worse is P(d < 0). p_worse_than_delta is P(d < -delta). "
    "Conditional on the discordant total, p01 / (p01 + p10) is "
    "Beta(1+n01, 1+n10), and the discordant mass p01 + p10 is "
    "Beta(n01+n10+2, n00+n11+2), independent of the split. "
    "d = (2φ - 1) θ under that representation. P(d < 0) is the regularized "
    "incomplete beta I_{1/2}(1+n01, 1+n10). Other probabilities integrate "
    "the split's beta density against the discordant-mass beta with "
    "deterministic Gauss-Legendre quadrature. No Monte Carlo draw is used. "
    "seed is stored and is not an input to the math. "
    + BAYES_LIMIT
    + " Text rounds to 6 decimals. JSON keeps the stored floats."
)


@dataclass(frozen=True)
class BayesResult:
    """Posterior summary for one paired comparison.

    ``observed_diff`` is ``(n01 - n10) / n``. ``posterior_mean`` is
    ``E[p01 - p10]``. ``ci_low`` and ``ci_high`` are the equal-tailed
    quantiles at ``ci_level``. ``p_worse`` is ``P(d < 0)``.
    ``p_worse_than_delta`` is ``P(d < -delta)``. ``seed`` is recorded and
    was not used. ``source`` is the comparison path when one was loaded.
    """

    n: int
    n11: int
    n10: int
    n01: int
    n00: int
    observed_diff: float
    posterior_mean: float
    ci_level: float
    ci_low: float
    ci_high: float
    p_worse: float
    delta: float
    p_worse_than_delta: float
    seed: int
    source: str | None = None

    @property
    def note(self) -> str:
        return BAYES_NOTE


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)``.

    ``a`` and ``b`` must be finite and positive. A non-finite ``x`` raises
    ``ValueError``. ``x <= 0`` returns 0 and ``x >= 1`` returns 1.
    ``I_x(1, b) = 1 - (1 - x)^b``, ``I_x(a, 1) = x^a``, and
    ``I_{1/2}(a, a) = 1/2``. Every other point uses the continued fraction.
    """
    aa = _positive_param("a", a)
    bb = _positive_param("b", b)
    # A huge int is not a finite float. ``float`` raises OverflowError
    # instead of returning infinity; that is still a non-finite x.
    xx = _as_float(x, "x must be a finite number")
    if math.isnan(xx):
        raise ValueError("x must be a finite number")
    if xx <= 0.0:
        return 0.0
    if xx >= 1.0:
        return 1.0
    if aa == 1.0 and bb == 1.0:
        return xx
    if aa == bb and xx == 0.5:
        return 0.5
    if aa == 1.0:
        # I_x(1, b) = 1 - (1 - x)^b
        return -math.expm1(bb * math.log1p(-xx))
    if bb == 1.0:
        return math.exp(aa * math.log(xx))
    log_bt = (
        math.lgamma(aa + bb)
        - math.lgamma(aa)
        - math.lgamma(bb)
        + aa * math.log(xx)
        + bb * math.log1p(-xx)
    )
    if log_bt < -745.0:
        front = 0.0
    else:
        front = math.exp(log_bt)
    if xx < (aa + 1.0) / (aa + bb + 2.0):
        value = front * _betacf(aa, bb, xx) / aa
    else:
        value = 1.0 - front * _betacf(bb, aa, 1.0 - xx) / bb
    return _unit_interval(value, "regularized incomplete beta")


def bayesian_comparison(
    comparison: Comparison,
    delta: float = DEFAULT_DELTA,
    seed: int = DEFAULT_SEED,
) -> BayesResult:
    """Posterior of ``d = p01 - p10`` for one paired comparison.

    ``delta`` must be in ``[0, 1)``. An empty comparison, a comparison whose
    case ids cannot be paired, a non-boolean ``passed`` bit, or a non-finite
    score raises ``ValueError``. ``seed`` is stored and is not read.
    """
    margin = _validate_delta(delta)
    stored_seed = _validate_seed(seed)
    n11, n10, n01, n00 = _cell_counts(comparison)
    n = n11 + n10 + n01 + n00
    if n == 0:
        raise ValueError("comparison has no cases")
    observed = (n01 - n10) / n
    mean = (n01 - n10) / (n + _PRIOR_STRENGTH)
    alpha01 = _PRIOR_PER_CELL + n01
    alpha10 = _PRIOR_PER_CELL + n10
    alpha_rest = _PRIOR_PER_CELL + n00 + _PRIOR_PER_CELL + n11
    alpha_disc = alpha01 + alpha10
    p_worse = _prob_d_less_than(0.0, alpha01, alpha10, alpha_disc, alpha_rest)
    if margin == 0.0:
        p_margin = p_worse
    else:
        p_margin = _prob_d_less_than(-margin, alpha01, alpha10, alpha_disc, alpha_rest)
    ci_low = _quantile(_TAIL, alpha01, alpha10, alpha_disc, alpha_rest)
    ci_high = _quantile(1.0 - _TAIL, alpha01, alpha10, alpha_disc, alpha_rest)
    return BayesResult(
        n=n,
        n11=n11,
        n10=n10,
        n01=n01,
        n00=n00,
        observed_diff=observed,
        posterior_mean=mean,
        ci_level=CI_LEVEL,
        ci_low=ci_low,
        ci_high=ci_high,
        p_worse=p_worse,
        delta=margin,
        p_worse_than_delta=p_margin,
        seed=stored_seed,
        source=None,
    )


def bayes_from_path(
    path: str | Path,
    delta: float = DEFAULT_DELTA,
    seed: int = DEFAULT_SEED,
) -> BayesResult:
    """Load ``compare --report-json`` and summarize the posterior.

    The loader rebuilds the two runs and pairs them by case id. Cells are
    counted from those runs' ``passed`` bits. A hand-edited ``per_case``
    block that disagrees with ``results`` is not a second source.
    """
    margin = _validate_delta(delta)
    stored_seed = _validate_seed(seed)
    comparison = load_comparison(path)
    result = bayesian_comparison(comparison, delta=margin, seed=stored_seed)
    return replace(result, source=str(path))


def format_bayes(result: BayesResult) -> str:
    """Text report. Ends with a newline. Floats print with 6 decimals."""
    lines = ["bayes"]
    if result.source is not None:
        lines.append(f"comparison={strip_unpaired_surrogates(result.source)}")
    lines.extend(
        [
            f"n={result.n}",
            f"n11={result.n11}",
            f"n10={result.n10}",
            f"n01={result.n01}",
            f"n00={result.n00}",
            f"observed_diff={result.observed_diff:.6f}",
            f"posterior_mean={result.posterior_mean:.6f}",
            f"ci_level={result.ci_level:.6f}",
            f"ci_low={result.ci_low:.6f}",
            f"ci_high={result.ci_high:.6f}",
            f"p_worse={result.p_worse:.6f}",
            f"delta={result.delta:.6f}",
            f"p_worse_than_delta={result.p_worse_than_delta:.6f}",
            f"seed={result.seed}",
            f"note={result.note}",
        ]
    )
    return strip_unpaired_surrogates("\n".join(lines) + "\n")


def bayes_to_dict(result: BayesResult) -> dict[str, object]:
    """JSON-ready report. Floats are the stored values, not the 6-decimal text."""
    source: str | None
    if result.source is None:
        source = None
    else:
        source = strip_unpaired_surrogates(result.source)
    return {
        "comparison": source,
        "n": result.n,
        "n11": result.n11,
        "n10": result.n10,
        "n01": result.n01,
        "n00": result.n00,
        "observed_diff": result.observed_diff,
        "posterior_mean": result.posterior_mean,
        "ci_level": result.ci_level,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "p_worse": result.p_worse,
        "delta": result.delta,
        "p_worse_than_delta": result.p_worse_than_delta,
        "seed": result.seed,
        "note": result.note,
    }


def write_bayes_json(payload: Mapping[str, object], path: str | Path) -> None:
    """Write ``payload`` as indented JSON. Rejects NaN and Infinity."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    target.write_text(strip_unpaired_surrogates(text), encoding="utf-8")


def _prob_d_less_than(
    x: float,
    alpha01: float,
    alpha10: float,
    alpha_disc: float,
    alpha_rest: float,
) -> float:
    """``P(d < x)`` for ``d = (2φ - 1) θ`` with the independent beta pair."""
    if x <= -1.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x == 0.0:
        return regularized_incomplete_beta(0.5, alpha01, alpha10)
    phi0 = 0.5 * (1.0 + x)
    if x > 0.0:
        # φ <= φ0 implies d = (2φ - 1) θ <= 2φ - 1 <= x.
        head = regularized_incomplete_beta(phi0, alpha01, alpha10)
        if head >= 1.0:
            return 1.0

        def weighted(phi: float) -> float:
            denom = 2.0 * phi - 1.0
            if denom <= 0.0:
                mass = 1.0
            else:
                y = x / denom
                if y >= 1.0:
                    mass = 1.0
                elif y <= 0.0:
                    mass = 0.0
                else:
                    mass = regularized_incomplete_beta(y, alpha_disc, alpha_rest)
            return _beta_pdf(phi, alpha01, alpha10) * mass

        tail = _phi_integral(weighted, phi0, 1.0, alpha01, alpha10)
        return _unit_interval(head + tail, "P(d < x)")

    def weighted_tail(phi: float) -> float:
        denom = 2.0 * phi - 1.0
        if denom >= 0.0:
            mass = 0.0
        else:
            y = x / denom
            if y >= 1.0:
                mass = 0.0
            elif y <= 0.0:
                mass = 1.0
            else:
                # 1 - I_y(c, r) = I_{1-y}(r, c), which keeps a small tail accurate.
                mass = regularized_incomplete_beta(1.0 - y, alpha_rest, alpha_disc)
        return _beta_pdf(phi, alpha01, alpha10) * mass

    total = _phi_integral(weighted_tail, 0.0, phi0, alpha01, alpha10)
    return _unit_interval(total, "P(d < x)")


def _quantile(
    p: float,
    alpha01: float,
    alpha10: float,
    alpha_disc: float,
    alpha_rest: float,
) -> float:
    """Equal-tailed quantile of ``d`` by bisection on ``(-1, 1)``."""
    lo = -1.0
    hi = 1.0
    for _ in range(60):
        if hi - lo <= 1e-12:
            break
        mid = 0.5 * (lo + hi)
        if _prob_d_less_than(mid, alpha01, alpha10, alpha_disc, alpha_rest) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _phi_integral(
    func: Callable[[float], float],
    lo: float,
    hi: float,
    a: float,
    b: float,
) -> float:
    """Integrate ``func`` on ``[lo, hi]``, with panels around the beta mode."""
    if not hi > lo:
        return 0.0
    total = 0.0
    cuts = _phi_panels(lo, hi, a, b)
    for left, right in zip(cuts, cuts[1:]):
        total += _panel_integral(func, left, right, 0)
    return total


def _phi_panels(lo: float, hi: float, a: float, b: float) -> list[float]:
    """Cuts about one standard deviation apart, covering the beta mode."""
    cuts = [lo, hi]
    variance = a * b / ((a + b) ** 2 * (a + b + 1.0))
    if variance <= 0.0:
        return [lo, hi]
    if a > 1.0 and b > 1.0:
        center = (a - 1.0) / (a + b - 2.0)
    else:
        center = a / (a + b)
    sd = math.sqrt(variance)
    for step in range(-12, 13):
        point = center + step * sd
        if lo < point < hi:
            cuts.append(point)
    cuts.sort()
    ordered = [cuts[0]]
    for point in cuts[1:]:
        if point > ordered[-1]:
            ordered.append(point)
    return ordered


def _panel_integral(func: Callable[[float], float], lo: float, hi: float, depth: int) -> float:
    fine = _gauss(func, lo, hi, 64)
    if depth >= 4 or hi - lo <= _QUAD_TOL:
        return fine
    coarse = _gauss(func, lo, hi, 32)
    if abs(fine - coarse) <= _QUAD_TOL:
        return fine
    mid = 0.5 * (lo + hi)
    return _panel_integral(func, lo, mid, depth + 1) + _panel_integral(func, mid, hi, depth + 1)


def _gauss(func: Callable[[float], float], lo: float, hi: float, order: int) -> float:
    nodes, weights = _gauss_legendre(order)
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    terms = [weight * func(mid + half * node) for node, weight in zip(nodes, weights, strict=True)]
    return half * math.fsum(terms)


@lru_cache(maxsize=4)
def _gauss_legendre(n: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Nodes and weights for ``∫_{-1}^{1} f(t) dt``."""
    if n < 1:
        raise ValueError("Gauss-Legendre order must be >= 1")
    nodes = [0.0] * n
    weights = [0.0] * n
    half_count = (n + 1) // 2
    for index in range(1, half_count + 1):
        z = math.cos(math.pi * (index - 0.25) / (n + 0.5))
        for _ in range(80):
            value, deriv = _legendre(n, z)
            if deriv == 0.0 or not math.isfinite(deriv):
                break
            delta = value / deriv
            z -= delta
            if abs(delta) <= 1e-15 * max(1.0, abs(z)):
                break
        _value, deriv = _legendre(n, z)
        weight = 2.0 / ((1.0 - z * z) * deriv * deriv)
        nodes[index - 1] = -z
        nodes[n - index] = z
        weights[index - 1] = weight
        weights[n - index] = weight
    return tuple(nodes), tuple(weights)


def _legendre(n: int, z: float) -> tuple[float, float]:
    """``(P_n(z), P_n'(z))`` by Bonnet's recursion."""
    p1 = 1.0
    p2 = 0.0
    for j in range(1, n + 1):
        p3 = p2
        p2 = p1
        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
    if abs(z * z - 1.0) < 1e-30:
        return p1, 0.0
    deriv = n * (z * p1 - p2) / (z * z - 1.0)
    return p1, deriv


def _beta_pdf(x: float, a: float, b: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    log_p = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + (a - 1.0) * math.log(x)
        + (b - 1.0) * math.log1p(-x)
    )
    if log_p < -745.0:
        return 0.0
    if log_p > 700.0:
        return math.inf
    return math.exp(log_p)


def _betacf(a: float, b: float, x: float) -> float:
    """Lentz continued fraction for the incomplete beta (Numerical Recipes)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _BETA_TINY:
        d = _BETA_TINY
    d = 1.0 / d
    h = d
    for m in range(1, _BETA_ITERS + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _BETA_TINY:
            d = _BETA_TINY
        c = 1.0 + aa / c
        if abs(c) < _BETA_TINY:
            c = _BETA_TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _BETA_TINY:
            d = _BETA_TINY
        c = 1.0 + aa / c
        if abs(c) < _BETA_TINY:
            c = _BETA_TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _BETA_EPS:
            return h
    raise ValueError("incomplete beta continued fraction did not converge")


def _cell_counts(comparison: Comparison) -> tuple[int, int, int, int]:
    """Return ``(n11, n10, n01, n00)`` from paired ``passed`` bits."""
    pairs = _pairs(comparison)
    n11 = 0
    n10 = 0
    n01 = 0
    n00 = 0
    for baseline, candidate in pairs:
        if baseline.passed and candidate.passed:
            n11 += 1
        elif baseline.passed and not candidate.passed:
            n10 += 1
        elif candidate.passed and not baseline.passed:
            n01 += 1
        else:
            n00 += 1
    return n11, n10, n01, n00


def _pairs(comparison: Comparison) -> list[tuple[CaseResult, CaseResult]]:
    baseline_results = _results(comparison.baseline, "baseline")
    candidate_results = _results(comparison.candidate, "candidate")
    if len(baseline_results) == 0 and len(candidate_results) == 0:
        raise ValueError("comparison has no cases")
    baseline_ids = [_case_id(case, "baseline") for case in baseline_results]
    candidate_ids = [_case_id(case, "candidate") for case in candidate_results]
    _reject_duplicates(baseline_ids, "baseline")
    _reject_duplicates(candidate_ids, "candidate")
    if set(baseline_ids) != set(candidate_ids):
        missing_from_candidate = sorted(set(baseline_ids) - set(candidate_ids))
        missing_from_baseline = sorted(set(candidate_ids) - set(baseline_ids))
        raise ValueError(
            "baseline and candidate case ids do not match: "
            f"missing from candidate {missing_from_candidate}, "
            f"missing from baseline {missing_from_baseline}"
        )
    by_id = {case.case_id: case for case in candidate_results}
    pairs: list[tuple[CaseResult, CaseResult]] = []
    for case in baseline_results:
        other = by_id[case.case_id]
        _require_passed(case, "baseline")
        _require_passed(other, "candidate")
        _finite_score(case.score, f"case {case.case_id!r} baseline score")
        _finite_score(other.score, f"case {case.case_id!r} candidate score")
        pairs.append((case, other))
    return pairs


def _results(run: RunResult, side: str) -> list[CaseResult]:
    results = run.results
    if not isinstance(results, list):
        raise ValueError(f"{side} results must be a list")
    return results


def _case_id(case: CaseResult, side: str) -> str:
    if not isinstance(case.case_id, str) or case.case_id == "":
        raise ValueError(f"{side} case_id must be a non-empty string")
    return case.case_id


def _reject_duplicates(ids: list[str], side: str) -> None:
    seen: set[str] = set()
    for case_id in ids:
        if case_id in seen:
            raise ValueError(f"{side} run repeats case id {case_id!r}")
        seen.add(case_id)


def _require_passed(case: CaseResult, side: str) -> None:
    if not isinstance(case.passed, bool):
        raise ValueError(f"case {case.case_id!r} {side} passed must be a boolean")


def _finite_score(value: object, label: str) -> float:
    number = _as_float(value, f"{label} must be a finite number")
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _validate_delta(delta: float) -> float:
    message = "delta must be a finite number in [0, 1)"
    number = _as_float(delta, message)
    if not math.isfinite(number) or number < 0.0 or number >= 1.0:
        raise ValueError(message)
    return number


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    return seed


def _positive_param(name: str, value: float) -> float:
    message = f"{name} must be a finite number > 0"
    number = _as_float(value, message)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(message)
    return number


def _as_float(value: object, message: str) -> float:
    """``float(value)`` for a real number.

    ``float`` of an int that does not fit raises ``OverflowError`` rather
    than returning infinity. Callers treat that as ``ValueError`` so a
    non-finite score or parameter exits 2 instead of crashing.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)
    try:
        return float(value)
    except OverflowError as exc:
        raise ValueError(message) from exc


def _unit_interval(value: float, label: str) -> float:
    if math.isnan(value):
        raise ValueError(f"{label} was not finite")
    if value < 0.0:
        if value > -1e-9:
            return 0.0
        raise ValueError(f"{label} was negative")
    if value > 1.0:
        if value < 1.0 + 1e-9:
            return 1.0
        raise ValueError(f"{label} exceeded 1")
    return value
