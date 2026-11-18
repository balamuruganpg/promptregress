"""Sample size and achieved power for a paired score comparison.

The sample-size formula is the normal approximation to a two-sided
one-sample test of the mean paired difference, which is the large-sample
version of the paired t-test:

    n = ((z_{1 - alpha/2} + z_{power}) / |d_z|)^2

``d_z`` is Cohen's dz. The approximation treats the diffs as Gaussian with
known variance 1 after standardising, so it is optimistic for small ``n``,
where a t critical value is larger than ``z``. ``required_n`` returns the
ceiling of that formula (at least 1).

The simulation draws the sample mean from ``Normal(d_z, 1/n)`` and rejects
when ``|mean| / (1/sqrt(n))`` exceeds the same ``z`` critical value. It
checks the formula. It does not add t-distribution uncertainty, and it is
not a bootstrap of an observed sample. Both powers are returned so a Monte
Carlo gap stays visible instead of being rounded away.

``achieved_power`` plugs the observed Cohen's dz and the observed ``n`` into
the same normal power formula. That treats the estimate as if it were the
true effect. It is not a corrected post-hoc power. ``n < 2`` returns 0
because a variance cannot be estimated. A non-finite dz (constant non-zero
diffs) returns 1, matching a paired t-test that already rejects with
probability 1 under that degenerate model. A zero effect returns ``alpha``.
"""

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from promptregress.loading import strip_utf8_bom_text
from promptregress.stats import cohens_dz

POWER_GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
_DEFAULT_SIMS = 2000
_NOTE = (
    "Normal approximation for a two-sided paired z-test with known variance. "
    "simulated_power draws Normal(d_z, 1/n) means with the same critical value."
)


@dataclass(frozen=True)
class RequiredN:
    """Analytic sample size and a seeded normal simulation at that ``n``."""

    effect_dz: float
    alpha: float
    power: float
    n: int
    n_continuous: float
    analytic_power: float
    simulated_power: float
    n_sim: int
    seed: int

    @property
    def disagreement(self) -> float:
        """Absolute gap between the analytic power and the simulation."""
        return abs(self.analytic_power - self.simulated_power)


def required_n(
    effect_dz: float,
    alpha: float = 0.05,
    power: float = 0.8,
    *,
    n_sim: int = _DEFAULT_SIMS,
    seed: int = 13,
) -> RequiredN:
    """Smallest ``n`` whose normal-approx power reaches ``power``.

    The effect is used as ``|effect_dz|``. A zero effect has no finite ``n``.
    An infinite effect returns ``n = 1``. ``power`` must be in ``(0, 1)``
    because the power quantile is infinite at 1.
    """
    effect = _effect(effect_dz)
    alpha_value = _alpha(alpha)
    power_value = _power_level(power)
    _check_sim(n_sim, seed)
    if math.isinf(effect):
        n_continuous = 1.0
        n_required = 1
    else:
        z_alpha = _norm_ppf(1.0 - alpha_value / 2.0)
        z_power = _norm_ppf(power_value)
        n_continuous = ((z_alpha + z_power) / effect) ** 2
        n_required = max(1, math.ceil(n_continuous - 1e-9))
    analytic = analytic_power(effect if math.isfinite(effect) else 1e6, n_required, alpha_value)
    if math.isinf(effect):
        analytic = 1.0
    simulated = simulate_power(
        effect if math.isfinite(effect) else 1e6,
        n_required,
        alpha_value,
        n_sim=n_sim,
        seed=seed,
    )
    return RequiredN(
        effect_dz=effect,
        alpha=alpha_value,
        power=power_value,
        n=n_required,
        n_continuous=n_continuous,
        analytic_power=analytic,
        simulated_power=simulated,
        n_sim=n_sim,
        seed=seed,
    )


def analytic_power(effect_dz: float, n: int, alpha: float = 0.05) -> float:
    """Two-sided normal power at a known ``|d_z|`` and sample size ``n``.

    A zero effect returns ``alpha``. An infinite effect returns 1.
    """
    count = _count(n)
    alpha_value = _alpha(alpha)
    if isinstance(effect_dz, bool) or not isinstance(effect_dz, (int, float)):
        raise ValueError("effect_dz must be a number")
    effect = abs(float(effect_dz))
    if math.isnan(effect):
        raise ValueError("effect_dz must be a number")
    if effect == 0.0:
        return alpha_value
    if math.isinf(effect):
        return 1.0
    z_critical = _norm_ppf(1.0 - alpha_value / 2.0)
    noncentrality = effect * math.sqrt(count)
    lower = _norm_cdf(-z_critical - noncentrality)
    upper = _norm_cdf(-z_critical + noncentrality)
    return min(1.0, max(0.0, lower + upper))


def simulate_power(
    effect_dz: float,
    n: int,
    alpha: float = 0.05,
    *,
    n_sim: int = _DEFAULT_SIMS,
    seed: int = 13,
) -> float:
    """Fraction of seeded normal draws that reject at ``alpha``.

    Each draw is one sample mean from ``Normal(effect_dz, 1/n)``. The test
    uses the known standard error ``1/sqrt(n)``, matching :func:`required_n`.
    """
    effect = _effect(effect_dz)
    count = _count(n)
    alpha_value = _alpha(alpha)
    _check_sim(n_sim, seed)
    if math.isinf(effect):
        effect = 1e6
    z_critical = _norm_ppf(1.0 - alpha_value / 2.0)
    standard_error = 1.0 / math.sqrt(count)
    rng = random.Random(seed)
    rejects = 0
    for _ in range(n_sim):
        mean = rng.gauss(effect, standard_error)
        if abs(mean / standard_error) >= z_critical:
            rejects += 1
    return rejects / n_sim


def achieved_power(diffs: Sequence[float], alpha: float = 0.05) -> float:
    """Plug-in normal power of a two-sided paired test on the observed diffs.

    Uses :func:`promptregress.stats.cohens_dz` as the effect and ``len(diffs)``
    as ``n``. See the module docstring for the optimism of that plug-in.
    """
    alpha_value = _alpha(alpha)
    values = [_finite(value) for value in diffs]
    n = len(values)
    if n < 2:
        return 0.0
    effect = abs(cohens_dz(values))
    if not math.isfinite(effect):
        return 1.0
    return analytic_power(effect, n, alpha_value)


def power_grid(
    effect_dz: float,
    alpha: float = 0.05,
    *,
    powers: Sequence[float] = POWER_GRID,
    n_sim: int = _DEFAULT_SIMS,
    seed: int = 13,
) -> list[RequiredN]:
    """``required_n`` at each target power. The seed is reused per row.

    Reusing the seed makes each row reproducible on its own. Rows are not
    one shared Monte Carlo stream.
    """
    return [required_n(effect_dz, alpha, power, n_sim=n_sim, seed=seed) for power in powers]


def load_report_diffs(path: str | Path) -> list[float]:
    """Read per-case ``diff`` values from a comparison JSON report."""
    target = Path(path)
    try:
        # utf-8-sig (+ residual / per-line BOM strip) so a Windows-saved
        # compare report loads for ``power --from-report``; json.loads
        # rejects a leading or mid-file line-leading U+FEFF.
        text = strip_utf8_bom_text(target.read_text(encoding="utf-8-sig"))
    except UnicodeError as exc:
        raise ValueError(f"{target}: not valid UTF-8: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{target}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("per_case"), list):
        raise ValueError(f"{target}: expected a comparison report with per_case")
    diffs: list[float] = []
    for index, row in enumerate(payload["per_case"]):
        if not isinstance(row, dict) or "diff" not in row:
            raise ValueError(f"{target}: per_case[{index}] needs a diff")
        raw = row["diff"]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{target}: per_case[{index}].diff must be a number")
        diffs.append(_finite(raw))
    if not diffs:
        raise ValueError(f"{target}: comparison report has no per-case diffs")
    return diffs


def required_n_to_dict(estimate: RequiredN) -> dict[str, object]:
    """JSON-ready sample-size row."""
    return {
        "effect_dz": _json_number(estimate.effect_dz),
        "alpha": estimate.alpha,
        "power": estimate.power,
        "n": estimate.n,
        "n_continuous": estimate.n_continuous,
        "analytic_power": estimate.analytic_power,
        "simulated_power": estimate.simulated_power,
        "disagreement": estimate.disagreement,
        "n_sim": estimate.n_sim,
        "seed": estimate.seed,
    }


def power_to_dict(
    estimates: Sequence[RequiredN],
    *,
    achieved: float | None = None,
    observed_dz: float | None = None,
) -> dict[str, object]:
    """JSON-ready power report."""
    return {
        "note": _NOTE,
        "observed_dz": None if observed_dz is None else _json_number(observed_dz),
        "achieved_power": achieved,
        "grid": [required_n_to_dict(estimate) for estimate in estimates],
    }


def format_power(
    estimates: Sequence[RequiredN],
    *,
    achieved: float | None = None,
    observed_dz: float | None = None,
) -> str:
    """Table of required ``n`` plus optional achieved power, then JSON."""
    lines = ["power", _NOTE]
    if observed_dz is not None:
        lines.append(f"observed_dz={_fmt_stat(observed_dz)}")
    if achieved is not None:
        lines.append(f"achieved_power={achieved:.6f}")
    if estimates:
        lines.append(f"effect_dz={_fmt_stat(estimates[0].effect_dz)}")
        lines.append(f"alpha={estimates[0].alpha:.6f}")
        lines.extend(
            [
                "",
                "| power | n | n_continuous | analytic_power | simulated_power |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for estimate in estimates:
            lines.append(
                f"| {estimate.power:.2f} | {estimate.n} | {estimate.n_continuous:.6f} | "
                f"{estimate.analytic_power:.6f} | {estimate.simulated_power:.6f} |"
            )
    else:
        lines.append("required_n=undefined")
    lines.extend(
        [
            "",
            json.dumps(
                power_to_dict(estimates, achieved=achieved, observed_dz=observed_dz),
                indent=2,
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _norm_ppf(p: float) -> float:
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    low = -20.0
    high = 20.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if _norm_cdf(mid) < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _effect(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("effect_dz must be a number")
    number = float(value)
    if math.isnan(number) or number == 0.0:
        raise ValueError("effect_dz must be a non-zero finite or infinite number")
    return abs(number)


def _alpha(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("alpha must be in (0, 1)")
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return number


def _power_level(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("power must be in (0, 1)")
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError("power must be in (0, 1)")
    return number


def _count(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be an integer >= 1")
    return n


def _check_sim(n_sim: int, seed: int) -> None:
    if isinstance(n_sim, bool) or not isinstance(n_sim, int) or n_sim < 1:
        raise ValueError("n_sim must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")


def _finite(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("diffs must be finite")
    return number


def _json_number(value: float) -> float | None:
    if math.isfinite(value):
        return value
    return None


def _fmt_stat(value: float) -> str:
    if not math.isfinite(value):
        if value > 0:
            return "inf"
        if value < 0:
            return "-inf"
        return "nan"
    return f"{value:.6f}"
