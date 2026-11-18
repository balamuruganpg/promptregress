"""Exact Dirichlet posterior of the paired accuracy difference."""

import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, replace
from functools import cache
from pathlib import Path
from typing import cast

import pytest

from helpers import run_with
from promptregress.bayes import (
    BAYES_LIMIT,
    BayesResult,
    _gauss,
    bayes_from_path,
    bayes_to_dict,
    bayesian_comparison,
    format_bayes,
    regularized_incomplete_beta,
)
from promptregress.models import Comparison
from promptregress.report import write_json
from promptregress.stats import compare_runs

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "bayes"


def _binomial_incomplete_beta(x: float, a: int, b: int) -> float:
    """``I_x(a, b)`` for positive integers, as a binomial sum."""
    trials = a + b - 1
    total = 0.0
    for index in range(a, trials + 1):
        total += math.comb(trials, index) * x**index * (1.0 - x) ** (trials - index)
    return total


def _comparison(
    baseline: list[tuple[str, float, bool]],
    candidate: list[tuple[str, float, bool]],
) -> Comparison:
    return Comparison(
        baseline=run_with(baseline, "baseline"),
        candidate=run_with(candidate, "candidate"),
        mean_diff=0.0,
        ci_low=0.0,
        ci_high=0.0,
        mcnemar_p=1.0,
        wins=0,
        losses=0,
        ties=0,
        per_case=[],
    )


def _cells(n11: int, n10: int, n01: int, n00: int) -> Comparison:
    baseline: list[tuple[str, float, bool]] = []
    candidate: list[tuple[str, float, bool]] = []
    index = 0
    for _ in range(n11):
        baseline.append((f"c{index}", 1.0, True))
        candidate.append((f"c{index}", 1.0, True))
        index += 1
    for _ in range(n10):
        baseline.append((f"c{index}", 1.0, True))
        candidate.append((f"c{index}", 0.0, False))
        index += 1
    for _ in range(n01):
        baseline.append((f"c{index}", 0.0, False))
        candidate.append((f"c{index}", 1.0, True))
        index += 1
    for _ in range(n00):
        baseline.append((f"c{index}", 0.0, False))
        candidate.append((f"c{index}", 0.0, False))
        index += 1
    return _comparison(baseline, candidate)


@cache
def _small_result() -> BayesResult:
    # n11=2, n10=3, n01=1, n00=2. Mean is (1-3)/(8+4) = -1/6.
    return bayesian_comparison(_cells(2, 3, 1, 2), delta=0.05, seed=13)


def _dirichlet_grid(
    n11: int,
    n10: int,
    n01: int,
    n00: int,
    points: list[float],
    n_grid: int = 600,
) -> tuple[float, float, dict[float, float]]:
    """Midpoint rule for the collapsed Dirichlet on ``(p10, p01, rest)``.

    The density uses ``lgamma`` only. It does not call the incomplete beta
    or the Gauss-Legendre panels.
    """
    alpha10 = 1 + n10
    alpha01 = 1 + n01
    rest = (1 + n00) + (1 + n11)
    log_c = (
        math.lgamma(alpha10 + alpha01 + rest)
        - math.lgamma(alpha10)
        - math.lgamma(alpha01)
        - math.lgamma(rest)
    )
    step = 1.0 / n_grid
    mass = 0.0
    mean = 0.0
    below = {point: 0.0 for point in points}
    for i in range(n_grid):
        u = (i + 0.5) * step
        log_u = 0.0 if alpha10 == 1 else (alpha10 - 1) * math.log(u)
        for j in range(n_grid):
            v = (j + 0.5) * step
            if u + v >= 1.0:
                continue
            rem = 1.0 - u - v
            log_v = 0.0 if alpha01 == 1 else (alpha01 - 1) * math.log(v)
            log_r = 0.0 if rest == 1 else (rest - 1) * math.log(rem)
            weight = math.exp(log_c + log_u + log_v + log_r) * step * step
            mass += weight
            diff = v - u
            mean += diff * weight
            for point in points:
                if diff < point:
                    below[point] += weight
    return mass, mean / mass, {point: below[point] / mass for point in points}


def test_incomplete_beta_matches_known_values() -> None:
    assert regularized_incomplete_beta(0.5, 4.5, 4.5) == 0.5
    assert regularized_incomplete_beta(0.25, 1.0, 1.0) == 0.25
    assert regularized_incomplete_beta(0.0, 2.0, 3.0) == 0.0
    assert regularized_incomplete_beta(-1.0, 2.0, 3.0) == 0.0
    assert regularized_incomplete_beta(1.0, 2.0, 3.0) == 1.0
    assert regularized_incomplete_beta(2.0, 2.0, 3.0) == 1.0

    # I_0.2(2, 3) = 113/625 = 0.1808, from the binomial sum.
    assert _binomial_incomplete_beta(0.2, 2, 3) == pytest.approx(113 / 625)
    assert regularized_incomplete_beta(0.2, 2, 3) == pytest.approx(113 / 625, abs=1e-12)
    assert regularized_incomplete_beta(0.2, 2, 3) == pytest.approx(0.1808, abs=1e-12)

    # I_0.5(3, 5) = 99/128. The continued fraction takes the complementary branch.
    assert _binomial_incomplete_beta(0.5, 3, 5) == pytest.approx(99 / 128)
    assert regularized_incomplete_beta(0.5, 3, 5) == pytest.approx(99 / 128, abs=1e-12)

    # Closed forms used by the special cases: I_x(a, 1) = x^a, I_x(1, b) = 1-(1-x)^b.
    assert regularized_incomplete_beta(0.3, 2, 1) == pytest.approx(0.09, abs=1e-12)
    assert regularized_incomplete_beta(0.5, 1, 5) == pytest.approx(31 / 32, abs=1e-15)
    assert regularized_incomplete_beta(0.5, 2, 4) == pytest.approx(26 / 32, abs=1e-12)

    left = regularized_incomplete_beta(0.3, 2.5, 1.5)
    right = regularized_incomplete_beta(0.7, 1.5, 2.5)
    assert left + right == pytest.approx(1.0, abs=1e-12)


def test_incomplete_beta_refuses_bad_arguments() -> None:
    with pytest.raises(ValueError, match="finite number"):
        regularized_incomplete_beta(float("nan"), 2.0, 3.0)
    with pytest.raises(ValueError, match="finite number > 0"):
        regularized_incomplete_beta(0.2, 0.0, 3.0)
    with pytest.raises(ValueError, match="finite number > 0"):
        regularized_incomplete_beta(0.2, True, 3.0)  # type: ignore[arg-type]


def test_gauss_legendre_matches_polynomials() -> None:
    assert _gauss(lambda t: 1.0, -1.0, 1.0, 16) == pytest.approx(2.0, abs=1e-12)
    assert _gauss(lambda t: t * t, -1.0, 1.0, 16) == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert _gauss(lambda t: t**4, -1.0, 1.0, 8) == pytest.approx(2.0 / 5.0, abs=1e-12)


def test_posterior_mean_and_sign_probability() -> None:
    result = _small_result()
    assert (result.n11, result.n10, result.n01, result.n00) == (2, 3, 1, 2)
    assert result.n == 8
    assert result.observed_diff == pytest.approx((1 - 3) / 8)
    assert result.posterior_mean == pytest.approx((1 - 3) / 12)
    assert result.posterior_mean == pytest.approx(-1 / 6)
    # P(d < 0) = I_{1/2}(1+n01, 1+n10) = I_{1/2}(2, 4) = 26/32.
    assert result.p_worse == pytest.approx(regularized_incomplete_beta(0.5, 2, 4))
    assert result.p_worse == pytest.approx(26 / 32, abs=1e-12)
    assert result.p_worse_than_delta < result.p_worse
    assert result.ci_low < result.posterior_mean < result.ci_high
    assert result.ci_level == 0.95
    assert result.delta == 0.05
    assert result.seed == 13
    assert BAYES_LIMIT in result.note


def test_posterior_matches_a_dirichlet_grid() -> None:
    result = _small_result()
    points = [0.0, -0.05, result.ci_low, result.ci_high]
    mass, mean, below = _dirichlet_grid(2, 3, 1, 2, points, n_grid=600)
    assert mass == pytest.approx(1.0, abs=1e-3)
    assert mean == pytest.approx(result.posterior_mean, abs=2e-3)
    assert below[0.0] == pytest.approx(result.p_worse, abs=2e-3)
    assert below[0.0] == pytest.approx(26 / 32, abs=2e-3)
    assert below[-0.05] == pytest.approx(result.p_worse_than_delta, abs=2e-3)
    assert below[result.ci_low] == pytest.approx(0.025, abs=2e-3)
    assert below[result.ci_high] == pytest.approx(0.975, abs=2e-3)


def test_cells_match_mcnemar() -> None:
    comparison = compare_runs(
        run_with(
            [("a", 1.0, True), ("b", 1.0, True), ("c", 0.0, False), ("d", 0.0, False)],
            "base",
        ),
        run_with(
            [("a", 0.0, False), ("b", 1.0, True), ("c", 1.0, True), ("d", 0.0, False)],
            "cand",
        ),
        n_boot=20,
    )
    result = bayesian_comparison(comparison)
    assert result.n10 == comparison.mcnemar_b == 1
    assert result.n01 == comparison.mcnemar_c == 1
    assert result.n11 == 1
    assert result.n00 == 1


def test_cells_follow_the_stored_passed_bit(tmp_path: Path) -> None:
    comparison = compare_runs(
        run_with([("a", 1.0, True), ("b", 1.0, True)], "base"),
        run_with([("a", 0.0, False), ("b", 1.0, True)], "cand"),
        n_boot=20,
    )
    path = tmp_path / "comparison.json"
    write_json(comparison, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    # a: candidate score 0.9 would pass a 0.5 cut; the stored bit is a fail.
    # b: baseline score 0.1 would fail a 0.5 cut; the stored bit is a pass.
    payload["candidate"]["results"][0]["score"] = 0.9
    payload["candidate"]["results"][0]["passed"] = False
    payload["baseline"]["results"][1]["score"] = 0.1
    payload["baseline"]["results"][1]["passed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = bayes_from_path(path)
    assert (result.n11, result.n10, result.n01, result.n00) == (1, 1, 0, 0)
    assert result.posterior_mean == pytest.approx(-1 / 6)
    assert result.source == str(path)


def test_concordant_pairs_stay_centered() -> None:
    result = bayesian_comparison(_cells(3, 0, 0, 1), delta=0.05, seed=13)
    assert result.observed_diff == 0.0
    assert result.posterior_mean == 0.0
    assert result.p_worse == 0.5
    assert result.ci_low == pytest.approx(-result.ci_high, abs=1e-9)
    assert result.ci_high > 0.2
    assert result.ci_low < -0.2


def test_swapping_discordant_cells_negates_d() -> None:
    left = bayesian_comparison(_cells(2, 3, 1, 2), delta=0.05, seed=1)
    right = bayesian_comparison(_cells(2, 1, 3, 2), delta=0.05, seed=99)
    assert right.posterior_mean == pytest.approx(-left.posterior_mean, abs=1e-12)
    assert right.observed_diff == pytest.approx(-left.observed_diff)
    assert left.p_worse + right.p_worse == pytest.approx(1.0, abs=1e-12)
    assert right.ci_low == pytest.approx(-left.ci_high, abs=1e-8)
    assert right.ci_high == pytest.approx(-left.ci_low, abs=1e-8)
    assert left.p_worse_than_delta != right.p_worse_than_delta


def test_seed_is_stored_and_unused() -> None:
    left = asdict(bayesian_comparison(_cells(2, 3, 1, 2), delta=0.05, seed=1))
    right = asdict(bayesian_comparison(_cells(2, 3, 1, 2), delta=0.05, seed=99))
    assert left["seed"] == 1
    assert right["seed"] == 99
    left["seed"] = 0
    right["seed"] = 0
    assert left == right


def test_delta_zero_matches_p_worse() -> None:
    result = bayesian_comparison(_cells(2, 3, 1, 2), delta=0, seed=13)
    assert result.p_worse_than_delta == result.p_worse
    assert result.delta == 0.0


def test_text_rounds_and_json_keeps_the_float() -> None:
    result = _small_result()
    text = format_bayes(result)
    assert text.endswith("\n")
    assert f"posterior_mean={result.posterior_mean:.6f}" in text
    assert f"p_worse={result.p_worse:.6f}" in text
    assert "comparison=" not in text
    payload = bayes_to_dict(result)
    assert payload["comparison"] is None
    assert payload["ci_low"] == result.ci_low
    assert payload["ci_high"] == result.ci_high
    assert payload["note"] == result.note
    assert BAYES_LIMIT in str(payload["note"])


def test_refusals() -> None:
    empty = _comparison([], [])
    with pytest.raises(ValueError, match="comparison has no cases"):
        bayesian_comparison(empty)
    mismatched = _comparison([("a", 1.0, True)], [("b", 0.0, False)])
    with pytest.raises(ValueError, match="do not match"):
        bayesian_comparison(mismatched)
    duplicated = _comparison(
        [("a", 1.0, True), ("a", 1.0, True)],
        [("a", 0.0, False), ("a", 0.0, False)],
    )
    with pytest.raises(ValueError, match="repeats case id 'a'"):
        bayesian_comparison(duplicated)

    base = run_with([("a", 1.0, True)], "baseline")
    cand = run_with([("a", 0.0, False)], "candidate")
    nan_case = replace(base.results[0], score=float("nan"))
    nan_comparison = replace(
        _comparison([("a", 1.0, True)], [("a", 0.0, False)]),
        baseline=replace(base, results=[nan_case]),
        candidate=cand,
    )
    with pytest.raises(ValueError, match="finite number"):
        bayesian_comparison(nan_comparison)
    inf_case = replace(cand.results[0], score=float("inf"))
    inf_comparison = replace(
        nan_comparison,
        baseline=base,
        candidate=replace(cand, results=[inf_case]),
    )
    with pytest.raises(ValueError, match="finite number"):
        bayesian_comparison(inf_comparison)
    flag = replace(base.results[0], passed=cast(bool, 1))
    flag_comparison = replace(
        nan_comparison,
        baseline=replace(base, results=[flag]),
        candidate=cand,
    )
    with pytest.raises(ValueError, match="passed must be a boolean"):
        bayesian_comparison(flag_comparison)

    ok = _cells(1, 0, 0, 0)
    for bad in (-0.1, 1, 1.5, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="delta"):
            bayesian_comparison(ok, delta=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        bayesian_comparison(ok, seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        bayesian_comparison(ok, seed=cast(int, 1.5))


def test_readme_quotes_the_example_command() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "promptregress",
            "bayes",
            "examples/bayes/comparison.json",
            "--delta",
            "0.05",
            "--seed",
            "13",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout == (EXAMPLE / "output.txt").read_text(encoding="utf-8")
    assert "posterior_mean=-0.400000" in completed.stdout
    assert "p_worse=0.968750" in completed.stdout
    assert "ci_high=0.024489" in completed.stdout
    assert "p_worse_than_delta=0.951522" in completed.stdout
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    spec_path = ROOT / "SPEC.md"
    spec = spec_path.read_text(encoding="utf-8") if spec_path.exists() else None
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for line in completed.stdout.splitlines():
        assert line in readme
    assert "promptregress bayes examples/bayes/comparison.json" in readme
    assert BAYES_LIMIT in completed.stdout
    assert BAYES_LIMIT in readme
    assert BAYES_LIMIT in changelog
    if spec is not None:
        assert BAYES_LIMIT in spec
        assert "Dirichlet(1, 1, 1, 1)" in spec
        assert "Gauss-Legendre" in spec
