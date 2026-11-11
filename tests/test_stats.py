"""Bootstrap intervals, McNemar's test, and paired run comparison."""

import math
import random

import pytest

from helpers import case_result, run_with
from promptregress.models import RunResult
from promptregress.stats import (
    StatsError,
    cohens_dz,
    compare_runs,
    effect_size_label,
    mcnemar_p_value,
    paired_bootstrap_ci,
    paired_ttest,
    wilcoxon_signed_rank,
)


def test_bootstrap_identical_diffs_are_a_point_at_zero() -> None:
    assert paired_bootstrap_ci([0.0, 0.0, 0.0, 0.0]) == (0.0, 0.0)


def test_bootstrap_is_reproducible_for_a_fixed_seed() -> None:
    diffs = [0.2, -0.4, 0.0, 0.1, -0.1, 0.3]
    first = paired_bootstrap_ci(diffs, n_boot=300, seed=13)
    second = paired_bootstrap_ci(diffs, n_boot=300, seed=13)
    assert first == second
    assert first[0] <= first[1]


def test_bootstrap_matches_manual_percentile() -> None:
    diffs = [1.0, 2.0, 3.0]
    n_boot = 100
    seed = 13
    alpha = 0.05
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(len(diffs)):
            total += diffs[rng.randrange(len(diffs))]
        means.append(total / len(diffs))
    means.sort()

    def percentile(pct: float) -> float:
        rank = (pct / 100.0) * (len(means) - 1)
        low = math.floor(rank)
        high = math.ceil(rank)
        if low == high:
            return means[low]
        weight = rank - low
        return means[low] * (1.0 - weight) + means[high] * weight

    low, high = paired_bootstrap_ci(diffs, n_boot=n_boot, seed=seed, alpha=alpha)
    assert low == percentile(2.5)
    assert high == percentile(97.5)


def test_bootstrap_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="n_boot"):
        paired_bootstrap_ci([0.0], n_boot=0)
    with pytest.raises(ValueError, match="alpha"):
        paired_bootstrap_ci([0.0], alpha=1.0)
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap_ci([math.nan])


def test_mcnemar_equal_counts_are_one() -> None:
    assert mcnemar_p_value(0, 0) == 1.0
    assert mcnemar_p_value(4, 4) == 1.0
    assert mcnemar_p_value(30, 30) == 1.0


def test_mcnemar_exact_known_value() -> None:
    # 2 * (C(10, 0) + C(10, 1)) / 1024 = 22/1024.
    assert mcnemar_p_value(9, 1) == pytest.approx(22 / 1024)
    assert mcnemar_p_value(1, 9) == pytest.approx(22 / 1024)
    assert mcnemar_p_value(9, 1) == pytest.approx(0.0215, abs=1e-4)
    assert mcnemar_p_value(3, 0) == pytest.approx(0.25)
    assert mcnemar_p_value(24, 0) == pytest.approx(2 / 2**24)


def test_mcnemar_chi_square_with_continuity_correction() -> None:
    b, c = 20, 5
    n = b + c
    assert n >= 25
    stat = (abs(b - c) - 1) ** 2 / n
    expected = math.erfc(math.sqrt(stat / 2.0))
    assert mcnemar_p_value(b, c) == pytest.approx(expected)
    assert mcnemar_p_value(c, b) == pytest.approx(expected)


def test_mcnemar_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mcnemar_p_value(-1, 0)


def test_identical_runs_have_zero_diff_and_ci_contains_zero() -> None:
    run = run_with([("a", 1.0, True), ("b", 0.0, False)], "cfg")
    comparison = compare_runs(run, run)
    assert comparison.mean_diff == 0.0
    assert comparison.ci_low <= 0.0 <= comparison.ci_high
    assert comparison.wins == 0
    assert comparison.losses == 0
    assert comparison.ties == 2
    assert comparison.mcnemar_p == 1.0
    assert comparison.mcnemar_b == 0
    assert comparison.mcnemar_c == 0


def test_compare_runs_matches_bootstrap_and_order() -> None:
    baseline = run_with([("a", 1.0, True), ("b", 0.0, False)], "baseline")
    candidate = run_with([("b", 1.0, True), ("a", 0.0, False)], "candidate")
    comparison = compare_runs(baseline, candidate, n_boot=50, seed=7, alpha=0.1)
    assert [row.case_id for row in comparison.per_case] == ["a", "b"]
    assert comparison.per_case[0].diff == -1.0
    assert comparison.per_case[0].outcome == "loss"
    assert comparison.per_case[1].diff == 1.0
    assert comparison.per_case[1].outcome == "win"
    assert comparison.wins == 1
    assert comparison.losses == 1
    assert comparison.ties == 0
    assert comparison.mean_diff == 0.0
    assert comparison.mcnemar_b == 1
    assert comparison.mcnemar_c == 1
    assert comparison.mcnemar_p == 1.0
    diffs = [row.diff for row in comparison.per_case]
    assert (comparison.ci_low, comparison.ci_high) == paired_bootstrap_ci(
        diffs, n_boot=50, seed=7, alpha=0.1
    )
    assert comparison.alpha == pytest.approx(0.1)


def test_compare_runs_rejects_mismatched_ids() -> None:
    left = run_with([("a", 1.0, True), ("b", 1.0, True)], "baseline")
    right = run_with([("a", 1.0, True), ("c", 1.0, True)], "candidate")
    with pytest.raises(StatsError, match="do not match"):
        compare_runs(left, right)


def test_wilcoxon_hand_computable_example() -> None:
    # diffs = [-1, 2, 3, 4, 5]. Zeros none. Absolute ranks 1,2,3,4,5.
    # W+ = 2+3+4+5 = 14. Under H0 each of 2^5 = 32 sign assignments is equal.
    # Two-sided: W+ <= 1 or W+ >= 14, which is 4 assignments, so p = 4/32.
    diffs = [-1.0, 2.0, 3.0, 4.0, 5.0]
    statistic, p_value = wilcoxon_signed_rank(diffs)
    assert statistic == 14.0
    assert p_value == pytest.approx(0.125)


def test_wilcoxon_all_zero_is_one() -> None:
    assert wilcoxon_signed_rank([0.0, 0.0, 0.0]) == (0.0, 1.0)


def test_wilcoxon_drops_zeros_then_is_exact() -> None:
    # Same as [1, 2, 3]: W+ = 6, n=3, p = 2/8 = 0.25.
    statistic, p_value = wilcoxon_signed_rank([0.0, 1.0, 2.0, 3.0])
    assert statistic == 6.0
    assert p_value == pytest.approx(0.25)


def test_wilcoxon_normal_approximation_for_large_n() -> None:
    diffs = [float(index) for index in range(1, 22)]
    statistic, p_value = wilcoxon_signed_rank(diffs)
    assert statistic == 21 * 22 / 2
    assert 0.0 < p_value < 1e-4


def test_paired_ttest_known_value() -> None:
    # diffs = [1, 2, 3, 4, 5], mean = 3, sample var = 2.5, t = 3 / sqrt(0.5)
    # = 3*sqrt(2), df = 4. Two-sided p = I_{2/11}(2, 1/2) = 1 - 36*sqrt(11)/121.
    diffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    t_stat, df, p_value = paired_ttest(diffs)
    assert t_stat == pytest.approx(3.0 * math.sqrt(2.0))
    assert df == pytest.approx(4.0)
    expected_p = 1.0 - 36.0 * math.sqrt(11.0) / 121.0
    # expected_p = 0.0132355995636827 (closed form, not the continued fraction).
    assert expected_p == pytest.approx(0.0132355995636827, abs=1e-15)
    assert p_value == pytest.approx(expected_p, abs=1e-12)


def test_paired_ttest_matches_cauchy_for_df_one() -> None:
    diffs = [1.0, 3.0]
    t_stat, df, p_value = paired_ttest(diffs)
    assert t_stat == pytest.approx(2.0)
    assert df == pytest.approx(1.0)
    expected = 1.0 - 2.0 * math.atan(2.0) / math.pi
    assert p_value == pytest.approx(expected, abs=1e-12)


def test_paired_ttest_zero_variance() -> None:
    t_stat, df, p_value = paired_ttest([0.0, 0.0, 0.0])
    assert t_stat == 0.0
    assert df == pytest.approx(2.0)
    assert p_value == 1.0
    t_stat, df, p_value = paired_ttest([1.0, 1.0, 1.0])
    assert math.isinf(t_stat) and t_stat > 0
    assert df == pytest.approx(2.0)
    assert p_value == 0.0


def test_paired_ttest_too_small() -> None:
    assert paired_ttest([]) == (0.0, 0.0, 1.0)
    assert paired_ttest([1.5]) == (0.0, 0.0, 1.0)


def test_bca_matches_percentile_on_symmetric_diffs() -> None:
    diffs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    percentile = paired_bootstrap_ci(diffs, n_boot=2000, seed=13, method="percentile")
    bca = paired_bootstrap_ci(diffs, n_boot=2000, seed=13, method="bca")
    assert bca[0] == pytest.approx(percentile[0], abs=0.05)
    assert bca[1] == pytest.approx(percentile[1], abs=0.05)


def test_bca_endpoints_ordered_and_within_data_range() -> None:
    rng = random.Random(42)
    diffs = [rng.uniform(-1.0, 3.0) for _ in range(20)]
    low, high = paired_bootstrap_ci(diffs, n_boot=1000, seed=7, method="bca")
    assert low <= high
    assert min(diffs) <= low <= high <= max(diffs)


def test_bca_falls_back_to_percentile_on_edge_cases() -> None:
    constant = [0.4, 0.4, 0.4, 0.4]
    tiny = [0.1, -0.2]
    for diffs in (constant, tiny, [1.0]):
        percentile = paired_bootstrap_ci(diffs, n_boot=200, seed=13, method="percentile")
        bca = paired_bootstrap_ci(diffs, n_boot=200, seed=13, method="bca")
        assert bca == percentile


def test_compare_runs_notes_bca_fallback() -> None:
    run = run_with([("a", 1.0, True), ("b", 1.0, True)], "cfg")
    comparison = compare_runs(run, run, n_boot=50, seed=1, ci_method="bca")
    assert comparison.ci_method == "bca"
    assert comparison.ci_low == comparison.ci_high == 0.0
    assert any("percentile" in note for note in comparison.details)


def test_slice_aggregation_sums_to_overall_counts() -> None:
    tags = {"a": ["x"], "b": ["x"], "c": ["y"], "d": ["y"]}
    baseline = run_with(
        [("a", 0.0, False), ("b", 0.0, False), ("c", 1.0, True), ("d", 0.5, True)],
        "baseline",
        tags=tags,
    )
    candidate = run_with(
        [("a", 1.0, True), ("b", 1.0, True), ("c", 0.0, False), ("d", 0.5, True)],
        "candidate",
        tags=tags,
    )
    comparison = compare_runs(baseline, candidate, n_boot=80, seed=3)
    assert comparison.wins == 2
    assert comparison.losses == 1
    assert comparison.ties == 1
    by_tag = {item.tag: item for item in comparison.slices}
    assert set(by_tag) == {"x", "y"}
    assert by_tag["x"].n == 2
    assert by_tag["x"].wins == 2
    assert by_tag["x"].losses == 0
    assert by_tag["x"].ties == 0
    assert by_tag["y"].n == 2
    assert by_tag["y"].wins == 0
    assert by_tag["y"].losses == 1
    assert by_tag["y"].ties == 1
    assert sum(item.n for item in comparison.slices) == len(comparison.per_case)
    assert sum(item.wins for item in comparison.slices) == comparison.wins
    assert sum(item.losses for item in comparison.slices) == comparison.losses
    assert sum(item.ties for item in comparison.slices) == comparison.ties


def test_effect_size_labels_at_boundaries() -> None:
    assert effect_size_label(0.0) == "negligible"
    assert effect_size_label(0.199999) == "negligible"
    assert effect_size_label(0.2) == "small"
    assert effect_size_label(-0.2) == "small"
    assert effect_size_label(0.499999) == "small"
    assert effect_size_label(0.5) == "medium"
    assert effect_size_label(-0.5) == "medium"
    assert effect_size_label(0.799999) == "medium"
    assert effect_size_label(0.8) == "large"
    assert effect_size_label(-0.8) == "large"
    assert effect_size_label(float("inf")) == "large"
    assert cohens_dz([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(3.0 / math.sqrt(2.5))
    assert cohens_dz([0.0, 0.0]) == 0.0
    assert math.isinf(cohens_dz([1.0, 1.0]))


def test_compare_runs_surfaces_new_stats() -> None:
    baseline = run_with([("a", 1.0, True), ("b", 0.5, True), ("c", 0.0, False)], "baseline")
    candidate = run_with([("a", 0.0, False), ("b", 0.5, True), ("c", 1.0, True)], "candidate")
    comparison = compare_runs(baseline, candidate, n_boot=40, seed=2)
    assert comparison.ci_method == "percentile"
    assert 0.0 <= comparison.wilcoxon_p <= 1.0
    assert 0.0 <= comparison.ttest_p <= 1.0
    assert comparison.effect_size_label in {"negligible", "small", "medium", "large"}
    assert comparison.slices == []


def test_compare_runs_rejects_bad_ci_method() -> None:
    run = run_with([("a", 1.0, True)], "cfg")
    with pytest.raises(ValueError, match="ci_method"):
        compare_runs(run, run, ci_method="jackknife")


def test_compare_runs_rejects_duplicate_ids() -> None:
    run = RunResult(
        suite_name="s",
        config_name="c",
        results=[case_result("a", 1.0, True), case_result("a", 1.0, True)],
    )
    with pytest.raises(StatsError, match="Duplicate case id"):
        compare_runs(run, run)
