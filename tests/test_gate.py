"""Each gate rule fails and passes with the numbers in the reason."""

from helpers import comparison_with
from promptregress.gate import GateRule, evaluate_gate


def test_regression_rule_fails_and_passes() -> None:
    comparison = comparison_with(
        ci_low=-0.4,
        ci_high=-0.1,
        mean_diff=-0.2,
        candidate_rows=[("a", 1.0, True)],
    )
    failed = evaluate_gate(comparison, max_regression=0.0)
    assert failed.passed is False
    assert any(
        reason.startswith("FAIL regression") and "0.400000" in reason and "0.000000" in reason
        for reason in failed.reasons
    )
    assert any("mean_diff -0.200000" in reason for reason in failed.reasons)
    passed = evaluate_gate(comparison, max_regression=0.4)
    assert passed.passed is True
    assert any(
        reason.startswith("PASS regression") and "0.400000" in reason for reason in passed.reasons
    )


def test_pass_rate_rule_fails_at_the_boundary() -> None:
    comparison = comparison_with(
        ci_low=0.0,
        ci_high=0.0,
        mean_diff=0.0,
        candidate_rows=[("a", 1.0, True), ("b", 0.0, False)],
    )
    failed = evaluate_gate(comparison, max_regression=0.0, min_candidate_pass_rate=0.75)
    assert failed.passed is False
    assert any(
        reason.startswith("FAIL pass rate") and "0.500000" in reason and "0.750000" in reason
        for reason in failed.reasons
    )
    held = evaluate_gate(comparison, max_regression=0.0, min_candidate_pass_rate=0.5)
    assert held.passed is True
    assert any(
        reason.startswith("PASS pass rate") and "0.500000" in reason for reason in held.reasons
    )


def test_require_significant_fails_and_passes() -> None:
    weak = comparison_with(
        ci_low=-0.01,
        ci_high=0.2,
        mean_diff=0.05,
        candidate_rows=[("a", 1.0, True)],
    )
    failed = evaluate_gate(weak, max_regression=0.01, require_significant=True)
    assert failed.passed is False
    assert any(
        reason.startswith("FAIL significance") and "-0.010000" in reason
        for reason in failed.reasons
    )
    strong = comparison_with(
        ci_low=0.01,
        ci_high=0.2,
        mean_diff=0.05,
        candidate_rows=[("a", 1.0, True)],
    )
    passed = evaluate_gate(strong, max_regression=0.0, require_significant=True)
    assert passed.passed is True
    assert any(
        reason.startswith("PASS significance") and "0.010000" in reason for reason in passed.reasons
    )


def test_significance_off_is_reported() -> None:
    comparison = comparison_with(
        ci_low=0.0,
        ci_high=0.0,
        mean_diff=0.0,
        candidate_rows=[("a", 1.0, True)],
    )
    verdict = evaluate_gate(comparison)
    assert verdict.passed is True
    assert any("require_significant is off" in reason for reason in verdict.reasons)


def test_all_three_rules_can_fail_together() -> None:
    comparison = comparison_with(
        ci_low=-0.5,
        ci_high=-0.1,
        mean_diff=-0.3,
        candidate_rows=[("a", 0.0, False)],
    )
    verdict = evaluate_gate(
        comparison,
        max_regression=0.0,
        min_candidate_pass_rate=0.9,
        require_significant=True,
    )
    assert verdict.passed is False
    assert sum(reason.startswith("FAIL ") for reason in verdict.reasons) == 3


def test_gate_rule_replaces_scalar_arguments() -> None:
    comparison = comparison_with(
        ci_low=-0.4,
        ci_high=-0.1,
        mean_diff=-0.2,
        candidate_rows=[("a", 1.0, True)],
    )
    rule = GateRule(max_regression=1.0, min_candidate_pass_rate=0.0, require_significant=False)
    verdict = evaluate_gate(comparison, max_regression=0.0, rule=rule)
    assert verdict.passed is True
    assert any("max_regression 1.000000" in reason for reason in verdict.reasons)
