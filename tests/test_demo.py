"""The bundled offline demo regresses three cases and fails the gate."""

from pathlib import Path

import pytest

from promptregress.cli import (
    DEMO_CANDIDATE_TEMPLATE,
    REGRESSED_ANSWERS,
    format_demo,
    main,
    run_demo,
)
from promptregress.loading import load_suite
from promptregress.scorers import score_case

ROOT = Path(__file__).resolve().parents[1]


def test_references_pass_and_brief_answers_score_zero() -> None:
    suite = load_suite(ROOT / "examples" / "support_qa.json")
    assert len(suite.cases) >= 12
    for case in suite.cases:
        assert case.reference is not None
        _, score, passed = score_case(case, case.reference)
        assert score == 1.0
        assert passed is True
        short = REGRESSED_ANSWERS.get(case.id)
        if short is None:
            continue
        _, brief_score, brief_passed = score_case(case, short)
        assert brief_score == 0.0
        assert brief_passed is False


def test_demo_numbers_are_exact() -> None:
    comparison, verdict = run_demo()
    assert DEMO_CANDIDATE_TEMPLATE.startswith("Answer briefly.")
    assert comparison.baseline.suite_name == "support_qa"
    assert comparison.mean_diff == -3 / 14
    assert comparison.ci_low == -0.42857142857142855
    assert comparison.ci_high == 0.0
    assert comparison.wins == 0
    assert comparison.losses == 3
    assert comparison.ties == 11
    assert comparison.mcnemar_b == 3
    assert comparison.mcnemar_c == 0
    assert comparison.mcnemar_p == 0.25
    assert comparison.baseline.mean_score == 1.0
    assert comparison.baseline.pass_rate == 1.0
    assert comparison.candidate.mean_score == pytest.approx(11 / 14)
    assert comparison.candidate.pass_rate == 11 / 14
    assert verdict.passed is False
    lost = [row.case_id for row in comparison.per_case if row.outcome == "loss"]
    assert lost == ["password_reset", "refund_policy", "shipping_time"]
    for row in comparison.per_case:
        if row.case_id in set(lost):
            assert row.baseline_score == 1.0
            assert row.candidate_score == 0.0
            assert row.baseline_passed is True
            assert row.candidate_passed is False
        else:
            assert row.diff == 0.0
            assert row.baseline_score == 1.0
            assert row.candidate_score == 1.0
            assert row.baseline_passed is True
            assert row.candidate_passed is True
    text = format_demo(comparison, verdict)
    assert "mean_diff=-0.214286" in text
    assert "ci_low=-0.428571" in text
    assert "ci_high=0.000000" in text
    assert "gate_passed=false" in text
    assert "FAIL regression" in text


def test_demo_is_deterministic() -> None:
    first, first_verdict = run_demo()
    second, second_verdict = run_demo()
    assert first.mean_diff == second.mean_diff
    assert first.ci_low == second.ci_low
    assert first.ci_high == second.ci_high
    assert first.mcnemar_p == second.mcnemar_p
    rows = [
        (row.case_id, row.diff, row.baseline_output, row.candidate_output) for row in first.per_case
    ]
    again = [
        (row.case_id, row.diff, row.baseline_output, row.candidate_output)
        for row in second.per_case
    ]
    assert rows == again
    assert first_verdict == second_verdict


def test_demo_command_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo"]) == 1
    output = capsys.readouterr().out
    assert "provider=mock" in output
    assert "losses=3" in output
    assert "wins=0" in output
    assert "ties=11" in output
    assert "mcnemar_p=0.250000" in output
    assert "candidate_pass_rate=0.785714" in output
    assert "password_reset baseline=1.000000 candidate=0.000000" in output
    assert "gate_passed=false" in output
