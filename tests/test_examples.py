"""Offline example suites: valid cases, mock rules, deterministic compare."""

import json
from pathlib import Path

from promptregress.loading import load_suite
from promptregress.models import Suite
from promptregress.providers import MockProvider
from promptregress.runner import run_suite
from promptregress.scorers import score_case
from promptregress.stats import compare_runs

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

BASELINE_TEMPLATE = "Answer using the case.\n{input}"
CANDIDATE_TEMPLATE = "Answer briefly.\n{input}"

# Brief answers are registered before the full-input rules. Each one drops a
# detail the assertions require, the same way the support demo does.
RAG_BRIEF = {
    "refund_window": "Refunds are available for unused seats.",
    "link_expiry": "Use the reset link sent to your email.",
    "ship_cutoff": "Orders ship from the warehouse.",
    "export_delay": "We email you a copy of your data.",
}

CODE_BRIEF = {
    "palindrome": "def is_palindrome(text: str) -> bool:\n    return False\n",
    "user_json": "name=Ada Lovelace",
    "fizzbuzz": "def fizzbuzz(n: int) -> list[str]:\n    return []\n",
}

SUMMARY_BRIEF = {
    "outage": "Checkout was down for 42 minutes.",
    "hiring": "The platform group is hiring three engineers.",
    "budget": "Travel spend is cut by 12 percent.",
}


def _rules(suite: Suite, brief: dict[str, str]) -> list[tuple[str, str]]:
    """Mock rules keyed to case content. Brief needles are listed first."""
    rules: list[tuple[str, str]] = []
    for case in suite.cases:
        short = brief.get(case.id)
        if short is not None:
            rules.append((f"Answer briefly.\n{case.input}", short))
    full = [(case.input, case.reference or "") for case in suite.cases]
    full.sort(key=lambda item: len(item[0]), reverse=True)
    rules.extend(full)
    return rules


def _compare(suite: Suite, brief: dict[str, str]):
    provider = MockProvider(
        rules=_rules(suite, brief),
        default="I do not know.",
        name="mock",
        model="examples",
    )
    baseline = run_suite(
        suite,
        provider,
        "baseline",
        prompt_template=BASELINE_TEMPLATE,
    )
    candidate = run_suite(
        suite,
        provider,
        "candidate",
        prompt_template=CANDIDATE_TEMPLATE,
    )
    return compare_runs(baseline, candidate, seed=13)


def _assert_suite(suite: Suite, expected: int, kinds: set[str]) -> None:
    assert suite.name
    assert len(suite.cases) == expected
    ids = [case.id for case in suite.cases]
    assert len(ids) == len(set(ids))
    inputs = [case.input for case in suite.cases]
    for left in inputs:
        for right in inputs:
            if left is not right:
                assert left not in right
    for case in suite.cases:
        assert case.reference
        assert case.tags
        assert case.assertions
        assert {assertion.kind for assertion in case.assertions} <= kinds
        _, score, passed = score_case(case, case.reference)
        assert score == 1.0
        assert passed is True


def _assert_compare(path: Path, brief: dict[str, str], expected: int, kinds: set[str]) -> None:
    suite = load_suite(path)
    _assert_suite(suite, expected, kinds)
    for case in suite.cases:
        short = brief.get(case.id)
        if short is None:
            continue
        _, score, passed = score_case(case, short)
        assert score < 1.0
        assert passed is False
    first = _compare(suite, brief)
    second = _compare(suite, brief)
    assert first.baseline.pass_rate == 1.0
    assert first.baseline.mean_score == 1.0
    assert first.losses == len(brief)
    assert first.wins == 0
    assert first.ties == expected - len(brief)
    assert first.mean_diff < 0.0
    assert first.ci_low <= first.ci_high
    assert 0.0 <= first.mcnemar_p <= 1.0
    assert 0.0 <= first.wilcoxon_p <= 1.0
    lost = [row.case_id for row in first.per_case if row.outcome == "loss"]
    assert lost == [case.id for case in suite.cases if case.id in brief]
    for row in first.per_case:
        if row.case_id in brief:
            assert row.candidate_output == brief[row.case_id]
            assert row.candidate_score < row.baseline_score
            assert row.baseline_passed is True
            assert row.candidate_passed is False
        else:
            assert row.diff == 0.0
            assert row.candidate_score == 1.0
            assert row.candidate_passed is True
        assert row.baseline_score == 1.0
        assert row.baseline_output
    tags = {tag for case in suite.cases for tag in case.tags}
    assert tags == {item.tag for item in first.slices}
    assert sum(item.n for item in first.slices) == sum(len(case.tags) for case in suite.cases)
    for item in first.slices:
        assert item.ci_low <= item.ci_high
        assert item.n >= 1
    assert first.mean_diff == second.mean_diff
    assert first.ci_low == second.ci_low
    assert first.ci_high == second.ci_high
    assert first.mcnemar_p == second.mcnemar_p
    assert [(row.case_id, row.diff, row.candidate_output) for row in first.per_case] == [
        (row.case_id, row.diff, row.candidate_output) for row in second.per_case
    ]


def test_rag_suite_compare_is_deterministic() -> None:
    suite = load_suite(EXAMPLES / "rag_qa.json")
    for case in suite.cases:
        for assertion in case.assertions:
            assert isinstance(assertion.value, str)
            assert assertion.value in case.input
            assert assertion.value in (case.reference or "")
    _assert_compare(EXAMPLES / "rag_qa.json", RAG_BRIEF, 12, {"contains"})
    comparison = _compare(suite, RAG_BRIEF)
    assert comparison.mean_diff == -len(RAG_BRIEF) / 12


def test_code_suite_compare_is_deterministic() -> None:
    suite = load_suite(EXAMPLES / "code_gen.json")
    for case in suite.cases:
        for assertion in case.assertions:
            if assertion.kind == "contains":
                assert isinstance(assertion.value, str)
                assert assertion.value in (case.reference or "")
            if assertion.kind == "json_schema":
                assert isinstance(assertion.value, dict)
    _assert_compare(
        EXAMPLES / "code_gen.json",
        CODE_BRIEF,
        10,
        {"contains", "regex", "json_valid", "json_schema"},
    )


def _trace_brief(suite: Suite) -> dict[str, str]:
    """Brief template: drop two refund steps, reverse the sales trace, keep one ops step."""
    steps = {case.id: case.assertions[0].value for case in suite.cases}
    return {
        "refund": json.dumps([steps["refund"][0]]),
        "sales_chart": json.dumps(list(reversed(steps["sales_chart"]))),
        "incident": json.dumps([steps["incident"][-1]]),
    }


def test_traces_suite_compare_is_deterministic() -> None:
    path = EXAMPLES / "traces.json"
    suite = load_suite(path)
    brief = _trace_brief(suite)
    for case in suite.cases:
        assert [assertion.kind for assertion in case.assertions] == ["trace_match"]
        assert isinstance(case.assertions[0].value, list)
        assert len(case.assertions[0].value) == 3
    _assert_compare(path, brief, 8, {"trace_match"})
    comparison = _compare(suite, brief)
    assert comparison.losses == 3
    assert comparison.wins == 0
    assert comparison.ties == 5
    assert comparison.baseline.pass_rate == 1.0
    assert comparison.candidate.pass_rate == 5 / 8
    assert comparison.mean_diff == 3 * ((1 / 3) - 1.0) / 8
    for row in comparison.per_case:
        if row.case_id in brief:
            assert row.candidate_score == 1 / 3
            assert row.baseline_passed is True
            assert row.candidate_passed is False
        else:
            assert row.diff == 0.0


def test_summarization_suite_compare_is_deterministic() -> None:
    suite = load_suite(EXAMPLES / "summarization.json")
    for case in suite.cases:
        kinds = [assertion.kind for assertion in case.assertions]
        assert "tfidf_cosine" in kinds
        assert kinds.count("contains") >= 2
        for assertion in case.assertions:
            if assertion.kind != "contains":
                continue
            assert isinstance(assertion.value, str)
            assert assertion.value.casefold() in (case.reference or "").casefold()
            assert assertion.value.casefold() in case.input.casefold()
    _assert_compare(
        EXAMPLES / "summarization.json",
        SUMMARY_BRIEF,
        10,
        {"tfidf_cosine", "contains"},
    )
