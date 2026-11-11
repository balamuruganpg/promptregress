"""Judge ensembles: mean scores, Fleiss' kappa, and single-judge regression."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from promptregress.calibration import fleiss_kappa, mean_pairwise_absolute
from promptregress.cli import (
    DEMO_BASELINE_TEMPLATE,
    DEMO_CANDIDATE_TEMPLATE,
    bundled_suite_path,
    demo_rules,
)
from promptregress.gate import evaluate_gate
from promptregress.irt import load_run_report
from promptregress.loading import load_suite
from promptregress.models import AssertionSpec, Case, Suite
from promptregress.providers import MockProvider
from promptregress.report import (
    comparison_to_dict,
    format_text,
    render_html,
    render_markdown,
    run_to_dict,
    write_run_json,
)
from promptregress.runner import run_suite
from promptregress.scorers import ScoringContext, score_case
from promptregress.stats import compare_runs

REGRESSED = {"password_reset", "refund_policy", "shipping_time"}


def _judges_for(suite: Suite) -> list[dict[str, object]]:
    strict_rules = [
        [f"\nOutput: {case.reference}\n", "1"] for case in suite.cases if case.reference is not None
    ]
    return [
        {
            "provider": "mock",
            "model": "strict",
            "rubric": "Score 1 only for the reference output.",
            "mock_rules": strict_rules,
            "mock_default": "0",
        },
        {
            "provider": "mock",
            "model": "lenient",
            "rubric": "Be generous.",
            "mock_default": "1",
        },
    ]


def _with_assertions(suite: Suite, assertions: list[AssertionSpec]) -> Suite:
    return replace(
        suite,
        cases=[replace(case, assertions=list(assertions)) for case in suite.cases],
    )


def test_fleiss_kappa_matches_hand_counted_tables() -> None:
    assert fleiss_kappa([]) is None
    assert fleiss_kappa([(1,)]) is None
    # P_bar = 2/3, chance = 0.5**2 + 0.5**2 = 0.5, kappa = (2/3 - 1/2) / (1/2) = 1/3.
    assert fleiss_kappa([(1, 1), (1, 0), (0, 0)]) == pytest.approx(1 / 3)
    assert fleiss_kappa([(1, 1), (1, 1)]) == pytest.approx(1.0)
    demo_ratings = [(1, 1)] * 11 + [(0, 1)] * 3
    assert fleiss_kappa(demo_ratings) == pytest.approx(-18 / 150)
    panels = [[0.0, 1.0]] * 3 + [[1.0, 1.0]] * 11
    assert mean_pairwise_absolute(panels) == pytest.approx(3 / 14)


def test_ensemble_mean_is_hand_computed() -> None:
    case = Case(
        id="c",
        input="question",
        reference="reference",
        assertions=[
            AssertionSpec(
                "llm_judge",
                params={
                    "judges": [
                        {
                            "provider": "mock",
                            "model": "strict",
                            "rubric": "marker",
                            "mock_rules": [["\nOutput: MARKER\n", "1"]],
                            "mock_default": "0",
                        },
                        {
                            "provider": "mock",
                            "model": "lenient",
                            "rubric": "lenient",
                            "mock_default": "0.25",
                        },
                    ]
                },
            )
        ],
    )
    context = ScoringContext(judge=MockProvider(default="0.9"))
    marked = score_case(case, "MARKER", context=context)
    assert marked[1] == pytest.approx((1.0 + 0.25) / 2)
    assert marked[2] is True
    assert context.judge is not None
    assert context.judge.call_count == 0
    detail = next(item for item in marked.details if item["kind"] == "llm_judge_ensemble")
    assert detail["aggregation"] == "mean"
    assert detail["score"] == pytest.approx(0.625)
    assert [row["score"] for row in detail["scores"]] == [1.0, 0.25]
    assert [row["label"] for row in detail["scores"]] == ["mock/strict", "mock/lenient"]
    assert marked.notes == ()

    plain = score_case(case, "other")
    assert plain[1] == pytest.approx((0.0 + 0.25) / 2)
    assert plain[2] is False


def test_failed_judge_is_zero_inside_the_mean() -> None:
    case = Case(
        id="c",
        input="q",
        reference=None,
        assertions=[
            AssertionSpec(
                "llm_judge",
                params={
                    "rubric": "shared",
                    "judges": [
                        {"provider": "mock", "model": "bad", "mock_default": "nope"},
                        {"provider": "mock", "model": "ok", "mock_default": "1"},
                    ],
                },
            )
        ],
    )
    scored = score_case(case, "output")
    assert scored[1] == pytest.approx((0.0 + 1.0) / 2)
    assert scored.notes
    assert "bad" in scored.notes[0]
    detail = next(item for item in scored.details if item["kind"] == "llm_judge_ensemble")
    assert detail["scores"][0]["score"] == 0.0
    assert detail["scores"][0]["error"]
    assert detail["scores"][1]["score"] == 1.0
    assert detail["scores"][1]["error"] is None


def test_malformed_judge_list_scores_zero_without_an_ensemble_record() -> None:
    case = Case(
        id="c",
        input="q",
        reference=None,
        assertions=[AssertionSpec("llm_judge", params={"judges": [], "rubric": "unused"})],
    )
    scored = score_case(case, "output")
    assert scored == ({"llm_judge": 0.0}, 0.0, False)
    assert "non-empty" in scored.notes[0]
    assert all(item.get("kind") != "llm_judge_ensemble" for item in scored.details)


def test_demo_suite_ensemble_agreement(tmp_path: Path) -> None:
    base = load_suite(bundled_suite_path())
    suite = _with_assertions(
        base,
        [AssertionSpec("llm_judge", params={"judges": _judges_for(base)})],
    )
    provider = MockProvider(rules=demo_rules(suite), default="I do not know.", model="demo")
    baseline = run_suite(
        suite,
        provider,
        "baseline",
        prompt_template=DEMO_BASELINE_TEMPLATE,
    )
    candidate = run_suite(
        suite,
        provider,
        "candidate",
        prompt_template=DEMO_CANDIDATE_TEMPLATE,
    )
    assert len(candidate.results) == 14
    for row in baseline.results:
        assert dict(row.judge_scores) == {"mock/strict": 1.0, "mock/lenient": 1.0}
        assert row.score == pytest.approx(1.0)
        assert row.scores["llm_judge"] == pytest.approx(1.0)
    strict_passes = 0
    for row in candidate.results:
        scores = dict(row.judge_scores)
        assert scores["mock/lenient"] == 1.0
        if row.case_id in REGRESSED:
            assert scores["mock/strict"] == 0.0
            assert row.score == pytest.approx(0.5)
        else:
            assert scores["mock/strict"] == 1.0
            assert row.score == pytest.approx(1.0)
            strict_passes += 1
    assert strict_passes == 11

    comparison = compare_runs(baseline, candidate, n_boot=20, seed=1)
    by_side = {item.side: item for item in comparison.judge_agreement}
    assert by_side["baseline"].mean_pairwise_abs == pytest.approx(0.0)
    assert by_side["baseline"].fleiss_kappa == pytest.approx(1.0)
    assert by_side["baseline"].per_judge_mean == pytest.approx((1.0, 1.0))
    agreed = by_side["candidate"]
    assert agreed.n_cases == 14
    assert agreed.n_judges == 2
    assert agreed.labels == ("mock/strict", "mock/lenient")
    assert agreed.mean_pairwise_abs == pytest.approx(3 / 14)
    assert agreed.fleiss_kappa == pytest.approx(-18 / 150)
    assert agreed.per_judge_mean == pytest.approx((11 / 14, 1.0))
    assert agreed.pass_threshold == 0.5

    verdict = evaluate_gate(comparison)
    text = format_text(comparison, verdict)
    markdown = render_markdown(comparison, verdict)
    html = render_html(comparison, verdict)
    assert "judge_agreement:" in text
    assert "mean_pairwise_abs=" in text
    assert "fleiss_kappa=" in text
    assert "Judge ensemble agreement" in markdown
    assert "Fleiss" in markdown
    assert "Judge ensemble agreement" in html
    assert "Fleiss" in html
    payload = comparison_to_dict(comparison, verdict)
    candidate_report = payload["judge_agreement"][1]
    assert candidate_report["side"] == "candidate"
    assert candidate_report["mean_pairwise_abs"] == pytest.approx(3 / 14)
    assert candidate_report["fleiss_kappa"] == pytest.approx(-18 / 150)
    assert "Fleiss" in candidate_report["definition"]

    path = tmp_path / "candidate.json"
    write_run_json(candidate, path)
    loaded = load_run_report(path)
    assert loaded.results[0].judge_scores == candidate.results[0].judge_scores
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["results"][0]["judge_scores"][0][0] == "mock/strict"


def test_single_judge_demo_report_has_no_agreement_section() -> None:
    base = load_suite(bundled_suite_path())
    suite = _with_assertions(
        base,
        [AssertionSpec("llm_judge", params={"rubric": "grade the output"})],
    )
    judge = MockProvider(
        rules=[
            (f"\nOutput: {case.reference}\n", "1")
            for case in base.cases
            if case.reference is not None
        ],
        default="0",
    )
    provider = MockProvider(rules=demo_rules(suite), default="I do not know.", model="demo")
    baseline = run_suite(
        suite,
        provider,
        "baseline",
        prompt_template=DEMO_BASELINE_TEMPLATE,
        context=ScoringContext(judge=judge),
    )
    assert judge.call_count == len(base.cases)
    assert all(row.score == pytest.approx(1.0) for row in baseline.results)
    assert all(row.judge_scores == () for row in baseline.results)
    direct = score_case(
        suite.cases[0],
        base.cases[0].reference or "",
        context=ScoringContext(judge=MockProvider(default="0.75")),
    )
    assert direct.details == ()
    assert direct.notes == ()
    assert direct[1] == 0.75

    comparison = compare_runs(baseline, baseline, n_boot=10, seed=1)
    assert comparison.judge_agreement == ()
    payload = comparison_to_dict(comparison, evaluate_gate(comparison))
    assert "judge_agreement" not in payload
    text = format_text(comparison)
    assert "judge_agreement" not in text
    assert "fleiss_kappa" not in text
    assert "judge_scores" not in run_to_dict(baseline)["results"][0]
