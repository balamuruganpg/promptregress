"""Offline pairwise judge: a marker wins, and an always-first judge ties."""

import json
from pathlib import Path

from promptregress.cli import main
from promptregress.gate import evaluate_gate
from promptregress.models import AssertionSpec, Case, CaseResult, Comparison, RunResult, Suite
from promptregress.pairwise import combine_orders, pairwise_judge, with_pairwise
from promptregress.providers import MockProvider
from promptregress.report import render_html, render_markdown, write_json
from promptregress.stats import compare_runs


def _comparison(n: int, baseline: str, candidate: str) -> tuple[Suite, Comparison]:
    cases: list[Case] = []
    base_rows: list[CaseResult] = []
    cand_rows: list[CaseResult] = []
    for index in range(n):
        case_id = f"c{index}"
        cases.append(
            Case(
                id=case_id,
                input=f"question {index}",
                reference="ref",
                assertions=[AssertionSpec("exact_match", value="unused")],
            )
        )
        base_rows.append(
            CaseResult(
                case_id=case_id,
                output=baseline,
                scores={"exact_match": 1.0},
                score=1.0,
                passed=True,
                latency_ms=1.0,
            )
        )
        cand_rows.append(
            CaseResult(
                case_id=case_id,
                output=candidate,
                scores={"exact_match": 1.0},
                score=1.0,
                passed=True,
                latency_ms=1.0,
            )
        )
    suite = Suite(name="pairs", cases=cases)
    comparison = compare_runs(
        RunResult("pairs", "baseline", base_rows),
        RunResult("pairs", "candidate", cand_rows),
        n_boot=40,
        seed=3,
    )
    return suite, comparison


class _Recording(MockProvider):
    def __init__(self) -> None:
        super().__init__(default="first")
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return super().complete(prompt)


def test_combine_orders_scores() -> None:
    assert combine_orders("candidate", "candidate") == ("win", 1.0)
    assert combine_orders("baseline", "baseline") == ("loss", 0.0)
    assert combine_orders("baseline", "candidate") == ("tie", 0.5)
    assert combine_orders("candidate", "tie") == ("tie", 0.5)
    assert combine_orders("tie", "tie") == ("tie", 0.5)


def test_marker_judge_makes_the_candidate_win_both_orders() -> None:
    suite, comparison = _comparison(3, "BASELINE_TEXT", "CANDIDATE_WINS")
    judge = MockProvider(
        rules=[
            ("Output A:\nCANDIDATE_WINS\n", "first"),
            ("Output B:\nCANDIDATE_WINS\n", "second"),
        ],
        default="tie",
    )
    summary = pairwise_judge(
        comparison,
        suite,
        judge,
        "prefer the marked answer",
        n_boot=30,
        seed=1,
    )
    assert judge.call_count == 6
    assert summary.win_rate == 1.0
    assert summary.wins == 3
    assert summary.losses == 0
    assert summary.ties == 0
    assert (summary.ci_low, summary.ci_high) == (1.0, 1.0)
    assert all(case.outcome == "win" and case.score == 1.0 for case in summary.cases)
    assert all(case.error is None for case in summary.cases)
    assert all(case.order_baseline_first == "candidate" for case in summary.cases)
    assert all(case.order_candidate_first == "candidate" for case in summary.cases)


def test_always_first_judge_is_neutralised_to_one_half() -> None:
    suite, comparison = _comparison(4, "BASE_OUTPUT", "CAND_OUTPUT")
    judge = _Recording()
    summary = pairwise_judge(comparison, suite, judge, "pick one", n_boot=25, seed=2)
    assert judge.call_count == 8
    assert len(judge.prompts) == 8
    for index, prompt in enumerate(judge.prompts):
        base_at = prompt.index("BASE_OUTPUT")
        cand_at = prompt.index("CAND_OUTPUT")
        if index % 2 == 0:
            assert base_at < cand_at
        else:
            assert cand_at < base_at
    assert summary.win_rate == 0.5
    assert summary.wins == 0
    assert summary.losses == 0
    assert summary.ties == 4
    assert (summary.ci_low, summary.ci_high) == (0.5, 0.5)
    assert all(case.order_baseline_first == "baseline" for case in summary.cases)
    assert all(case.order_candidate_first == "candidate" for case in summary.cases)
    assert all(case.outcome == "tie" and case.score == 0.5 for case in summary.cases)


def test_unreadable_judge_is_a_tie_and_does_not_raise() -> None:
    suite, comparison = _comparison(1, "left", "right")
    judge = MockProvider(default="maybe")
    summary = pairwise_judge(comparison, suite, judge, "rubric", n_boot=10, seed=1)
    assert summary.win_rate == 0.5
    assert summary.cases[0].outcome == "tie"
    assert summary.cases[0].error is not None
    assert "maybe" in summary.cases[0].error


def test_reports_include_the_pairwise_interval(tmp_path: Path) -> None:
    suite, comparison = _comparison(2, "BASELINE_TEXT", "CANDIDATE_WINS")
    judge = MockProvider(
        rules=[
            ("Output A:\nCANDIDATE_WINS\n", "first"),
            ("Output B:\nCANDIDATE_WINS\n", "second"),
        ],
        default="tie",
    )
    summary = pairwise_judge(comparison, suite, judge, "marked", n_boot=10, seed=1)
    judged = with_pairwise(comparison, summary)
    verdict = evaluate_gate(judged)
    path = tmp_path / "cmp.json"
    write_json(judged, path, verdict)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pairwise"]["win_rate"] == 1.0
    assert payload["pairwise"]["ci_low"] == 1.0
    assert payload["pairwise"]["ci_high"] == 1.0
    markdown = render_markdown(judged, verdict)
    html = render_html(judged, verdict)
    assert "Pairwise judge" in markdown
    assert "Mean pairwise score" in markdown
    assert "1.000000" in markdown
    assert "Pairwise judge" in html
    assert "Mean pairwise score" in html
    assert "<script" not in html
    assert "http://" not in html
    assert "https://" not in html


def test_cli_pairwise_judge(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [
                    {
                        "id": "a",
                        "input": "alpha",
                        "reference": "never",
                        "assertions": [{"kind": "exact_match"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rules = [
        ["Output A:\nCANDIDATE_WINS ", "first"],
        ["Output B:\nCANDIDATE_WINS ", "second"],
    ]
    report = tmp_path / "report.json"
    code = main(
        [
            "compare",
            str(suite),
            "--provider",
            "echo",
            "--baseline-template",
            "BASE {input}",
            "--candidate-template",
            "CANDIDATE_WINS {input}",
            "--pairwise-judge",
            "--judge-provider",
            "mock",
            "--judge-rubric",
            "prefer the marker",
            "--judge-mock-rules",
            json.dumps(rules),
            "--judge-mock-default",
            "tie",
            "--max-regression",
            "1",
            "--report-json",
            str(report),
            "--no-history",
            "--seed",
            "1",
        ]
    )
    assert code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["pairwise"]["win_rate"] == 1.0
    assert payload["pairwise"]["cases"][0]["outcome"] == "win"
    assert payload["gate"]["passed"] is True
