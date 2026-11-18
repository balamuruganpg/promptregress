"""Active case selection: priority formula, suite copy, and the shipped example."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from promptregress.cli import main
from promptregress.loading import load_suite
from promptregress.models import (
    AssertionSpec,
    Case,
    CaseResult,
    Comparison,
    PairedRow,
    RunResult,
    ScoringConfig,
    Suite,
)
from promptregress.providers import MockProvider
from promptregress.report import write_json
from promptregress.runner import run_suite
from promptregress.select import (
    PRIORITY_FORMULA,
    SELECT_LIMIT,
    SELECT_NOTE,
    VARIANCE_SCALE,
    format_selection,
    prioritize,
    select_suite,
    selection_to_dict,
)
from promptregress.stats import compare_runs
from promptregress.synth import write_suite

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "select"

EXAMPLE_DESCRIPTION = (
    "Source suite for active selection. billing flips pass-to-fail in every saved "
    "comparison. latency flips once and also drops while still passing. greeting "
    "stays a pass with score 1. unseen has no comparison history. The comparisons "
    "also record retired, which is not in the source suite."
)

_TAGS = {
    "billing": "account",
    "latency": "perf",
    "greeting": "tone",
    "retired": "legacy",
    "unseen": "shipping",
}


def _case(case_id: str, question: str, answer: str, tag: str, planted: str) -> Case:
    return Case(
        id=case_id,
        input=question,
        reference=answer,
        assertions=[AssertionSpec(kind="exact_match")],
        tags=[tag],
        metadata={"planted": planted},
    )


def example_suite() -> Suite:
    """The planted suite, in an order that is not the selection order."""
    return Suite(
        name="select-demo",
        description=EXAMPLE_DESCRIPTION,
        cases=[
            _case(
                "unseen",
                "Where is my order?",
                "Track the order from the orders page.",
                "shipping",
                "no-history",
            ),
            _case("greeting", "Say hello.", "Hello.", "tone", "stable"),
            _case(
                "latency",
                "Why is the page slow?",
                "The page loads in under two seconds.",
                "perf",
                "flaky-score",
            ),
            _case(
                "billing",
                "What is the billing cycle?",
                "Bills are issued on the first of the month.",
                "account",
                "pass-to-fail",
            ),
        ],
    )


def _result(case_id: str, score: float, passed: bool) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        output="pass" if passed else "fail",
        scores={"exact_match": score},
        score=score,
        passed=passed,
        latency_ms=1.0,
        tags=[_TAGS[case_id]] if case_id in _TAGS else [],
    )


def _paired(rows: list[tuple[str, float, bool, float, bool]]) -> Comparison:
    baseline = RunResult(
        suite_name="select-demo",
        config_name="baseline",
        results=[_result(case_id, b_score, b_pass) for case_id, b_score, b_pass, _, _ in rows],
    )
    candidate = RunResult(
        suite_name="select-demo",
        config_name="candidate",
        results=[_result(case_id, c_score, c_pass) for case_id, _, _, c_score, c_pass in rows],
    )
    return compare_runs(baseline, candidate, seed=13)


def example_comparisons() -> list[Comparison]:
    """Three comparisons written with ``compare_runs`` and ``write_json``.

    ``retired`` is history that the suite does not contain. ``latency`` on
    the third comparison drops from 1 to 0.5 and still passes.
    """
    first = [
        ("billing", 1.0, True, 0.0, False),
        ("latency", 1.0, True, 1.0, True),
        ("greeting", 1.0, True, 1.0, True),
        ("retired", 1.0, True, 0.0, False),
    ]
    second = [
        ("billing", 1.0, True, 0.0, False),
        ("latency", 1.0, True, 0.0, False),
        ("greeting", 1.0, True, 1.0, True),
        ("retired", 1.0, True, 1.0, True),
    ]
    third = [
        ("billing", 1.0, True, 0.5, False),
        ("latency", 1.0, True, 0.5, True),
        ("greeting", 1.0, True, 1.0, True),
    ]
    return [_paired(first), _paired(second), _paired(third)]


def write_example_inputs(dest: Path) -> None:
    """Write the suite and the three comparison files with the project writers."""
    dest.mkdir(parents=True, exist_ok=True)
    write_suite(example_suite(), dest / "suite.json")
    for index, comparison in enumerate(example_comparisons(), start=1):
        write_json(comparison, dest / f"cmp_{index:02d}.json")


def _row(
    case_id: str,
    baseline_score: float,
    candidate_score: float,
    *,
    baseline_passed: bool = True,
    candidate_passed: bool = False,
) -> PairedRow:
    return PairedRow(
        case_id=case_id,
        baseline_output="b",
        candidate_output="c",
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        diff=candidate_score - baseline_score,
        baseline_passed=baseline_passed,
        candidate_passed=candidate_passed,
        outcome="loss",
    )


def _comparison(rows: list[PairedRow]) -> Comparison:
    return Comparison(
        baseline=RunResult("s", "baseline", []),
        candidate=RunResult("s", "candidate", []),
        mean_diff=0.0,
        ci_low=0.0,
        ci_high=0.0,
        mcnemar_p=1.0,
        wins=0,
        losses=0,
        ties=0,
        per_case=rows,
    )


def _suite(ids: list[str], name: str = "demo") -> Suite:
    return Suite(
        name=name,
        cases=[
            Case(
                id=case_id,
                input=case_id,
                reference=case_id,
                assertions=[AssertionSpec(kind="exact_match")],
            )
            for case_id in ids
        ],
    )


def _priority(flip_rate: float, variance: float, mean_drop: float) -> float:
    return (flip_rate + variance / VARIANCE_SCALE + mean_drop) / 3.0


def test_formula_components_and_exact_sums() -> None:
    rows = [
        _row("mix", 1.0, 0.0, baseline_passed=True, candidate_passed=False),
        _row("mix", 1.0, 1.0, baseline_passed=True, candidate_passed=True),
    ]
    # Two comparisons, one row each. Scores 0 and 1: population variance is 0.25.
    report = prioritize(
        _suite(["mix"]),
        [_comparison([rows[0]]), _comparison([rows[1]])],
        budget=1,
        seed=13,
    )
    item = report.selected[0]
    scores = [0.0, 1.0]
    mean = math.fsum(sorted(scores)) / 2
    variance = math.fsum(sorted((score - mean) ** 2 for score in scores)) / 2
    assert item.history is True
    assert item.n_observed == 2
    assert item.n_flips == 1
    assert item.flip_rate == 0.5
    assert item.candidate_variance == variance == 0.25
    assert item.variance_component == 1.0
    assert item.n_regressed == 1
    assert item.mean_drop == 1.0
    assert item.priority == _priority(0.5, 0.25, 1.0)
    assert VARIANCE_SCALE == 0.25
    assert PRIORITY_FORMULA == "(flip_rate + candidate_variance / 0.25 + mean_drop) / 3"


def test_one_observation_has_zero_variance_and_gains_do_not_cancel() -> None:
    only = prioritize(
        _suite(["once"]),
        [_comparison([_row("once", 1.0, 0.25)])],
        budget=1,
    )
    assert only.selected[0].candidate_variance == 0.0
    assert only.selected[0].variance_component == 0.0
    assert only.selected[0].mean_drop == 0.75

    gain = prioritize(
        _suite(["swing"]),
        [
            _comparison([_row("swing", 1.0, 0.0)]),
            _comparison([_row("swing", 0.0, 1.0, baseline_passed=False, candidate_passed=True)]),
        ],
        budget=1,
    )
    assert gain.selected[0].n_regressed == 1
    assert gain.selected[0].mean_drop == 1.0
    assert gain.selected[0].n_flips == 1


def test_flip_and_score_drop_are_separate_and_tiny_drops_count() -> None:
    flip_only = prioritize(
        _suite(["flip"]),
        [_comparison([_row("flip", 1.0, 1.0, baseline_passed=True, candidate_passed=False)])],
        budget=1,
    )
    assert flip_only.selected[0].flip_rate == 1.0
    assert flip_only.selected[0].mean_drop == 0.0
    assert flip_only.selected[0].n_regressed == 0

    drop_only = prioritize(
        _suite(["drop"]),
        [_comparison([_row("drop", 1.0, 0.0, baseline_passed=True, candidate_passed=True)])],
        budget=1,
    )
    assert drop_only.selected[0].flip_rate == 0.0
    assert drop_only.selected[0].mean_drop == 1.0

    tiny = 1.0 - 1e-13
    assert tiny < 1.0
    assert abs(tiny - 1.0) <= 1e-12
    counted = prioritize(
        _suite(["tiny"]),
        [_comparison([_row("tiny", 1.0, tiny, baseline_passed=True, candidate_passed=True)])],
        budget=1,
    )
    assert counted.selected[0].n_regressed == 1
    assert counted.selected[0].mean_drop == 1.0 - tiny


def test_scores_outside_unit_interval_are_not_clamped() -> None:
    report = prioritize(
        _suite(["wide"]),
        [
            _comparison([_row("wide", 2.0, 0.0, baseline_passed=True, candidate_passed=False)]),
            _comparison([_row("wide", 2.0, 2.0, baseline_passed=True, candidate_passed=True)]),
        ],
        budget=1,
    )
    item = report.selected[0]
    assert item.candidate_variance == 1.0
    assert item.variance_component == 4.0
    assert item.mean_drop == 2.0
    assert item.priority is not None
    assert item.priority > 1.0
    assert item.priority == _priority(item.flip_rate or 0.0, 1.0, 2.0)


def test_ties_break_by_code_point_and_no_history_ranks_last() -> None:
    same = _row("ignored", 1.0, 0.0)
    report = prioritize(
        _suite(["b", "A", "m", "stable"]),
        [
            _comparison(
                [
                    _row("b", 1.0, 0.0),
                    _row("A", 1.0, 0.0),
                    _row("stable", 1.0, 1.0, baseline_passed=True, candidate_passed=True),
                ]
            )
        ],
        budget=10,
    )
    assert [item.case_id for item in report.selected] == ["A", "b", "stable", "m"]
    assert report.selected[0].priority == report.selected[1].priority
    assert report.selected[2].priority == 0.0
    assert report.selected[2].history is True
    assert report.selected[3].history is False
    assert report.selected[3].priority is None
    assert report.n_with_history == 3
    assert report.n_without_history == 1
    assert len(report.selected) == 4
    text = format_selection(report)
    assert "- m history=false priority=- " in text
    assert same.case_id == "ignored"


def test_budget_cut_keeps_history_ahead_of_zero_priority_only_when_ranked() -> None:
    report = prioritize(
        _suite(["quiet", "flat"]),
        [_comparison([_row("flat", 1.0, 1.0, baseline_passed=True, candidate_passed=True)])],
        budget=1,
    )
    assert [item.case_id for item in report.selected] == ["flat"]
    assert report.n_suite == 2
    assert report.n_with_history == 1


def test_ignored_history_ids_are_unique_and_sorted() -> None:
    report = prioritize(
        _suite(["kept"]),
        [
            _comparison([_row("kept", 1.0, 0.0), _row("ghost", 1.0, 0.0)]),
            _comparison([_row("kept", 1.0, 0.0), _row("ghost", 1.0, 1.0), _row("other", 1.0, 0.0)]),
        ],
        budget=1,
    )
    assert report.ignored_history_ids == ("ghost", "other")
    assert "ignored_history_ids=ghost, other" in format_selection(report)
    assert report.selected[0].n_observed == 2
    assert len(report.ignored_history_ids) == 2


def test_comparison_order_does_not_change_priority() -> None:
    first = _comparison([_row("a", 1.0, 0.0)])
    second = _comparison([_row("a", 0.2, 0.8, baseline_passed=False, candidate_passed=True)])
    left = prioritize(_suite(["a"]), [first, second], budget=1, seed=1)
    right = prioritize(_suite(["a"]), [second, first], budget=1, seed=99)
    assert left.selected[0].priority == right.selected[0].priority
    assert left.selected[0].candidate_variance == right.selected[0].candidate_variance
    assert left.selected[0].mean_drop == right.selected[0].mean_drop
    assert left.seed == 1
    assert right.seed == 99


def test_seed_is_recorded_and_does_not_change_the_suite_bytes(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    write_suite(_suite(["a", "b"], name="seeded"), suite_path)
    comparison = tmp_path / "cmp.json"
    write_json(
        _paired([("a", 1.0, True, 0.0, False), ("b", 1.0, True, 1.0, True)]),
        comparison,
    )
    first_out = tmp_path / "one.json"
    second_out = tmp_path / "two.json"
    first = select_suite(suite_path, [comparison], budget=1, out_path=first_out, seed=0)
    second = select_suite(suite_path, [comparison], budget=1, out_path=second_out, seed=-3)
    assert first.seed == 0
    assert second.seed == -3
    assert [item.case_id for item in first.selected] == [item.case_id for item in second.selected]
    assert first_out.read_bytes() == second_out.read_bytes()
    loaded = load_suite(first_out)
    assert loaded.name == "seeded-selected"
    assert [case.id for case in loaded.cases] == ["a"]


def test_refusals() -> None:
    suite = _suite(["a"])
    comparison = _comparison([_row("a", 1.0, 0.0)])
    with pytest.raises(ValueError, match="budget must be an integer >= 1"):
        prioritize(suite, [comparison], budget=0)
    with pytest.raises(ValueError, match="budget must be an integer >= 1"):
        prioritize(suite, [comparison], budget=-2)
    with pytest.raises(ValueError, match="budget must be an integer >= 1"):
        prioritize(suite, [comparison], budget=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed must be an integer"):
        prioritize(suite, [comparison], budget=1, seed=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="select needs at least one comparison"):
        prioritize(suite, [], budget=1)
    with pytest.raises(ValueError, match="sequence of comparisons"):
        prioritize(suite, "nope", budget=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="suite is empty"):
        prioritize(Suite(name="e", cases=[]), [comparison], budget=1)
    with pytest.raises(ValueError, match="suite name must be a non-empty string"):
        prioritize(Suite(name="", cases=suite.cases), [comparison], budget=1)
    with pytest.raises(ValueError, match="duplicate case id"):
        prioritize(
            Suite(name="d", cases=[suite.cases[0], suite.cases[0]]),
            [comparison],
            budget=1,
        )
    with pytest.raises(ValueError, match="repeats case id"):
        prioritize(suite, [_comparison([_row("a", 1.0, 0.0), _row("a", 1.0, 0.5)])], budget=1)
    with pytest.raises(ValueError, match="finite"):
        prioritize(suite, [_comparison([_row("a", 1.0, math.nan)])], budget=1)
    with pytest.raises(ValueError, match="finite"):
        prioritize(suite, [_comparison([_row("a", True, 0.0)])], budget=1)  # type: ignore[arg-type]
    bad_pass = _row("a", 1.0, 0.0)
    object.__setattr__(bad_pass, "baseline_passed", 1)
    with pytest.raises(ValueError, match="booleans"):
        prioritize(suite, [_comparison([bad_pass])], budget=1)


def test_case_objects_are_copied_and_the_suite_runs(tmp_path: Path) -> None:
    source = tmp_path / "suite.json"
    document = {
        "name": "weights",
        "description": "keep me",
        "owner": "qa",
        "scoring": {"weights": {"contains": 2}},
        "cases": [
            {
                "id": "keep",
                "input": "hello there",
                "reference": "hello",
                "comment": "extra",
                "tags": ["t"],
                "metadata": {"k": "v"},
                "assertions": [
                    {
                        "kind": "contains",
                        "value": "hello",
                        "weight": 2,
                        "params": {"case_sensitive": True},
                        "comment": "field",
                    }
                ],
            },
            {
                "id": "drop",
                "input": "other",
                "reference": "other",
                "assertions": [{"kind": "exact_match"}],
            },
        ],
    }
    source.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    comparison = tmp_path / "cmp.json"
    write_json(
        compare_runs(
            RunResult(
                "weights",
                "baseline",
                [
                    CaseResult("keep", "hello", {"contains": 1.0}, 1.0, True, 1.0),
                    CaseResult("drop", "other", {"exact_match": 1.0}, 1.0, True, 1.0),
                ],
            ),
            RunResult(
                "weights",
                "candidate",
                [
                    CaseResult("keep", "nope", {"contains": 0.0}, 0.0, False, 1.0),
                    CaseResult("drop", "other", {"exact_match": 1.0}, 1.0, True, 1.0),
                ],
            ),
            seed=13,
        ),
        comparison,
    )
    out = tmp_path / "out.json"
    report = select_suite(source, [comparison], budget=1, out_path=out, seed=13)
    assert [item.case_id for item in report.selected] == ["keep"]
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["name"] == "weights-selected"
    assert written["description"] == "keep me"
    assert written["owner"] == "qa"
    assert written["scoring"] == {"weights": {"contains": 2}}
    assert list(written["cases"][0]) == list(document["cases"][0])
    assert written["cases"][0] == document["cases"][0]
    loaded = load_suite(out)
    assert loaded.name == "weights-selected"
    assert loaded.scoring == ScoringConfig(weights={"contains": 2.0})
    assert loaded.cases[0].assertions[0].params == {"case_sensitive": True}
    assert loaded.cases[0].assertions[0].weight == 2.0
    provider = MockProvider(rules=[("hello there", "hello")], default="nope")
    result = run_suite(loaded, provider, "selected", prompt_template="{input}")
    assert [row.case_id for row in result.results] == ["keep"]
    assert result.results[0].passed is True
    assert result.results[0].score == 1.0


def test_format_families_and_unwritable_output(tmp_path: Path) -> None:
    yaml_source = tmp_path / "suite.yaml"
    yaml_source.write_text(
        "\n".join(
            [
                'name: "yaml-demo"',
                'description: "yaml cases"',
                "cases:",
                '  - id: "b"',
                '    input: "beta"',
                '    reference: "beta"',
                "    assertions:",
                '      - kind: "exact_match"',
                '  - id: "a"',
                '    input: "alpha"',
                '    reference: "alpha"',
                "    assertions:",
                '      - kind: "exact_match"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    comparison = tmp_path / "cmp.json"
    write_json(_paired([("b", 1.0, True, 0.0, False), ("a", 1.0, True, 1.0, True)]), comparison)
    yml_out = tmp_path / "picked.yml"
    report = select_suite(yaml_source, [comparison], budget=1, out_path=yml_out)
    loaded = load_suite(yml_out)
    assert loaded.name == "yaml-demo-selected"
    assert [case.id for case in loaded.cases] == ["b"]
    assert report.suite_name == "yaml-demo-selected"
    with pytest.raises(ValueError, match="same format family"):
        select_suite(yaml_source, [comparison], budget=1, out_path=tmp_path / "picked.json")

    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "b",
                        "input": "beta",
                        "reference": "beta",
                        "extra": 1,
                        "assertions": [{"kind": "exact_match"}],
                    }
                ),
                json.dumps(
                    {
                        "id": "a",
                        "input": "alpha",
                        "reference": "alpha",
                        "assertions": [{"kind": "exact_match"}],
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    jsonl_out = tmp_path / "picked.jsonl"
    jsonl_report = select_suite(jsonl, [comparison], budget=1, out_path=jsonl_out)
    parsed = json.loads(jsonl_out.read_text(encoding="utf-8").splitlines()[0])
    assert parsed["extra"] == 1
    assert parsed["id"] == "b"
    assert load_suite(jsonl_out).name == "picked"
    assert jsonl_report.suite_name == "rows-selected"

    csv_source = tmp_path / "sheet.csv"
    csv_source.write_text(
        'id,input,reference,note\nb,"be,ta",beta,keep\na,alpha,alpha,drop\n', encoding="utf-8"
    )
    csv_out = tmp_path / "picked.csv"
    select_suite(csv_source, [comparison], budget=1, out_path=csv_out)
    csv_loaded = load_suite(csv_out)
    assert [case.id for case in csv_loaded.cases] == ["b"]
    assert csv_loaded.cases[0].input == "be,ta"
    assert "keep" in csv_out.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="different path"):
        select_suite(yaml_source, [comparison], budget=1, out_path=yaml_source)
    directory = tmp_path / "blocked"
    directory.mkdir()
    with pytest.raises(ValueError, match="directory"):
        select_suite(yaml_source, [comparison], budget=1, out_path=directory)
    missing = tmp_path / "missing" / "out.yaml"
    with pytest.raises(ValueError, match="cannot write suite"):
        select_suite(yaml_source, [comparison], budget=1, out_path=missing)
    with pytest.raises(ValueError, match="same format family"):
        select_suite(yaml_source, [comparison], budget=1, out_path=tmp_path / "out.txt")


def test_json_report_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite_path = tmp_path / "suite.json"
    write_suite(_suite(["a", "z", "quiet"], name="cli"), suite_path)
    comparison = tmp_path / "cmp.json"
    write_json(_paired([("a", 1.0, True, 0.0, False), ("z", 1.0, True, 1.0, True)]), comparison)
    out = tmp_path / "selected.json"
    code = main(
        [
            "select",
            "--suite",
            str(suite_path),
            "--budget",
            "1",
            "--out",
            str(out),
            "--seed",
            "13",
            "--json",
            str(comparison),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["suite_name"] == "cli-selected"
    assert payload["seed"] == 13
    assert payload["selected"][0]["case_id"] == "a"
    assert payload["selected"][0]["history"] is True
    assert payload["note"] == SELECT_NOTE
    assert SELECT_LIMIT in payload["note"]
    assert load_suite(out).name == "cli-selected"

    text_code = main(
        [
            "select",
            "--suite",
            str(suite_path),
            "--budget",
            "3",
            "--out",
            str(tmp_path / "all.json"),
            str(comparison),
        ]
    )
    assert text_code == 0
    text = capsys.readouterr().out
    assert text.startswith("select\n")
    assert "history=true" in text
    assert "history=false" in text
    assert f"priority={payload['selected'][0]['priority']:.6f}" in text
    assert "does not change the gate" in text


def test_cli_refusals(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite_path = tmp_path / "suite.json"
    write_suite(_suite(["a"]), suite_path)
    comparison = tmp_path / "cmp.json"
    write_json(_paired([("a", 1.0, True, 0.0, False)]), comparison)
    out = tmp_path / "out.json"
    empty = tmp_path / "empty.json"
    empty.write_text('{"name": "empty", "cases": []}\n', encoding="utf-8")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    history = tmp_path / "history.jsonl"
    history.write_text(
        json.dumps({"command": "compare", "metrics": {"mean_score": 0.5}}) + "\n",
        encoding="utf-8",
    )
    cases = [
        (
            [
                "select",
                "--suite",
                str(suite_path),
                "--budget",
                "0",
                "--out",
                str(out),
                str(comparison),
            ],
            "budget",
        ),
        (
            ["select", "--suite", str(suite_path), "--budget", "1", "--out", str(out)],
            "at least one",
        ),
        (
            ["select", "--suite", str(empty), "--budget", "1", "--out", str(out), str(comparison)],
            "empty",
        ),
        (
            [
                "select",
                "--suite",
                str(suite_path),
                "--budget",
                "1",
                "--out",
                str(blocked),
                str(comparison),
            ],
            "directory",
        ),
        (
            [
                "select",
                "--suite",
                str(suite_path),
                "--budget",
                "1",
                "--out",
                str(tmp_path / "missing" / "out.json"),
                str(comparison),
            ],
            "cannot write",
        ),
        (
            [
                "select",
                "--suite",
                str(suite_path),
                "--budget",
                "1",
                "--out",
                str(out),
                str(history),
            ],
            "history",
        ),
    ]
    for argv, message in cases:
        assert main(argv) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("error:")
        assert message in captured.err


def test_shipped_example_matches_the_writers_and_the_formula(tmp_path: Path) -> None:
    write_example_inputs(tmp_path)
    for name in ("suite.json", "cmp_01.json", "cmp_02.json", "cmp_03.json"):
        assert (tmp_path / name).read_bytes() == (EXAMPLE / name).read_bytes()

    out = tmp_path / "selected.json"
    report = select_suite(
        EXAMPLE / "suite.json",
        [EXAMPLE / "cmp_01.json", EXAMPLE / "cmp_02.json", EXAMPLE / "cmp_03.json"],
        budget=3,
        out_path=out,
        seed=13,
    )
    assert out.read_bytes() == (EXAMPLE / "selected.json").read_bytes()
    assert [item.case_id for item in report.selected] == ["billing", "latency", "greeting"]
    billing, latency, greeting = report.selected
    assert billing.n_flips == 3 and billing.flip_rate == 1.0
    assert billing.n_regressed == 3
    assert latency.n_flips == 1 and latency.flip_rate == pytest.approx(1 / 3)
    assert latency.n_regressed == 2
    assert latency.mean_drop == 0.75
    assert greeting.priority == 0.0
    assert greeting.n_flips == 0 and greeting.n_regressed == 0
    assert billing.priority is not None and latency.priority is not None
    assert billing.priority > latency.priority > greeting.priority
    assert report.ignored_history_ids == ("retired",)
    assert report.n_with_history == 3
    assert report.n_without_history == 1
    assert report.n_suite == 4
    assert report.suite_name == "select-demo-selected"

    source = json.loads((EXAMPLE / "suite.json").read_text(encoding="utf-8"))
    selected = json.loads((EXAMPLE / "selected.json").read_text(encoding="utf-8"))
    assert selected["name"] == "select-demo-selected"
    assert selected["description"] == source["description"]
    by_id = {case["id"]: case for case in source["cases"]}
    assert [case["id"] for case in selected["cases"]] == ["billing", "latency", "greeting"]
    for case in selected["cases"]:
        assert case == by_id[case["id"]]
        assert list(case) == list(by_id[case["id"]])
    assert by_id["unseen"]["metadata"] == {"planted": "no-history"}

    loaded = load_suite(EXAMPLE / "selected.json")
    provider = MockProvider(
        rules=[(case.input, case.reference or "") for case in loaded.cases],
        default="nope",
    )
    result = run_suite(loaded, provider, "selected", prompt_template="{input}")
    assert result.pass_rate == 1.0
    assert [row.case_id for row in result.results] == ["billing", "latency", "greeting"]

    payload = selection_to_dict(report)
    assert payload["n_ignored_history"] == 1
    assert payload["selected"][0]["variance_component"] == billing.variance_component


def test_readme_quotes_the_example_command(tmp_path: Path) -> None:
    out = tmp_path / "selected.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "promptregress",
            "select",
            "--suite",
            "examples/select/suite.json",
            "--budget",
            "3",
            "--out",
            str(out),
            "--seed",
            "13",
            "examples/select/cmp_01.json",
            "examples/select/cmp_02.json",
            "examples/select/cmp_03.json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert out.read_bytes() == (EXAMPLE / "selected.json").read_bytes()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # SPEC.md is a local working document and is not part of the pushed
    # tree, so its assertions only run where the file exists.
    spec_path = ROOT / "SPEC.md"
    spec = spec_path.read_text(encoding="utf-8") if spec_path.exists() else None
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for line in completed.stdout.splitlines():
        if line.startswith("out="):
            continue
        assert line in readme
    assert "out=examples/select/selected.json" in readme
    assert "promptregress select" in readme
    if spec is not None:
        assert f"priority = {PRIORITY_FORMULA}" in spec
    assert f"priority = {PRIORITY_FORMULA}" in SELECT_NOTE
    limit_sentences = (
        "Selection is only as good as the history behind it.",
        (
            "A case that never regressed in the recorded comparisons may still "
            "regress on a new kind of change."
        ),
    )
    for sentence in limit_sentences:
        assert sentence in SELECT_LIMIT
        assert sentence in readme
        if spec is not None:
            assert sentence in spec
        assert sentence in changelog
        assert sentence in completed.stdout


def test_select_help_mentions_the_limit(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["select", "--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    assert "--budget" in text
    assert "--seed" in text
    assert "does not change the ranking" in text
    assert "never regressed" in text
