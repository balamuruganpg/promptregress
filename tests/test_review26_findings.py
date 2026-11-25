"""Printed numbers, history types, and report agreement from the v0.15.4 review.

Each finding test fails on v0.15.4 and locks the contract added in 0.15.5.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

from helpers import comparison_with, run_with
from promptregress.baseline import BaselineError, ensure_baseline_matches, save_baseline
from promptregress.calibration import calibrate, format_calibration
from promptregress.changepoint import detect_changepoint
from promptregress.cli import main
from promptregress.flaky import _population_sd
from promptregress.gate import evaluate_gate
from promptregress.history import format_history_table, suite_points
from promptregress.models import (
    AssertionSpec,
    Case,
    Comparison,
    CostEstimate,
    DriftCase,
    DriftSummary,
    Outcome,
    PairedRow,
    Suite,
)
from promptregress.power import achieved_power, load_report_diffs
from promptregress.report import format_text, write_json
from promptregress.risk import load_comparison
from promptregress.stats import compare_runs

_TINY = 1.0000000005838672e-07
_LATENCY_PAIR = re.compile(
    r"p95 latency (?P<obs>-?\d+\.\d+)ms (?:exceeds|is within) budget (?P<budget>-?\d+\.\d+)ms"
)
_COST_PAIR = re.compile(
    r"cost \$(?P<obs>-?\d+\.\d+) (?:exceeds|is within) budget \$(?P<budget>-?\d+\.\d+)"
)
_DRIFT_PAIR = re.compile(
    r"mean_drift (?P<obs>-?\d+\.\d+), max_drift (?P<max>-?\d+\.\d+), "
    r"max_mean_drift (?P<budget>-?\d+\.\d+)"
)
_CI_TOKEN = re.compile(r"ci_low (?P<value>-?\d+\.\d+)")
_DIFF_TOKEN = re.compile(r"diff=(?P<value>\S+)")


def _field(text: str, name: str) -> str:
    prefix = f"{name}="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    raise AssertionError(f"{name} missing")


def _agrees(token: str, value: float, other: float) -> None:
    parsed = float(token)
    assert parsed == value
    assert (value > other) == (parsed > other)
    assert (value < other) == (parsed < other)
    assert (value == other) == (parsed == other)


def _row(case_id: str, diff: float, outcome: Outcome) -> PairedRow:
    return PairedRow(
        case_id=case_id,
        baseline_output="1.0000001",
        candidate_output="1",
        baseline_score=1.0 - diff,
        candidate_score=1.0,
        diff=diff,
        baseline_passed=True,
        candidate_passed=True,
        outcome=outcome,
    )


def _tiny_comparison(diff: float, outcome: Outcome) -> Comparison:
    base = comparison_with(
        ci_low=diff,
        ci_high=diff,
        mean_diff=diff,
        candidate_rows=[("c0", 1.0, True)],
        baseline_rows=[("c0", 1.0 - diff, True)],
    )
    return replace(base, per_case=[_row("c0", diff, outcome)])


def _history_line(metrics: dict[str, object], **extra: object) -> str:
    record: dict[str, object] = {
        "timestamp": "t",
        "command": "run",
        "suite": "s",
        "metrics": metrics,
    }
    record.update(extra)
    return json.dumps(record) + "\n"


def _equal_report(path: Path) -> None:
    baseline = run_with([("a", 1.0, True), ("b", 1.0, True)], "baseline")
    candidate = run_with([("a", 1.0, True), ("b", 1.0, True)], "candidate")
    write_json(compare_runs(baseline, candidate, n_boot=20, seed=1), path)


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_significance_and_scoreboard_show_a_nonzero_bound() -> None:
    """A PASS above 0 must not print ci_low 0.000000, and a win must show its diff."""
    positive = _tiny_comparison(_TINY, "win")
    text = format_text(positive)
    for name in ("mean_diff", "ci_low", "ci_high"):
        token = _field(text, name)
        assert token != "0.000000"
        _agrees(token, _TINY, 0.0)
    verdict = evaluate_gate(positive, max_regression=1.0, require_significant=True)
    assert verdict.passed
    match = _CI_TOKEN.search(
        next(reason for reason in verdict.reasons if "significance:" in reason)
    )
    assert match is not None
    assert match.group("value") != "0.000000"
    _agrees(match.group("value"), _TINY, 0.0)
    diff = _DIFF_TOKEN.search(next(line for line in text.splitlines() if line.startswith("- c0 ")))
    assert diff is not None
    assert diff.group("value") != "0.000000"
    _agrees(diff.group("value"), _TINY, 0.0)
    assert text.splitlines()[-1].endswith(" win")

    negative = _tiny_comparison(-_TINY, "loss")
    text = format_text(negative)
    for name in ("mean_diff", "ci_low", "ci_high"):
        token = _field(text, name)
        assert token != "-0.000000"
        _agrees(token, -_TINY, 0.0)
    failed = evaluate_gate(negative, max_regression=1.0, require_significant=True)
    assert not failed.passed
    match = _CI_TOKEN.search(next(reason for reason in failed.reasons if "significance:" in reason))
    assert match is not None
    assert match.group("value") != "-0.000000"
    _agrees(match.group("value"), -_TINY, 0.0)
    assert "loss" in next(line for line in text.splitlines() if line.startswith("- c0 "))

    zero = format_text(
        comparison_with(ci_low=0.0, ci_high=0.0, mean_diff=0.0, candidate_rows=[("z", 1.0, True)])
    )
    assert _field(zero, "mean_diff") == "0.000000"
    assert _field(zero, "ci_low") == "0.000000"
    assert _field(zero, "ci_high") == "0.000000"


def _assert_compared(token_left: str, token_right: str, left: float, right: float) -> None:
    assert float(token_left) == left
    assert float(token_right) == right
    assert (left > right) == (float(token_left) > float(token_right))


def test_latency_cost_and_drift_reasons_match_the_comparison() -> None:
    observed = 1.0
    failing_budget = 0.9999995
    passing_budget = 1.0000000000222045
    comparison = comparison_with(
        ci_low=0.0,
        ci_high=0.0,
        mean_diff=0.0,
        candidate_rows=[("alpha", 1.0, True)],
    )
    failed = evaluate_gate(comparison, max_p95_latency_ms=failing_budget)
    reason = next(item for item in failed.reasons if item.startswith("FAIL latency:"))
    match = _LATENCY_PAIR.search(reason)
    assert match is not None
    _assert_compared(match.group("obs"), match.group("budget"), observed, failing_budget)

    passed = evaluate_gate(comparison, max_p95_latency_ms=passing_budget)
    reason = next(item for item in passed.reasons if item.startswith("PASS latency:"))
    match = _LATENCY_PAIR.search(reason)
    assert match is not None
    _assert_compared(match.group("obs"), match.group("budget"), observed, passing_budget)

    priced = replace(
        comparison, candidate=replace(comparison.candidate, cost=CostEstimate("mock", observed))
    )
    cost_budget = 0.9999999999777955
    failed = evaluate_gate(priced, max_cost_usd=cost_budget)
    reason = next(item for item in failed.reasons if item.startswith("FAIL cost:"))
    match = _COST_PAIR.search(reason)
    assert match is not None
    _assert_compared(match.group("obs"), match.group("budget"), observed, cost_budget)

    drift_budget = 0.9999999999666933
    drifted = replace(
        comparison,
        drift=DriftSummary(
            embedder="hashing",
            mean_drift=observed,
            max_drift=observed,
            per_case=[DriftCase("alpha", observed)],
            top=[DriftCase("alpha", observed)],
        ),
    )
    failed = evaluate_gate(drifted, max_regression=1.0, max_mean_drift=drift_budget)
    reason = next(item for item in failed.reasons if item.startswith("FAIL drift:"))
    match = _DRIFT_PAIR.search(reason)
    assert match is not None
    _assert_compared(match.group("obs"), match.group("budget"), observed, drift_budget)


def _threshold_token(text: str) -> str:
    lines = [line for line in text.splitlines() if line.startswith("threshold=")]
    assert len(lines) == 1
    return lines[0].split("=", 1)[1]


def test_calibration_threshold_paste_reproduces_the_bins() -> None:
    cut = 0.5000000000222045
    report = calibrate([0.5, 0.5], [0.5, 1.0], threshold=cut)
    text = format_calibration(report)
    token = _threshold_token(text)
    printed = float(token)
    assert token != "0.500000"
    assert printed == cut
    assert (0.5 >= printed) == (0.5 >= cut)
    again = calibrate(report.judge_scores, report.human_scores, threshold=printed)
    assert (again.both_pass, again.judge_only, again.human_only, again.both_fail) == (
        report.both_pass,
        report.judge_only,
        report.human_only,
        report.both_fail,
    )
    assert report.both_fail == 1
    assert report.human_only == 1
    ordinary = _threshold_token(format_calibration(calibrate([0.5], [0.5], threshold=0.5)))
    assert ordinary == "0.500000"


def test_history_does_not_treat_the_string_false_as_a_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "command": "compare",
                "suite": "s",
                "metrics": {"mean_score": 0.5, "gate_passed": "false"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["history", "--file", str(path)]) == 0
    captured = capsys.readouterr()
    assert "mean_score=0.500000" in captured.out
    assert "gate=pass" not in captured.out
    assert "gate=fail" not in captured.out
    assert "gate_passed" in captured.err
    assert "boolean" in captured.err

    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "command": "compare",
                "suite": "s",
                "metrics": {"mean_score": 0.5, "gate_passed": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["history", "--file", str(path)]) == 0
    captured = capsys.readouterr()
    assert "gate=fail" in captured.out
    assert captured.err == ""

    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "command": "compare",
                "suite": "s",
                "metrics": {"mean_score": 0.5, "gate_passed": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["history", "--file", str(path)]) == 0
    captured = capsys.readouterr()
    assert "mean_score=0.500000" in captured.out
    assert "gate=" not in captured.out
    assert "boolean" in captured.err


@pytest.mark.parametrize("mutate", ["ghost", "truncate", "edit", "anonymous"])
def test_power_and_bayes_count_the_rebuilt_pairs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mutate: str
) -> None:
    path = tmp_path / "report.json"
    _equal_report(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutate == "ghost":
        payload["per_case"].append({"case_id": "ghost", "diff": -1.0, "outcome": "loss"})
    elif mutate == "truncate":
        payload["per_case"] = payload["per_case"][:1]
    elif mutate == "edit":
        payload["per_case"][0]["diff"] = -1.0
        payload["mean_diff"] = -0.5
    else:
        payload["per_case"].append({"diff": 5.0})
    _write_payload(path, payload)

    comparison = load_comparison(path)
    diffs = load_report_diffs(path)
    assert diffs == [row.diff for row in comparison.per_case]
    assert len(diffs) == 2
    assert main(["power", "--from-report", str(path)]) == 0
    power = capsys.readouterr()
    assert power.err == ""
    assert "observed_dz=0.000000" in power.out
    assert "required_n=undefined" in power.out
    assert main(["bayes", str(path), "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["n"] == len(diffs)
    assert body["n11"] == 2
    assert body["observed_diff"] == 0.0


def test_flaky_overflow_is_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    save_baseline(run_with([("a", 0.0, False)], "r1", suite="pair"), "r1", tmp_path)
    save_baseline(run_with([("a", 1e308, True)], "r2", suite="pair"), "r2", tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["flaky", "pair"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: population standard deviation is not a finite number\n"
    with pytest.raises(ValueError, match="population standard deviation is not a finite number"):
        _population_sd([0.0, 1e308])
    assert _population_sd([1e308, 1e308]) == 0.0


def test_power_accepts_a_report_write_json_can_emit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "overflow.json"
    baseline = run_with([("a", 1e308, True), ("b", 1.0, True)], "baseline")
    candidate = run_with([("a", -1e308, False), ("b", 1.0, True)], "candidate")
    write_json(compare_runs(baseline, candidate, n_boot=20, seed=1), path)
    text = path.read_text(encoding="utf-8")
    assert "-Infinity" in text
    assert "NaN" not in text
    diffs = load_report_diffs(path)
    assert any(math.isinf(value) for value in diffs)
    assert main(["power", "--from-report", str(path)]) == 0
    power = capsys.readouterr()
    assert power.err == ""
    assert "diffs must be finite" not in power.out
    assert main(["bayes", str(path), "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["n"] == len(diffs)
    assert body["n11"] == 1
    assert body["n10"] == 1

    edited = tmp_path / "edited.json"
    _equal_report(edited)
    payload = json.loads(edited.read_text(encoding="utf-8"))
    payload["per_case"][0]["diff"] = float("-inf")
    payload["candidate"]["results"][0]["passed"] = False
    _write_payload(edited, payload)
    assert "-Infinity" in edited.read_text(encoding="utf-8")
    assert main(["power", "--from-report", str(edited)]) == 0
    power = capsys.readouterr()
    assert power.err == ""
    assert "observed_dz=0.000000" in power.out
    assert main(["bayes", str(edited), "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["n"] == 2
    assert body["n10"] == 1

    shorthand = tmp_path / "shorthand.json"
    shorthand.write_text(
        json.dumps({"per_case": [{"diff": float("inf")}]}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="diffs must be finite"):
        load_report_diffs(shorthand)


def test_huge_integers_name_the_row(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text(
        json.dumps(
            {"timestamp": "t", "command": "run", "suite": "s", "metrics": {"mean_score": 10**400}}
        )
        + "\n",
        encoding="utf-8",
    )
    commands = [
        ["history", "--file", str(path)],
        ["trend", "s", "--file", str(path), "--out", str(tmp_path / "trend.html")],
        ["changepoint", "--file", str(path)],
    ]
    for args in commands:
        assert main(args) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert str(path) in captured.err
        assert ":1:" in captured.err
        assert "mean_score" in captured.err
        assert "finite" in captured.err
        assert "int too large" not in captured.err

    data = tmp_path / "human.jsonl"
    data.write_text(
        json.dumps({"input": "q", "output": "a", "human_score": 10**400}) + "\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "calibrate",
                str(data),
                "--rubric",
                "grade",
                "--judge-provider",
                "mock",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(data) in captured.err
    assert ":1:" in captured.err
    assert "human_score" in captured.err
    assert "finite" in captured.err
    assert "int too large" not in captured.err

    with pytest.raises(ValueError, match=r"record 0 \(t\): mean_score must be a finite number"):
        detect_changepoint([{"timestamp": "t", "metrics": {"mean_score": 10**400}}])
    with pytest.raises(ValueError, match=r"record \(t\): mean_score must be a finite number"):
        suite_points(
            [{"suite": "s", "timestamp": "t", "metrics": {"mean_score": 10**400}}],
            "s",
        )
    report = tmp_path / "diffs.json"
    report.write_text(json.dumps({"per_case": [{"diff": 10**400}]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"per_case\[0\]\.diff must be a finite number"):
        load_report_diffs(report)
    with pytest.raises(ValueError, match="diffs must be finite"):
        achieved_power([10**400, 1.0])

    infinity = tmp_path / "inf.jsonl"
    infinity.write_text(
        '{"timestamp":"t","command":"run","suite":"s","metrics":{"mean_score":1e309}}\n',
        encoding="utf-8",
    )
    assert main(["history", "--file", str(infinity)]) == 0
    captured = capsys.readouterr()
    assert "mean_score=inf" in captured.out
    assert captured.err == ""
    assert main(["changepoint", "--file", str(infinity)]) == 2
    assert "non-finite" in capsys.readouterr().err


def test_hyphen_baseline_names_work_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [
                    {
                        "id": "a",
                        "input": "alpha",
                        "reference": "yes",
                        "assertions": [{"kind": "exact_match"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    argv = [
        "run",
        str(suite),
        "--provider",
        "mock",
        "--mock-default",
        "yes",
        "--config-name",
        "c",
        "--out",
        str(tmp_path / "run.json"),
        "--save-baseline",
        "-foo",
        "--no-history",
    ]
    assert main(argv) == 0
    assert argv[argv.index("--save-baseline") + 1] == "-foo"
    saved = capsys.readouterr().out
    assert "saved baseline -foo" in saved
    assert (tmp_path / ".promptregress" / "baselines" / "-foo.json").is_file()
    assert main(["baseline", "list"]) == 0
    assert "-foo" in capsys.readouterr().out
    assert (
        main(
            [
                "compare",
                str(suite),
                "--provider",
                "mock",
                "--mock-default",
                "yes",
                "--candidate-template",
                "{input}",
                "--baseline-name",
                "-foo",
                "--no-history",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["baseline", "delete", "-foo"]) == 0
    assert capsys.readouterr().out == "deleted baseline -foo\n"

    with pytest.raises(SystemExit) as help_exit:
        main(["baseline", "delete", "-h"])
    assert help_exit.value.code == 0
    assert "usage:" in capsys.readouterr().out

    with pytest.raises(SystemExit) as missing:
        main(
            [
                "run",
                str(suite),
                "--provider",
                "mock",
                "--mock-default",
                "yes",
                "--config-name",
                "c",
                "--out",
                str(tmp_path / "run.json"),
                "--save-baseline",
                "--no-history",
            ]
        )
    assert missing.value.code == 2
    assert not (tmp_path / ".promptregress" / "baselines" / "--no-history.json").exists()


def test_differing_case_ids_are_quoted() -> None:
    def case(case_id: str) -> Case:
        return Case(case_id, "x", "y", [AssertionSpec("exact_match")])

    stored = run_with([("a", 1.0, True), ("b", 1.0, True)], "base", suite="s")
    with pytest.raises(BaselineError) as trailing:
        ensure_baseline_matches(stored, Suite("s", [case("a "), case("b")]), "plain")
    assert "first differing case id: 'a '" in str(trailing.value)
    with pytest.raises(BaselineError) as extra:
        ensure_baseline_matches(stored, Suite("s", [case("a"), case("b"), case("c")]), "extra")
    assert "first differing case id: 'c'" in str(extra.value)


@pytest.mark.parametrize("mean_score", ["0.85", True, None])
def test_mistyped_mean_score_is_not_a_blank_history_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mean_score: object
) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text(_history_line({"mean_score": mean_score}), encoding="utf-8")
    assert main(["history", "--file", str(path)]) == 0
    captured = capsys.readouterr()
    assert [line for line in captured.out.splitlines() if line][-1] == "(no runs)"
    assert "mean_score" in captured.err
    assert "finite" in captured.err
    assert main(["trend", "s", "--file", str(path), "--out", str(tmp_path / "trend.html")]) == 2
    trend = capsys.readouterr()
    assert trend.out == ""
    assert "warning:" in trend.err
    assert "no history rows" in trend.err
    assert main(["changepoint", "--file", str(path)]) == 2
    changed = capsys.readouterr()
    assert "warning:" in changed.err
    assert "record 0 (t): mean_score must be a finite number" in changed.err


def test_empty_metrics_still_print_a_row_and_summary_order_is_stable() -> None:
    table = format_history_table([json.loads(_history_line({}))])
    assert "(no runs)" not in table
    assert table.splitlines()[1].startswith("t")
    ordered = format_history_table(
        [
            {
                "timestamp": "t",
                "command": "compare",
                "suite": "s",
                "metrics": {
                    "pairwise_win_rate": 0.25,
                    "gate_passed": False,
                    "mean_diff": -0.1,
                    "mean_score": 0.5,
                },
            }
        ]
    )
    assert "mean_score=0.500000 mean_diff=-0.100000 gate=fail pairwise_win_rate=0.250000" in ordered


def test_first_mistyped_metric_is_the_only_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text(
        _history_line({"mean_score": "0.85", "gate_passed": "false"}),
        encoding="utf-8",
    )
    assert main(["history", "--file", str(path)]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines()[-1] == "(no runs)"
    assert "metrics.mean_score must be a finite number" in captured.err
    assert "gate_passed" not in captured.err
