"""Cross-command regressions from findings 4–10 of the fourth review."""

import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from helpers import run_with
from promptregress.baseline import save_baseline
from promptregress.changepoint import detect_changepoint, format_changepoint
from promptregress.cli import main
from promptregress.gate import evaluate_gate
from promptregress.history import format_history_table
from promptregress.junit_xml import write_compare_junit
from promptregress.matrix import MatrixConfig, rank_runs, render_matrix_markdown
from promptregress.models import Comparison, DriftCase, DriftSummary
from promptregress.report import format_text, render_html, render_markdown, write_json
from promptregress.risk import load_comparison, train
from promptregress.stats import compare_runs


@pytest.fixture
def tiny() -> Comparison:
    tags = {"billing-1": ["billing"]}
    baseline = run_with([("billing-1", 1.0, True)], "baseline", tags=tags)
    candidate = run_with([("billing-1", 1 - 1e-8, True)], "candidate", tags=tags)
    drift = [DriftCase("billing-1", 1e-8)]
    return replace(
        compare_runs(baseline, candidate, n_boot=10),
        drift=DriftSummary("hash", 1e-8, 1e-8, drift, drift),
    )


def _value(text: str, field: str) -> float:
    return float(re.search(rf"\b{field}=([^\s]+)", text)[1])


def _cells(text: str, prefix: str) -> list[str]:
    line = next(line for line in text.splitlines() if line.startswith(f"| {prefix} |"))
    return [cell.strip() for cell in line.strip("|").split("|")]


def test_slice_precision(tiny: Comparison) -> None:
    line = next(line for line in format_text(tiny).splitlines() if line.startswith("- billing n="))
    assert [_value(line, name) for name in ("mean_diff", "ci_low", "ci_high")] == [
        tiny.mean_diff
    ] * 3
    assert "losses=1" in line


def test_drift_precision(tiny: Comparison) -> None:
    text = format_text(tiny, evaluate_gate(tiny, max_mean_drift=0))
    assert _value(text, "mean_drift") == _value(text, "max_drift") == 1e-8
    assert float(re.search(r"top_drift:\n- billing-1 (\S+)", text)[1]) == 1e-8


def test_markdown_precision(tiny: Comparison) -> None:
    text = render_markdown(tiny, evaluate_gate(tiny))
    for field in ("Mean diff (candidate - baseline)", "95% CI low", "95% CI high"):
        assert float(_cells(text, field)[1]) == tiny.mean_diff
    assert list(map(float, _cells(text, "billing")[2:5])) == [tiny.mean_diff] * 3
    row = next(line for line in text.splitlines() if line.endswith("| loss |"))
    assert list(map(float, row.split("|")[2:5])) == [1.0, 1 - 1e-8, tiny.mean_diff]
    assert float(_cells(text, "Mean drift")[1]) == 1e-8


def test_html_precision(tiny: Comparison) -> None:
    html = render_html(tiny, evaluate_gate(tiny))
    for field, value in (("delta", tiny.mean_diff), ("candidate-score", 1 - 1e-8), ("drift", 1e-8)):
        assert float(re.search(rf'data-{field}="([^"]+)"', html)[1]) == value
    assert ">-0.000000<" not in html


def test_junit_precision(tiny: Comparison, tmp_path: Path) -> None:
    path = tmp_path / "report.xml"
    write_compare_junit(path, tiny)
    note = ElementTree.parse(path).findtext(".//system-out")
    match = re.fullmatch(r"regressed: baseline (\S+) -> candidate (\S+)", note)
    assert tuple(map(float, match.groups())) == (1.0, 1 - 1e-8)


def test_history_precision(tiny: Comparison) -> None:
    record = {"command": "compare", "metrics": {"mean_diff": tiny.mean_diff, "gate_passed": False}}
    text = format_history_table([record])
    assert "gate=fail" in text
    assert _value(text, "mean_diff") == tiny.mean_diff


def test_changepoint_precision() -> None:
    records = [{"metrics": {"mean_score": score}} for score in [1.0] * 4 + [1 - 1e-8] * 4]
    report = detect_changepoint(records)
    text = format_changepoint(report)
    assert report.change_index == 4
    for name in ("before_mean", "after_mean", "mean_shift", "statistic"):
        assert _value(text, name) == getattr(report, name)


def test_matrix_precision_including_adjacent_nonleaders() -> None:
    configs = [("best", 1.0), ("second", 0.8), ("third", 0.8 - 1e-8), ("close", 1 - 1e-8)]
    runs = [
        (MatrixConfig(name, "mock", ""), run_with([("id", score, True)], name), "mock")
        for name, score in configs
    ]
    text = render_matrix_markdown(rank_runs(runs, suite_name="matrix", n_boot=10))
    for rank, (_, score) in enumerate(sorted(configs, key=lambda item: -item[1]), 1):
        row = _cells(text, str(rank))
        assert float(row[2]) == score
        if rank == 2:
            assert list(map(float, row[4:7])) == [score - 1.0] * 3


def _suite(path: Path, inputs: list[str]) -> Path:
    cases = [
        {"id": f"café{index}", "input": text, "assertions": [{"kind": "contains", "value": "YES"}]}
        for index, text in enumerate(inputs)
    ]
    path.write_text(json.dumps({"name": "suite", "cases": cases}), encoding="utf-8")
    return path


def _compare(suite: Path, report: Path, *, candidate: str = "YES") -> list[str]:
    rules = [["BASE", "YES"], ["REG", "NO"], ["CAND", candidate]]
    return [
        "compare",
        str(suite),
        "--provider",
        "mock",
        "--mock-rules",
        json.dumps(rules),
        "--baseline-template",
        "BASE {input}",
        "--candidate-template",
        "CAND {input}",
        "--report-json",
        str(report),
    ]


@pytest.mark.parametrize("budget", [25, 26])
def test_full_budget_selection_preserves_risk_holdout(tmp_path: Path, budget: int) -> None:
    suite = _suite(
        tmp_path / "suite.json", ["OK" if index % 5 == 4 else "REG" for index in range(25)]
    )
    original, selected, rerun = (tmp_path / name for name in ("a.json", "selected.json", "b.json"))
    assert main([*_compare(suite, original), "--no-history"]) == 1
    assert main(["risk", str(original)]) == 0
    assert (
        main(
            [
                "select",
                "--suite",
                str(suite),
                "--budget",
                str(budget),
                "--out",
                str(selected),
                str(original),
            ]
        )
        == 0
    )
    assert main([*_compare(selected, rerun), "--no-history"]) == 1
    assert main(["risk", str(rerun)]) == 0
    assert json.loads(selected.read_bytes())["cases"] == json.loads(suite.read_bytes())["cases"]


@pytest.mark.parametrize("command", ["pareto", "audit"])
def test_pool_default_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    suite = _suite(tmp_path / "suite.json", ["OK"] * 5)
    first, second = (tmp_path / name / "report.json" for name in ("first", "second"))
    assert main([*_compare(suite, first), "--no-history"]) == 0
    assert main([*_compare(suite, second, candidate="NO"), "--no-history"]) == 1
    capsys.readouterr()
    assert main([command, str(first), str(second), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    names = (
        payload["configs"] if command == "audit" else [row["name"] for row in payload["variants"]]
    )
    assert len(set(names)) == 4


def test_ascii_stdout_keeps_gate_and_history(tmp_path: Path) -> None:
    suite = _suite(tmp_path / "suite.json", ["REG"])
    report = tmp_path / "report.json"
    env = {
        **os.environ,
        "PYTHONIOENCODING": "ascii:strict",
        "LC_ALL": "C",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "promptregress", *_compare(suite, report)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
    )
    assert proc.returncode == 1, proc.stderr
    stdout = proc.stdout.decode("ascii")
    assert "gate_passed=false" in stdout
    assert r"caf\xe90" in stdout
    assert json.loads(report.read_bytes())["per_case"][0]["case_id"] == "café0"
    history = json.loads((tmp_path / ".promptregress/history.jsonl").read_bytes())
    assert history["metrics"]["gate_passed"] is False


@pytest.mark.parametrize("command", [["baseline", "list"], ["flaky", "suite"]])
def test_unprintable_mtime_is_isolated(tmp_path, monkeypatch, capsys, command) -> None:
    monkeypatch.chdir(tmp_path)
    run = run_with([("case", 1.0, True)], "test")
    for name in ("bad", "good"):
        save_baseline(run, name)
    real_stat = Path.stat

    def stat(path, *args, **kwargs):
        info = real_stat(path, *args, **kwargs)
        if path.name == "bad.json":
            return SimpleNamespace(
                st_mode=info.st_mode, st_mtime=253402300800.0, st_mtime_ns=253402300800 * 10**9
            )
        return info

    monkeypatch.setattr(Path, "stat", stat)
    assert main(command) == 0
    output = capsys.readouterr()
    assert "bad.json" in output.err
    assert ("good" if command[0] == "baseline" else "case") in output.out


def test_risk_fits_length_only_signal_from_compare(tmp_path: Path) -> None:
    long = "REG " + "雪" * 400
    suite = _suite(tmp_path / "suite.json", [long, "OK"] * 10)
    report = tmp_path / "report.json"
    assert main([*_compare(suite, report), "--no-history"]) == 1
    model = train([load_comparison(report)])
    index = model.feature_names.index("input_length")
    assert model.holdout_auc == 1.0
    assert model.means[index] == (len(long) + 2) / 2
    assert model.coefficients[index] > 0
    assert "case_input" not in json.loads(report.read_bytes())["baseline"]["results"][0]


@pytest.mark.parametrize("outlier", [0, 16], ids=["train-outlier", "holdout-outlier"])
def test_extreme_finite_scores_compare_bayes_and_train(tmp_path: Path, outlier: int) -> None:
    def comparison_at(scale: float) -> Comparison:
        baseline = [-1.0] * 20
        baseline[outlier] = 1.0
        candidate = [0.0, -1.5] * 10
        runs = [
            run_with(
                [(str(index), value * scale, value >= 0) for index, value in enumerate(values)],
                name,
            )
            for values, name in ((baseline, "baseline"), (candidate, "candidate"))
        ]
        return compare_runs(*runs, n_boot=20, seed=1)

    normal, extreme = comparison_at(1), comparison_at(1e308)
    assert extreme.mean_diff == pytest.approx(normal.mean_diff * 1e308)
    report = tmp_path / "report.json"
    write_json(extreme, report)
    assert main(["bayes", str(report)]) == 0
    loaded = load_comparison(report)
    expected, actual = train([normal]), train([loaded])
    assert actual.coefficients == pytest.approx(expected.coefficients)
    assert actual.holdout_auc == expected.holdout_auc
