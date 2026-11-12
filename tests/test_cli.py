"""CLI init, run, compare exit codes, and demo."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import promptregress
from promptregress.cli import main
from promptregress.loading import load_suite

ROOT = Path(__file__).resolve().parents[1]


def _suite(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [
                    {
                        "id": "a",
                        "input": "alpha",
                        "reference": "alpha",
                        "assertions": [{"kind": "exact_match"}],
                    },
                    {
                        "id": "b",
                        "input": "beta",
                        "reference": "beta",
                        "assertions": [{"kind": "exact_match"}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_version_export() -> None:
    assert promptregress.__version__
    assert promptregress.Suite.__name__ == "Suite"
    assert promptregress.GateVerdict.__name__ == "GateVerdict"


def test_help_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    for command in (
        "init",
        "run",
        "compare",
        "matrix",
        "ablate",
        "history",
        "trend",
        "changepoint",
        "baseline",
        "flaky",
        "schema",
        "validate",
        "audit",
        "calibrate",
        "irt",
        "risk",
        "pareto",
        "synth",
        "power",
        "doctor",
        "demo",
    ):
        assert command in help_text
    with pytest.raises(SystemExit) as compare_help:
        main(["compare", "--help"])
    assert compare_help.value.code == 0
    compare_text = capsys.readouterr().out
    assert "candidate pass rate" in compare_text
    assert "upper bound" in compare_text
    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0
    assert promptregress.__version__ in capsys.readouterr().out


def test_init_writes_a_loadable_suite(tmp_path: Path) -> None:
    directory = tmp_path / "proj"
    assert main(["init", str(directory)]) == 0
    suite = load_suite(directory / "suite.json")
    assert suite.name == "example"
    assert len(suite.cases) >= 1
    config = (directory / "promptregress.yaml").read_text(encoding="utf-8")
    assert "baseline_template" in config
    assert "max_regression" in config


def test_run_echo_writes_a_passing_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = _suite(tmp_path / "suite.json")
    out = tmp_path / "echo.json"
    code = main(
        [
            "run",
            str(suite),
            "--provider",
            "echo",
            "--config-name",
            "echo",
            "--template",
            "{input}",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["config_name"] == "echo"
    assert payload["suite_name"] == "tiny"
    assert payload["pass_rate"] == 1.0
    assert payload["mean_score"] == 1.0
    assert "pass_rate=1.000000" in capsys.readouterr().out


def test_run_missing_suite_exits_2(tmp_path: Path) -> None:
    code = main(
        [
            "run",
            str(tmp_path / "missing.json"),
            "--provider",
            "echo",
            "--config-name",
            "x",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert code == 2


def test_compare_exit_codes_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite = _suite(tmp_path / "suite.json")
    report_json = tmp_path / "cmp.json"
    report_md = tmp_path / "cmp.md"
    passed = main(
        [
            "compare",
            str(suite),
            "--provider",
            "echo",
            "--baseline-template",
            "{input}",
            "--candidate-template",
            "{input}",
            "--max-regression",
            "0",
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
        ]
    )
    assert passed == 0
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["gate"]["passed"] is True
    assert payload["mean_diff"] == 0.0
    assert "PASS" in report_md.read_text(encoding="utf-8")

    failed = main(
        [
            "compare",
            str(suite),
            "--provider",
            "echo",
            "--baseline-template",
            "{input}",
            "--candidate-template",
            "nope {input}",
            "--max-regression",
            "0",
            "--min-pass-rate",
            "1",
            "--report-html",
            str(tmp_path / "report.html"),
        ]
    )
    assert failed == 1
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "tiny" in html
    assert "baseline" in html
    assert "candidate" in html
    assert "<style>" in html
    assert 'class="verdict fail"' in html
    assert "FAIL regression" in html
    assert "Per-case results" in html
    captured = capsys.readouterr()
    assert "not implemented" not in captured.err
    assert "skipped" not in captured.err


def test_compare_ci_method_bca(tmp_path: Path) -> None:
    suite = _suite(tmp_path / "suite.json")
    code = main(
        [
            "compare",
            str(suite),
            "--provider",
            "echo",
            "--baseline-template",
            "{input}",
            "--candidate-template",
            "{input}",
            "--ci-method",
            "bca",
            "--max-regression",
            "0",
        ]
    )
    assert code == 0


def test_changepoint_command_refuses_empty_and_reports_a_shift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as help_exit:
        main(["changepoint", "--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "--seed" in help_text
    assert "--json" in help_text
    assert "--out" in help_text
    assert "--file" in help_text
    assert "does not change the gate" in help_text

    missing = main(["changepoint"])
    assert missing == 2
    assert "empty history" in capsys.readouterr().err

    path = tmp_path / "history.jsonl"
    for index, score in enumerate([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "timestamp": f"t{index}",
                        "command": "run",
                        "suite": "tiny",
                        "metrics": {"mean_score": score},
                    }
                )
                + "\n"
            )
    short = main(["changepoint", "--file", str(path)])
    assert short == 2
    assert "at least 8" in capsys.readouterr().err

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": "t7",
                    "command": "run",
                    "suite": "tiny",
                    "metrics": {"mean_score": 1.0},
                }
            )
            + "\n"
        )
    out = tmp_path / "nested" / "changepoint.json"
    code = main(
        [
            "changepoint",
            "--file",
            str(path),
            "--json",
            "--out",
            str(out),
            "--seed",
            "13",
            "--permutations",
            "50",
        ]
    )
    assert code == 0
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == payload
    assert payload["change_index"] == 3
    assert payload["before_mean"] == 0.0
    assert payload["after_mean"] == 1.0
    assert payload["mean_shift"] == 1.0
    assert payload["seed"] == 13
    assert payload["permutations"] == 50
    assert payload["before_runs"][2]["timestamp"] == "t2"
    assert payload["after_runs"][0]["timestamp"] == "t3"
    assert "not proof of cause" in payload["note"]
    assert "does not change the gate" in payload["note"]


def test_demo_module_entrypoint() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "promptregress", "demo"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 1
    assert "mean_diff=-0.214286" in completed.stdout
    assert "losses=3" in completed.stdout
    assert "gate_passed=false" in completed.stdout
    assert "Answer briefly." in completed.stdout
