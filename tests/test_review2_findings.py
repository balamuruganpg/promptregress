"""Regression tests for bugs found in the independent v0.2.3 review cycle."""

from __future__ import annotations

import itertools
import json
import os
import random
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from promptregress.calibration import parse_preference, position_bias_audit, verbosity_bias
from promptregress.cli import main
from promptregress.doctor import Check
from promptregress.history import append_history, load_history
from promptregress.matrix import MatrixConfig, MatrixError, load_matrix, rank_runs
from promptregress.models import AssertionSpec, Case, CaseResult, RunResult
from promptregress.scorers import (
    _trace_actual,
    _trace_expected,
    _trace_matched,
    _trace_steps_match,
    score_case,
)

ROOT = Path(__file__).resolve().parents[1]
TRACE_DP_SEED = 20261111
TRACE_DP_TRIALS = 400


def _action_compare_script() -> str:
    text = (ROOT / "action.yml").read_text(encoding="utf-8")
    match = re.search(r"python - <<'PY'\n(.*)\n        PY", text, re.S)
    assert match is not None
    lines = match.group(1).splitlines()
    return textwrap.dedent(
        "\n".join(line[8:] if line.startswith("        ") else line for line in lines)
    )


def _run_result(name: str, scores: list[float]) -> RunResult:
    results = [
        CaseResult(
            case_id=f"c{index}",
            output="o",
            scores={"exact_match": score},
            score=score,
            passed=score >= 0.5,
            latency_ms=1.0,
        )
        for index, score in enumerate(scores)
    ]
    return RunResult(suite_name="s", config_name=name, results=results)


def test_parse_preference_identical_outputs_are_a_tie() -> None:
    """When both sides are the same text, echoing that text is a tie, not an error.

    ``_side`` required ``raw != second`` / ``raw != first`` before accepting an
    exact-text match, so identical outputs raised ``ValueError`` even though the
    reply was literally one of the two outputs (both of them). That crashed
    position and verbosity audits on equal pairs.
    """
    assert parse_preference("hello", "hello", "hello") == "tie"
    assert parse_preference("a\n", "a\n", "a\n") == "tie"
    assert parse_preference("a\n", "a\n", "a") == "tie"
    assert parse_preference(" hello ", "hello", "hello") == "second"
    assert parse_preference(" hello", "hello ", "hello") == "tie"
    assert parse_preference("", "", "") == "tie"

    def prefer_left(first: str, second: str) -> str:
        return first

    audit = position_bias_audit([("same text", "same text")], prefer_left)
    assert audit.n_pairs == 1
    assert audit.n_flips == 0
    assert audit.flagged is False

    bias = verbosity_bias([("same text", "same text")], prefer_left)
    assert bias.preferences == [0.0]


def _brute_trace_matched(
    expected: list[tuple[str, dict[str, str]]],
    actual: list[object],
) -> int:
    """Largest in-order pairing by enumerating index combinations."""
    parsed = [_trace_actual(step) for step in actual]
    n_expected = len(expected)
    n_actual = len(parsed)
    for width in range(min(n_expected, n_actual), -1, -1):
        for expected_idx in itertools.combinations(range(n_expected), width):
            for actual_idx in itertools.combinations(range(n_actual), width):
                if all(
                    _trace_steps_match(expected[expected_idx[offset]], parsed[actual_idx[offset]])
                    for offset in range(width)
                ):
                    return width
    return 0


def _random_trace_args(rng: random.Random) -> dict[str, object]:
    args: dict[str, object] = {}
    if rng.random() < 0.75:
        args["q"] = rng.choice(["", "a", "ab", "abc", "x", "xy", "xyz"])
    if rng.random() < 0.35:
        args["n"] = rng.choice([1, 2, True, None, "1", 1.0])
    if rng.random() < 0.2:
        args["extra"] = "z"
    return args


def test_trace_match_dp_matches_brute_force_on_random_traces() -> None:
    """``_trace_matched`` must equal a brute-force LCS-style checker."""
    rng = random.Random(TRACE_DP_SEED)
    tools = ("a", "b", "c", "search")
    for trial in range(TRACE_DP_TRIALS):
        n_expected = rng.randint(0, 5)
        n_actual = rng.randint(0, 6)
        expected_raw = [
            {"tool": rng.choice(tools), "args": _random_trace_args(rng)} for _ in range(n_expected)
        ]
        actual: list[object] = []
        for _ in range(n_actual):
            if rng.random() < 0.12:
                actual.append(rng.choice([1, "x", None, {"tool": 3}, {"args": {}}]))
                continue
            actual.append({"tool": rng.choice(tools), "args": _random_trace_args(rng)})
        expected = _trace_expected(expected_raw)
        dp = _trace_matched(expected, actual)
        brute = _brute_trace_matched(expected, actual)
        assert dp == brute, (
            f"trial {trial}: dp={dp} brute={brute} expected={expected_raw!r} actual={actual!r}"
        )
        if expected:
            case = Case(
                id=f"t{trial}",
                input="q",
                reference=None,
                assertions=[AssertionSpec(kind="trace_match", value=expected_raw)],
            )
            scored = score_case(case, json.dumps(actual))
            assert scored[0]["trace_match"] == pytest.approx(brute / len(expected))


def test_history_skips_corrupt_trailing_line(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    append_history(
        {"timestamp": "t", "command": "run", "suite": "s", "metrics": {"mean_score": 1}},
        path,
    )
    append_history(
        {"timestamp": "u", "command": "run", "suite": "s", "metrics": {"mean_score": 0.5}},
        path,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"partial":')
    rows, notes = load_history(path)
    assert len(rows) == 2
    assert len(notes) == 1
    assert "invalid JSON" in notes[0]


def test_workflow_yaml_files_parse() -> None:
    for relative in (
        ".github/workflows/ci.yml",
        ".github/workflows/pr-selftest.yml",
        ".github/workflows/release.yml",
        "action.yml",
    ):
        loaded = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
        assert isinstance(loaded, dict), relative


def test_history_empty_file_and_concurrent_appends(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert load_history(empty) == ([], [])

    path = tmp_path / "concurrent.jsonl"

    def write_batch(worker: int) -> None:
        for index in range(40):
            append_history(
                {
                    "timestamp": f"t{worker}-{index}",
                    "command": "run",
                    "suite": "s",
                    "metrics": {"mean_score": index / 40},
                },
                path,
            )

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(write_batch, range(8)))
    rows, notes = load_history(path)
    assert notes == []
    assert len(rows) == 320
    assert len({row["timestamp"] for row in rows}) == 320


def test_matrix_tie_break_is_deterministic_and_rejects_duplicates(tmp_path: Path) -> None:
    scores = [0.5, 1.0]

    def entry(name: str) -> tuple[MatrixConfig, RunResult, str]:
        config = MatrixConfig(name=name, provider="echo", template="{input}")
        return config, _run_result(name, scores), "m"

    finished = [entry("beta"), entry("alpha"), entry("gamma")]
    first = rank_runs(finished, suite_name="s", n_boot=80, seed=13)
    second = rank_runs(finished, suite_name="s", n_boot=80, seed=13)
    assert [row.name for row in first.rows] == ["alpha", "beta", "gamma"]
    assert [(row.name, row.diff_vs_best, row.ci_low, row.ci_high) for row in first.rows] == [
        (row.name, row.diff_vs_best, row.ci_low, row.ci_high) for row in second.rows
    ]

    single = tmp_path / "one.yaml"
    single.write_text("only:\n  provider: echo\n  template: '{input}'\n", encoding="utf-8")
    with pytest.raises(MatrixError, match="at least two"):
        load_matrix(single)

    duplicate = tmp_path / "dup.yaml"
    duplicate.write_text(
        "- name: a\n  provider: echo\n  template: '{input}'\n"
        "- name: a\n  provider: mock\n  template: '{input}'\n",
        encoding="utf-8",
    )
    with pytest.raises(MatrixError, match="Duplicate matrix config name"):
        load_matrix(duplicate)


@pytest.mark.parametrize(
    "check_name,target",
    [
        ("python", "promptregress.doctor.python_check"),
        ("package", "promptregress.doctor.package_check"),
        ("providers", "promptregress.doctor.provider_check"),
        ("cache", "promptregress.doctor.cache_check"),
        ("scorer_plugins", "promptregress.doctor.plugin_check"),
        ("smoke", "promptregress.doctor.smoke_check"),
    ],
)
def test_doctor_each_check_failure_exits_1(
    check_name: str,
    target: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-promptregress-doctor-secret")
    for name in (
        "XAI_API_KEY",
        "PROMPTREGRESS_API_KEY",
        "PROMPTREGRESS_BASE_URL",
        "PROMPTREGRESS_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    def forced(*_args: object, **_kwargs: object) -> Check:
        return Check(check_name, False, f"{check_name} forced")

    monkeypatch.setattr(target, forced)
    assert main(["doctor"]) == 1
    report = capsys.readouterr().out
    assert f"{check_name}: FAIL {check_name} forced" in report
    assert "result: fail (" in report
    assert "1 failed)" in report
    assert "sk-promptregress-doctor-secret" not in report


def test_action_forwards_mock_default_without_mock_rules(tmp_path: Path) -> None:
    """``mock-default`` must reach compare even when ``mock-rules`` is empty.

    The composite action only appended ``--mock-default`` inside ``if rules:``,
    so a mock provider config that set ``mock_default`` alone silently used the
    CLI empty default and scored every case as the empty string.
    """
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "s",
                "cases": [
                    {
                        "id": "a",
                        "input": "hi",
                        "reference": "bye",
                        "assertions": [{"kind": "exact_match"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.yaml"
    candidate = tmp_path / "candidate.yaml"
    baseline.write_text(
        "provider: mock\ntemplate: '{input}'\nmock_default: bye\n",
        encoding="utf-8",
    )
    candidate.write_text("template: '{input}'\n", encoding="utf-8")

    captured: dict[str, list[str]] = {}

    class _Result:
        returncode = 0

    def fake_run(command: list[str], **_kwargs: object) -> _Result:
        captured["command"] = list(command)
        return _Result()

    env = {
        "SUITE": str(suite),
        "BASELINE_CONFIG": str(baseline),
        "CANDIDATE_CONFIG": str(candidate),
        "MAX_REGRESSION": "0",
        "PROVIDER": "mock",
        "MOCK_RULES": "",
        "MOCK_DEFAULT": "",
        "REPORT_JSON": "",
        "REPORT_MD": "",
        "REPORT_HTML": "",
        "GITHUB_OUTPUT": str(tmp_path / "github-output.txt"),
    }
    script = _action_compare_script()
    with patch.dict(os.environ, env, clear=False):
        with patch("subprocess.run", fake_run):
            with pytest.raises(SystemExit) as exited:
                exec(compile(script, "action-compare.py", "exec"), {"__name__": "__main__"})
    assert exited.value.code == 0
    command = captured["command"]
    assert "--mock-default" in command
    assert command[command.index("--mock-default") + 1] == "bye"
    assert "--mock-rules" not in command


def test_workflows_reference_existing_paths_and_release_builds_dist() -> None:
    for relative in (
        "examples/support_qa.json",
        "examples/ci/baseline.yaml",
        "examples/ci/candidate.yaml",
        "docs/assets/banner.svg",
        "docs/assets/demo.svg",
        "docs/assets/bootstrap.png",
        "docs/assets/per-case.png",
        "docs/examples/report.html",
    ):
        assert (ROOT / relative).is_file(), relative

    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "python -m build" in release
    assert "dist/*" in release
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "sdist" in changelog and "wheel" in changelog

    action = yaml.safe_load((ROOT / "action.yml").read_text(encoding="utf-8"))
    script = "\n".join(step.get("run", "") for step in action["runs"]["steps"])
    for flag in (
        "--provider",
        "--baseline-template",
        "--candidate-template",
        "--max-regression",
        "--mock-rules",
        "--mock-default",
        "--report-json",
        "--report-md",
        "--report-html",
        "--model",
        "--no-history",
    ):
        assert flag in script, flag


def test_review1_json_and_whitespace_preference_still_hold() -> None:
    assert parse_preference("hello\n", "world\n", "hello\n") == "first"
    assert parse_preference("hello\n", "world\n", "hello") == "first"
    assert parse_preference(" answer ", "other", " answer ") == "first"
    case = Case(id="j", input="q", reference=None, assertions=[AssertionSpec(kind="json_valid")])
    for output in ("NaN", "Infinity", "-Infinity", '{"a": NaN}', "[1, Infinity]"):
        assert score_case(case, output)[1] == 0.0
    assert score_case(case, "null")[1] == 1.0
    assert score_case(case, '{"a": 1}')[1] == 1.0
