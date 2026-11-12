"""GitHub Actions annotations for failed cases and gate summaries."""

import json
from pathlib import Path

import pytest

from promptregress.annotations import (
    annotation_lines,
    case_annotation,
    escape_message,
    escape_property,
    github_annotations_enabled,
)
from promptregress.cli import main
from promptregress.loading import load_suite
from promptregress.models import CaseResult


def _fail(
    case_id: str,
    *,
    source_line: int | None = None,
    scores: dict[str, float] | None = None,
    error: str | None = None,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        output="out",
        scores={"contains": 0.0} if scores is None else scores,
        score=0.0,
        passed=False,
        latency_ms=1.0,
        error=error,
        source_line=source_line,
    )


def _pass(case_id: str) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        output="out",
        scores={"contains": 1.0},
        score=1.0,
        passed=True,
        latency_ms=1.0,
        source_line=1,
    )


def test_property_and_message_escaping() -> None:
    assert escape_property("a%b\r\nc:d,e") == "a%25b%0D%0Ac%3Ad%2Ce"
    assert escape_message("a%b\r\nc:d,e") == "a%25b%0D%0Ac:d,e"
    line = case_annotation(_fail("id:%\r\n:,"), "dir/%\r\n:,.json")
    assert line == (
        "::error file=dir/%25%0D%0A%3A%2C.json::id:%25%0D%0A:,: contains=0.000000 (score 0.000000)"
    )
    numbered = case_annotation(_fail("id:%\r\n:,", source_line=4), "dir/%\r\n:,.json")
    assert numbered == (
        "::error file=dir/%25%0D%0A%3A%2C.json,line=4::"
        "id:%25%0D%0A:,: contains=0.000000 (score 0.000000)"
    )


def test_annotation_lines_skip_passes_and_summarize_a_clean_gate_failure() -> None:
    cases = annotation_lines([_pass("ok"), _fail("drop", source_line=3)], "suite.json")
    assert cases == ["::error file=suite.json,line=3::drop: contains=0.000000 (score 0.000000)"]

    from promptregress.gate import GateVerdict

    summary = annotation_lines(
        [_pass("ok")],
        "suite.json",
        GateVerdict(passed=False, reasons=["PASS regression: ok.", "FAIL drift: over budget."]),
    )
    assert summary == ["::error::FAIL drift: over budget."]
    quiet = annotation_lines(
        [_pass("ok")],
        "suite.json",
        GateVerdict(passed=True, reasons=["PASS regression: ok."]),
    )
    assert quiet == []
    # A failed case suppresses the summary even when the gate also failed.
    both = annotation_lines(
        [_fail("drop")],
        "suite.json",
        GateVerdict(passed=False, reasons=["FAIL drift: over budget."]),
    )
    assert both == ["::error file=suite.json::drop: contains=0.000000 (score 0.000000)"]
    assert github_annotations_enabled(False, {}) is False
    assert github_annotations_enabled(False, {"GITHUB_ACTIONS": "false"}) is False
    assert github_annotations_enabled(False, {"GITHUB_ACTIONS": "true"}) is True
    assert github_annotations_enabled(True, {}) is True


def test_loader_records_case_lines(tmp_path: Path) -> None:
    jsonl = tmp_path / "cases.jsonl"
    jsonl.write_text(
        '{"id": "a", "input": "q", "assertions": []}\n'
        "\n"
        '{"id": "b", "input": "q", "assertions": []}\n',
        encoding="utf-8",
    )
    assert [case.source_line for case in load_suite(jsonl).cases] == [1, 3]

    document = tmp_path / "cases.json"
    document.write_text(
        '{\n  "name": "s",\n  "cases": [\n'
        '    {"id": "a", "input": "q", "assertions": []},\n'
        '    {"id": "b", "input": "q", "assertions": []}\n  ]\n}\n',
        encoding="utf-8",
    )
    assert [case.source_line for case in load_suite(document).cases] == [4, 5]

    yaml_path = tmp_path / "cases.yaml"
    yaml_path.write_text(
        "name: s\ncases:\n  - id: a\n    input: q\n  - id: b\n    input: q\n",
        encoding="utf-8",
    )
    assert [case.source_line for case in load_suite(yaml_path).cases] == [3, 5]

    csv_path = tmp_path / "cases.csv"
    csv_path.write_text("id,input,reference\na,q,q\nb,r,r\n", encoding="utf-8")
    assert [case.source_line for case in load_suite(csv_path).cases] == [2, 3]

    compact = tmp_path / "one.json"
    compact.write_text(
        '{"name":"s","cases":[{"id":"a","input":"q","assertions":[]},'
        '{"id":"b","input":"q","assertions":[]}]}\n',
        encoding="utf-8",
    )
    assert [case.source_line for case in load_suite(compact).cases] == [1, 1]

    weird = tmp_path / "weird.json"
    weird.write_text(
        json.dumps({"name": "s", "cases": [{"id": "a\nb", "input": "q", "assertions": []}]}) + "\n",
        encoding="utf-8",
    )
    # Escaped newline stays on one physical line; compose marks that line.
    assert load_suite(weird).cases[0].source_line == 1


def _failing_jsonl(path: Path) -> Path:
    path.write_text(
        '{"id": "keep", "input": "hello", "assertions": '
        '[{"kind": "contains", "value": "hello"}]}\n'
        "\n"
        '{"id": "drop", "input": "hello", "assertions": '
        '[{"kind": "contains", "value": "missing"}]}\n'
        '{"id": "a%b:c,d", "input": "hello", "assertions": '
        '[{"kind": "contains", "value": "missing"}]}\n',
        encoding="utf-8",
    )
    return path


def _run_failing(path: Path, extra: list[str]) -> int:
    return main(
        [
            "run",
            str(path),
            "--provider",
            "mock",
            "--mock-default",
            "hello",
            "--config-name",
            "ann",
            "--out",
            str(path.with_suffix(".out.json")),
            "--no-history",
            *extra,
        ]
    )


def test_cli_exact_format_and_message_escaping(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    suite = _failing_jsonl(tmp_path / "cases.jsonl")
    code = _run_failing(suite, ["--github-annotations"])
    assert code == 0
    captured = capsys.readouterr()
    assert "::error" not in captured.out
    lines = captured.err.splitlines()
    assert lines == [
        f"::error file={suite},line=3::drop: contains=0.000000 (score 0.000000)",
        f"::error file={suite},line=4::a%25b:c,d: contains=0.000000 (score 0.000000)",
    ]


def test_no_annotations_without_flag_or_env(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    suite = _failing_jsonl(tmp_path / "cases.jsonl")
    assert _run_failing(suite, []) == 0
    assert "::error" not in capsys.readouterr().err

    monkeypatch.setenv("GITHUB_ACTIONS", "false")
    assert _run_failing(suite, []) == 0
    assert "::error" not in capsys.readouterr().err


def test_github_actions_env_enables_annotations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    suite = _failing_jsonl(tmp_path / "cases.jsonl")
    assert _run_failing(suite, []) == 0
    err = capsys.readouterr().err
    assert f"::error file={suite},line=3::drop: contains=0.000000 (score 0.000000)" in err


def test_gate_failure_without_a_failed_case_is_one_summary_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    suite = tmp_path / "hello.json"
    suite.write_text(
        json.dumps(
            {
                "name": "hello",
                "cases": [
                    {
                        "id": "a",
                        "input": "hello",
                        "assertions": [{"kind": "contains", "value": "hello"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "compare",
            str(suite),
            "--provider",
            "echo",
            "--baseline-template",
            "{input}",
            "--candidate-template",
            "note {input}",
            "--max-regression",
            "0",
            "--max-mean-drift",
            "0",
            "--github-annotations",
            "--no-history",
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    fail_lines = [line[2:] for line in captured.out.splitlines() if line.startswith("- FAIL ")]
    assert fail_lines
    assert captured.err.splitlines() == ["::error::" + "; ".join(fail_lines)]
    assert "FAIL drift:" in captured.err
    assert "file=" not in captured.err


def test_budget_breach_summary_when_every_case_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    suite = tmp_path / "ok.json"
    suite.write_text(
        json.dumps(
            {
                "name": "ok",
                "cases": [
                    {
                        "id": "a",
                        "input": "hello",
                        "assertions": [{"kind": "contains", "value": "hello"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "run",
            str(suite),
            "--provider",
            "mock",
            "--mock-default",
            "hello",
            "--config-name",
            "ok",
            "--out",
            str(tmp_path / "run.json"),
            "--max-p95-latency-ms",
            "0",
            "--github-annotations",
            "--no-history",
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "p95 latency 1.000000ms exceeds budget 0.000000ms" in captured.out
    assert captured.err.splitlines() == [
        "::error::FAIL latency: p95 latency 1.000000ms exceeds budget 0.000000ms."
    ]


def test_case_failure_suppresses_the_budget_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    suite = _failing_jsonl(tmp_path / "cases.jsonl")
    code = _run_failing(suite, ["--github-annotations", "--max-p95-latency-ms", "0"])
    assert code == 1
    err_lines = capsys.readouterr().err.splitlines()
    assert err_lines[0].startswith(f"::error file={suite},line=")
    assert all(not line.startswith("::error::") for line in err_lines)
