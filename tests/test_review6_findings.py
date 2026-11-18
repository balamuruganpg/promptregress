"""Regression tests for bugs found in the independent v0.6.5 review cycle.

Scope is the 0.6.4–0.6.5 surface: GitHub Actions annotation escaping, cost /
latency budget reporting on ``run``, and unpaired-surrogate handling that
compare/audit already covered but ``run`` / annotation summaries missed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptregress.annotations import annotation_lines, case_annotation, escape_property
from promptregress.cli import _run_summary, main
from promptregress.gate import GateVerdict
from promptregress.models import CaseResult, RunResult


def test_annotation_property_escapes_comma_as_percent_2c() -> None:
    """Property commas must use ``%2C``, the GitHub Actions runner mapping.

    ``escape_property`` wrote ``%3B`` (semicolon) for ``,``. The actions
    toolkit and the runner decode ``%2C`` → ``,`` and ``%3B`` → ``;``, so a
    suite path such as ``suites/prod,staging/qa.json`` was rewritten to
    ``suites/prod;staging/qa.json`` and the annotation pointed at the wrong
    file. Message text still leaves commas unescaped.
    """
    assert escape_property("suites/prod,staging/qa.json") == "suites/prod%2Cstaging/qa.json"
    assert escape_property("a%b\r\nc:d,e") == "a%25b%0D%0Ac%3Ad%2Ce"
    row = CaseResult(
        case_id="id",
        output="o",
        scores={"contains": 0.0},
        score=0.0,
        passed=False,
        latency_ms=1.0,
        source_line=2,
    )
    line = case_annotation(row, "dir/a,b.json")
    assert line.startswith("::error file=dir/a%2Cb.json,line=2::")


def test_run_summary_strips_surrogates_so_utf8_print_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run`` must not exit 2 when the suite name carries a lone surrogate.

    Compare routes through ``format_text``, which strips U+D800–U+DFFF before
    printing. ``run`` printed ``_run_summary`` raw, so a suite JSON name
    like ``\"su\\ud800te\"`` raised ``UnicodeEncodeError`` on a UTF-8 stdout
    and the CLI mapped that to exit 2 — after the run JSON was already
    written — instead of exiting 0 for a scored run.
    """
    summary = _run_summary(
        RunResult(
            suite_name="su\ud800te",
            config_name="c",
            results=[
                CaseResult(
                    case_id="a",
                    output="hi",
                    scores={"contains": 1.0},
                    score=1.0,
                    passed=True,
                    latency_ms=1.0,
                )
            ],
        )
    )
    assert "\ud800" not in summary
    assert "suite=sute" in summary
    summary.encode("utf-8")

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "su\ud800te",
                "cases": [
                    {
                        "id": "a",
                        "input": "hi",
                        "assertions": [{"kind": "contains", "value": "hi"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.json"
    code = main(
        [
            "run",
            str(suite),
            "--provider",
            "mock",
            "--mock-default",
            "hi",
            "--config-name",
            "c",
            "--out",
            str(out),
            "--no-history",
        ]
    )
    assert code == 0
    assert out.is_file()


def test_gate_summary_annotation_strips_surrogates_in_fail_reasons() -> None:
    """Summary ``::error::`` lines must strip unpaired surrogates like case lines.

    Case annotations strip surrogates from the id and assertion text. A gate
    failure with no failed case joins FAIL reasons as-is; a cost-unknown note
    that quotes a model id containing ``\\ud800`` left a lone surrogate in the
    command, so ``emit_annotations`` raised ``UnicodeEncodeError`` on UTF-8
    stderr and the CLI exited 2 instead of emitting the annotation.
    """
    passed = CaseResult(
        case_id="a",
        output="ok",
        scores={"contains": 1.0},
        score=1.0,
        passed=True,
        latency_ms=1.0,
    )
    lines = annotation_lines(
        [passed],
        "suite.json",
        GateVerdict(
            passed=False,
            reasons=["FAIL cost: cost is unknown (unknown model 'm\ud800x') (budget $1.000000)."],
        ),
    )
    assert len(lines) == 1
    assert "\ud800" not in lines[0]
    assert "unknown model 'mx'" in lines[0]
    lines[0].encode("utf-8")
