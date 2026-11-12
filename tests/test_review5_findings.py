"""Regression tests for bugs found in the independent v0.6.3 review cycle.

Scope is the 0.6.0–0.6.3 surface: interactive HTML / Markdown reports, suite
audit text output, and the same unpaired-surrogate class JUnit already fixed
in 0.6.3.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from helpers import case_result, run_with
from promptregress.audit import audit_paths, format_audit
from promptregress.cli import main
from promptregress.gate import evaluate_gate
from promptregress.history import write_trend_html
from promptregress.models import CaseResult, RunResult
from promptregress.report import format_text, write_html, write_markdown
from promptregress.stats import compare_runs


def test_html_and_markdown_reports_strip_surrogates_so_utf8_write_succeeds(
    tmp_path: Path,
) -> None:
    """Unpaired surrogates in model output must not crash report writers.

    JUnit export already strips U+D800–U+DFFF (0.6.3). HTML and Markdown
    writers passed the raw string to ``Path.write_text(..., encoding="utf-8")``,
    which raises ``UnicodeEncodeError`` on a lone surrogate — a plausible
    completion when a provider echoes a ``\\ud800`` JSON escape. The report
    must write successfully and keep the surrounding text after the surrogate
    is removed.
    """
    baseline = run_with([("a", 1.0, True)], "baseline")
    candidate = RunResult(
        suite_name="suite",
        config_name="candidate",
        results=[
            replace(
                case_result("a", 0.0, False, output="before\ud800after < &"),
                case_input="ask\ud800me",
            )
        ],
    )
    comparison = compare_runs(baseline, candidate, n_boot=40, seed=1)
    verdict = evaluate_gate(comparison, max_regression=1.0)

    html_path = tmp_path / "report.html"
    md_path = tmp_path / "report.md"
    write_html(comparison, html_path, verdict)
    write_markdown(comparison, md_path, verdict)

    html = html_path.read_text(encoding="utf-8")
    md = md_path.read_text(encoding="utf-8")
    assert "\ud800" not in html
    assert "\ud800" not in md
    assert "beforeafter" in html
    assert "beforeafter" in md
    assert "&lt;" in html
    assert "askme" in html


def test_compare_report_html_survives_surrogate_mock_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``compare --report-html`` must exit from the gate, not UnicodeEncodeError.

    A mock candidate that returns a lone surrogate previously made the CLI
    catch ``UnicodeEncodeError`` (an ``OSError``) and exit 2 before the HTML
    file was written.
    """
    monkeypatch.chdir(tmp_path)
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "s",
                "cases": [
                    {
                        "id": "a",
                        "input": "hello",
                        "reference": "hello",
                        "assertions": [{"kind": "exact_match"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "out.html"
    rules = json.dumps([["BASE", "hello"], ["CAND", "x\ud800y"]])
    code = main(
        [
            "compare",
            str(suite),
            "--provider",
            "mock",
            "--mock-rules",
            rules,
            "--baseline-template",
            "BASE {input}",
            "--candidate-template",
            "CAND {input}",
            "--max-regression",
            "1",
            "--report-html",
            str(report),
            "--no-history",
        ]
    )
    assert code == 0
    text = report.read_text(encoding="utf-8")
    assert "\ud800" not in text
    assert "xy" in text


def test_format_text_strips_surrogates_from_case_ids() -> None:
    """Plain-text compare output must be printable on a UTF-8 stdout."""
    baseline = RunResult(
        suite_name="s",
        config_name="b",
        results=[CaseResult("id\ud800", "ok", {}, 1.0, True, 0.0)],
    )
    candidate = RunResult(
        suite_name="s",
        config_name="c",
        results=[CaseResult("id\ud800", "bad", {}, 0.0, False, 0.0)],
    )
    comparison = compare_runs(baseline, candidate, n_boot=20, seed=1)
    text = format_text(comparison, evaluate_gate(comparison, max_regression=1.0))
    assert "\ud800" not in text
    assert "id" in text
    text.encode("utf-8")


def test_audit_text_report_strips_surrogates_so_print_succeeds(tmp_path: Path) -> None:
    """Audit findings that quote surrogate case ids must still print.

    A suite JSON ``\"id\": \"a\\ud800b\"`` loads as a string with a lone
    surrogate. ``format_audit`` previously kept that character, so
    ``promptregress audit`` raised ``UnicodeEncodeError`` on UTF-8 stdout
    and the CLI mapped it to exit 2 instead of reporting the finding.
    """
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "s",
                "cases": [
                    {
                        "id": "a\ud800b",
                        "input": "x",
                        "assertions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = audit_paths([str(suite)])
    assert report.error_count == 1
    assert report.findings[0].check == "no_assertions"
    text = format_audit(report)
    assert "\ud800" not in text
    assert "case: ab" in text
    text.encode("utf-8")
    code = main(["audit", str(suite)])
    assert code == 1


def test_trend_html_strips_surrogates_in_suite_name(tmp_path: Path) -> None:
    """Trend HTML write must tolerate a suite name that carries a surrogate."""
    path = tmp_path / "trend.html"
    write_trend_html("su\ud800te", [("t", 0.5)], path)
    text = path.read_text(encoding="utf-8")
    assert "\ud800" not in text
    assert "sute" in text
