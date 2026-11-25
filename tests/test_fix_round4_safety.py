"""Round-4 review, findings 1-3: output paths must not destroy inputs,
writes must be atomic, and history must survive partial/odd lines."""

import json
import os
from pathlib import Path

import pytest

from promptregress.cli import main
from promptregress.history import append_history, load_history
from promptregress.junit_xml import atomic_write_text


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


def _compare_args(suite: Path, report: Path) -> list[str]:
    return [
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
        "--no-history",
        "--report-json",
        str(report),
    ]


def test_compare_report_json_refuses_the_suite_path(tmp_path: Path) -> None:
    suite = _suite(tmp_path / "suite.json")
    before = suite.read_bytes()
    assert main(_compare_args(suite, suite)) == 2
    assert suite.read_bytes() == before


def test_compare_report_json_refuses_a_symlink_to_the_suite(tmp_path: Path) -> None:
    suite = _suite(tmp_path / "suite.json")
    alias = tmp_path / "alias.json"
    alias.symlink_to(suite)
    before = suite.read_bytes()
    assert main(_compare_args(suite, alias)) == 2
    assert suite.read_bytes() == before


def test_compare_report_json_refuses_a_hard_link_to_the_suite(tmp_path: Path) -> None:
    suite = _suite(tmp_path / "suite.json")
    alias = tmp_path / "alias.json"
    os.link(suite, alias)
    before = suite.read_bytes()
    assert main(_compare_args(suite, alias)) == 2
    assert suite.read_bytes() == before


def _comparison(tmp_path: Path) -> Path:
    suite = _suite(tmp_path / "suite.json")
    report = tmp_path / "cmp.json"
    assert main(_compare_args(suite, report)) == 0
    return report


def test_pareto_out_refuses_its_input(tmp_path: Path) -> None:
    report = _comparison(tmp_path)
    before = report.read_bytes()
    assert main(["pareto", str(report), "--out", str(report)]) == 2
    assert report.read_bytes() == before


def test_bayes_out_refuses_its_input(tmp_path: Path) -> None:
    report = _comparison(tmp_path)
    before = report.read_bytes()
    assert main(["bayes", str(report), "--out", str(report)]) == 2
    assert report.read_bytes() == before


def test_atomic_write_keeps_the_previous_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "report.json"
    target.write_text('{"marker": "previous-good-report"}', encoding="utf-8")

    def fail_replace(src: object, dst: object) -> None:
        raise OSError("simulated crash before replace")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        atomic_write_text(target, '{"marker": "new"}')
    assert target.read_text(encoding="utf-8") == '{"marker": "previous-good-report"}'
    assert list(tmp_path.glob("*.tmp")) == []


def test_append_history_after_a_partial_line_keeps_the_new_row(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text('{"timestamp": "cut-off-mid', encoding="utf-8")  # no newline
    append_history({"timestamp": "t2", "command": "compare"}, path)
    rows, notes = load_history(path)
    assert [row["timestamp"] for row in rows] == ["t2"]
    assert notes  # the partial line is reported, not silently glued


def test_load_history_keeps_rows_with_unicode_line_separators(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    record = {"timestamp": "t1", "summary": "a\u2028b\u2029c\u0085d"}
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    rows, notes = load_history(path)
    assert rows == [record]
    assert notes == []
