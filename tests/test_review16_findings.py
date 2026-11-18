"""Regression tests for bugs found in the independent v0.6.15 review cycle.

Scope is quote-blind per-line BOM stripping after 0.6.15 taught every
loader to delete line-leading ``U+FEFF``, and the run/compare report
loaders that still only stripped a leading residual BOM.
"""

from __future__ import annotations

import json
from pathlib import Path

from promptregress.irt import load_run_report
from promptregress.loading import load_suite
from promptregress.models import CaseResult, RunResult
from promptregress.power import load_report_diffs
from promptregress.report import run_to_dict


def test_csv_preserves_line_leading_feff_inside_quoted_field(tmp_path: Path) -> None:
    """A ``U+FEFF`` at the start of a quoted CSV continuation line is data.

    0.6.15 stripped every line-leading ``U+FEFF`` before ``DictReader``.
    That fixed mid-file Windows-chunk BOMs on new rows, but a multiline
    quoted field whose next physical line began with ZWNBSP (or a pasted
    BOM-as-data) lost that character — ``hello\\n\\ufeffworld`` became
    ``hello\\nworld``. Skip BOM stripping while inside double quotes so
    chunk BOMs outside quotes still clear and in-field ``U+FEFF`` stays.
    """
    path = tmp_path / "cases.csv"
    path.write_text('id,input,reference\na,"hello\n\ufeffworld",ref\n', encoding="utf-8")
    suite = load_suite(path)
    assert suite.cases[0].input == "hello\n\ufeffworld"
    assert suite.cases[0].source_line == 3


def test_csv_midfile_bom_before_quoted_row_still_loads(tmp_path: Path) -> None:
    """A mid-file BOM before a quoted CSV row must still strip and parse.

    Quote-aware stripping must not keep a chunk BOM that sits outside the
    quotes (``\\ufeff\"b\\nc\"``); that would leave the opening quote off
    column zero and split one record into broken rows. Strip outside
    quotes, keep ``source_line`` on the record's last physical line.
    """
    path = tmp_path / "cases.csv"
    path.write_bytes(b'id,input,reference\na,x,y\n\xef\xbb\xbf"b\nc",z,w\n')
    suite = load_suite(path)
    assert [(case.id, case.source_line) for case in suite.cases] == [("a", 2), ("b\nc", 4)]


def test_yaml_preserves_line_leading_feff_in_double_quoted_scalar(tmp_path: Path) -> None:
    """Double-quoted YAML continuations keep a line-leading ``U+FEFF``.

    The same blind per-line strip that broke CSV deleted ``U+FEFF`` at the
    start of an unindented double-quoted continuation, so the loaded
    input lost a character that ``yaml.safe_load`` on the raw text kept.
    Quote-aware stripping matches the raw parse for that scalar.
    """
    path = tmp_path / "suite.yaml"
    path.write_text(
        "name: t\n"
        "cases:\n"
        "- id: a\n"
        '  input: "hello\n'
        '\ufeffworld"\n'
        "  reference: x\n"
        "  assertions:\n"
        "  - kind: contains\n"
        "    value: hello\n",
        encoding="utf-8",
    )
    suite = load_suite(path)
    assert suite.cases[0].input == "hello \ufeffworld"


def test_run_and_compare_reports_strip_midfile_bom(tmp_path: Path) -> None:
    """Pretty-printed run/compare JSON with a mid-file BOM must still load.

    Suite/pricing/history paths use ``strip_utf8_bom_text``. ``load_run_report``
    and ``load_report_diffs`` only stripped a leading residual ``U+FEFF``, so
    a line-leading BOM on a later pretty-printed line (pasted Windows chunk)
    made ``json.loads`` fail with a misleading property-name error. Shared
    mid-file stripping keeps baselines, ``irt``, and ``power --from-report``
    consistent with the other loaders.
    """
    run = RunResult(
        suite_name="s",
        config_name="c",
        results=[
            CaseResult(
                case_id="a",
                output="x",
                scores={"exact_match": 1.0},
                score=1.0,
                passed=True,
                latency_ms=1.0,
            )
        ],
    )
    run_path = tmp_path / "run.json"
    run_text = json.dumps(run_to_dict(run), indent=2)
    run_lines = run_text.splitlines(keepends=True)
    for index, line in enumerate(run_lines):
        if "config_name" in line:
            run_lines[index] = "\ufeff" + line
            break
    run_path.write_text("".join(run_lines), encoding="utf-8")
    loaded = load_run_report(run_path)
    assert loaded.config_name == "c"
    assert loaded.results[0].case_id == "a"

    compare_path = tmp_path / "compare.json"
    compare_text = json.dumps(
        {"per_case": [{"case_id": "a", "diff": 0.1}, {"case_id": "b", "diff": -0.2}]},
        indent=2,
    )
    compare_lines = compare_text.splitlines(keepends=True)
    for index, line in enumerate(compare_lines):
        if '"diff": -0.2' in line:
            compare_lines[index] = "\ufeff" + line
            break
    compare_path.write_text("".join(compare_lines), encoding="utf-8")
    assert load_report_diffs(compare_path) == [0.1, -0.2]
