"""Regression tests for bugs found in the independent v0.7.1 review cycle #18.

Scope is quote-state BOM stripping after escaped quotes in JSON/YAML, and
JUnit ``time`` for signed-zero latency left by the 0.7.1 negative clamp.
"""

from __future__ import annotations

import json
from pathlib import Path

from promptregress.irt import load_run_report
from promptregress.junit_xml import render_junit_xml
from promptregress.loading import load_suite, strip_utf8_bom_text
from promptregress.models import CaseResult, RunResult
from promptregress.power import load_report_diffs
from promptregress.report import run_to_dict


def test_json_backslash_quote_does_not_desync_bom_strip(tmp_path: Path) -> None:
    """A JSON ``\\"`` must not leave quote-state open across lines.

    ``strip_utf8_bom_text`` only treated CSV ``""`` as an escaped quote.
    Pretty-printed run reports put model output with embedded quotes on one
    line as ``\\"``. That backslash-quote toggled ``in_double`` off one step
    early, so a mid-file chunk BOM on the next structural line was kept and
    ``load_run_report`` / ``load_report_diffs`` failed with a property-name
    error. Skip the character after a backslash inside quotes (JSON/YAML)
    so chunk BOMs still strip and baselines / ``irt`` / ``power`` load.
    """
    run = RunResult(
        suite_name="s",
        config_name="c",
        results=[
            CaseResult(
                case_id="a",
                output='one " quote',
                scores={"exact_match": 1.0},
                score=1.0,
                passed=True,
                latency_ms=1.0,
            )
        ],
    )
    run_path = tmp_path / "run.json"
    run_lines = json.dumps(run_to_dict(run), indent=2).splitlines(keepends=True)
    for index, line in enumerate(run_lines):
        if "output" in line and "one" in line:
            run_lines[index + 1] = "\ufeff" + run_lines[index + 1]
            break
    blob = "".join(run_lines)
    assert "\ufeff" not in strip_utf8_bom_text(blob)
    run_path.write_text(blob, encoding="utf-8")
    loaded = load_run_report(run_path)
    assert loaded.results[0].output == 'one " quote'

    compare_path = tmp_path / "compare.json"
    compare_text = json.dumps(
        {
            "per_case": [
                {"case_id": "a", "diff": 0.1, "note": 'said "hi"'},
                {"case_id": "b", "diff": -0.2},
            ]
        },
        indent=2,
    )
    compare_lines = compare_text.splitlines(keepends=True)
    for index, line in enumerate(compare_lines):
        if "said" in line:
            compare_lines[index + 1] = "\ufeff" + compare_lines[index + 1]
            break
    compare_path.write_text("".join(compare_lines), encoding="utf-8")
    assert load_report_diffs(compare_path) == [0.1, -0.2]


def test_yaml_backslash_quote_keeps_continuation_feff(tmp_path: Path) -> None:
    """YAML ``\\"`` must not end a double-quoted scalar for BOM tracking.

    With only CSV ``""`` escapes, a single ``\\"`` left quote-state false on
    the next physical line, so a legitimate line-leading ``U+FEFF`` inside
    the scalar was stripped and the loaded input lost a character that
    ``yaml.safe_load`` on the raw text kept. Backslash-escaping restores
    the in-quote region through the continuation.
    """
    path = tmp_path / "suite.yaml"
    path.write_text(
        "name: t\n"
        "cases:\n"
        "- id: a\n"
        '  input: "say \\"hi\n'
        '\ufeffworld"\n'
        "  reference: x\n"
        "  assertions:\n"
        "  - kind: contains\n"
        "    value: hi\n",
        encoding="utf-8",
    )
    suite = load_suite(path)
    assert suite.cases[0].input == 'say "hi \ufeffworld'


def test_junit_signed_zero_latency_time_is_zero() -> None:
    """JUnit ``time`` must not render signed zero as ``-0.000000``.

    0.7.1 clamped strictly negative latency to ``0.000000``, but ``-0.0``
    compares equal to ``0`` and is not ``< 0``, so ``f"{-0.0:.6f}"`` still
    emitted a leading minus. ``MockProvider(latency_ms=-0.0)`` and loaded
    reports can carry signed zero; normalise it like other non-positive
    values so CI consumers never see a negative ``time``.
    """
    result = RunResult(
        suite_name="s",
        config_name="c",
        results=[
            CaseResult(
                case_id="a",
                output="x",
                scores={"contains": 1.0},
                score=1.0,
                passed=True,
                latency_ms=-0.0,
            )
        ],
    )
    xml = render_junit_xml([result])
    assert 'time="-0.000000"' not in xml
    assert 'time="0.000000"' in xml
