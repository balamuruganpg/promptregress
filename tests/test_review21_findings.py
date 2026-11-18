"""Regression tests for bugs found in independent review cycle #21.

Scope is the reporting surface: JUnit character data must round-trip the
same way attributes already do.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from promptregress.junit_xml import failure_message, render_junit_xml
from promptregress.models import CaseResult, RunResult


def test_junit_character_data_preserves_carriage_return() -> None:
    """JUnit element text must keep CR the same way attributes already do.

    ElementTree escapes CR in attributes as ``&#13;`` but writes a raw CR in
    element text. An XML 1.0 parser normalizes that raw CR, and CR+LF, to LF
    before parsing, so ``<failure>`` text no longer matched the ``message``
    attribute and ``<system-out>`` dropped carriage returns from model output
    and regression notes. The serializer rewrites leftover raw CR to ``&#13;``
    after ``ElementTree.write``. Parsers expand that reference back to CR, and
    attribute escapes already written as ``&#13;`` are not encoded again.
    """
    output = "line1\rline2\r\nline3 < &"
    error = "boom\rnext < &"
    case_id = "id\rname"
    failed = CaseResult(
        case_id=case_id,
        output=output,
        scores={"contains": 1.0},
        score=0.0,
        passed=False,
        latency_ms=1.0,
        error=error,
    )
    note = "regressed: baseline 1.000000 -> candidate 0.500000\r\nmore < &"
    passing = CaseResult(
        case_id="ok",
        output="kept",
        scores={"contains": 1.0},
        score=1.0,
        passed=True,
        latency_ms=1.0,
    )
    run = RunResult(
        suite_name="suite\rname",
        config_name="cfg\rname",
        results=[failed, passing],
    )
    xml = render_junit_xml([run], regression_notes={"ok": note}, candidate=run)

    assert "\r" not in xml
    assert "&#13;" in xml

    root = ET.fromstring(xml)
    suite = root.find("testsuite")
    assert suite is not None
    assert suite.get("name") == "cfg\rname"

    cases = {case.get("name"): case for case in suite.findall("testcase")}
    failed_case = cases[case_id]
    assert failed_case.get("classname") == "suite\rname"
    message = failure_message(failed)
    assert "\r" in message
    failure = failed_case.find("failure")
    assert failure is not None
    assert failure.get("message") == message
    assert failure.text == message
    system_out = failed_case.find("system-out")
    assert system_out is not None
    assert system_out.text == output

    noted = cases["ok"]
    noted_out = noted.find("system-out")
    assert noted_out is not None
    assert noted_out.text == note
