"""Regression tests for bugs found in the independent v0.6.2 review cycle.

Scope is the 0.6.1–0.6.2 surface: JUnit XML export, suite audit, and the
interactive HTML report.
"""

from __future__ import annotations

from xml.etree.ElementTree import fromstring

from promptregress.junit_xml import render_junit_xml
from promptregress.models import CaseResult, RunResult


def test_junit_strips_xml10_surrogates_so_output_round_trips() -> None:
    """Unpaired surrogates are XML 1.0-forbidden and must be stripped.

    ElementTree encodes them as numeric character references (``&#55296;``),
    which every strict parser — including ElementTree on the way back —
    rejects. C0 controls and U+FFFE/U+FFFF were already removed; surrogates
    in the Cs range were not, so a case id or completion containing one
    produced a file consumers could not parse.
    """
    result = RunResult(
        suite_name="s\ud800",
        config_name="c\udfff",
        results=[
            CaseResult(
                case_id="a\ud800b",
                output="x\ud800y < &",
                scores={"exact_match": 0.0},
                score=0.0,
                passed=False,
                latency_ms=0.0,
                error="boom\ud800",
            )
        ],
    )
    raw = render_junit_xml([result])
    assert "&#55296;" not in raw
    assert "&#57343;" not in raw
    root = fromstring(raw)
    suite = root.find("testsuite")
    assert suite is not None
    assert suite.get("name") == "c"
    case = suite.find("testcase")
    assert case is not None
    assert case.get("classname") == "s"
    assert case.get("name") == "ab"
    failure = case.find("failure")
    assert failure is not None
    assert failure.get("message") == "score=0.000000; exact_match=0.000000"
    system_out = case.find("system-out")
    assert system_out is not None
    assert system_out.text == "xy < &"
