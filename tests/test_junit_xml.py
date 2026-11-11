"""JUnit XML from an offline mock compare, plus the failure-message rules."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from promptregress.cli import main
from promptregress.junit_xml import failure_message, render_junit_xml
from promptregress.models import AssertionSpec, Case, CaseResult, RunResult
from promptregress.scorers import score_case

CASE_ID = "id<&\"'"
OUTPUT = "yes < & \" ' end"
SUITE_NAME = "suite<&\"'"
BASE_LABEL = "base<&\"'"
CAND_LABEL = "cand<&\"'"
REFERENCE = "alpha beta gamma delta epsilon zeta"
CANDIDATE_SOFT = "alpha beta gamma delta"


def _compare_suite(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "name": SUITE_NAME,
                "cases": [
                    {
                        "id": "keep",
                        "input": "KEEP",
                        "reference": "KEEP OK",
                        "assertions": [{"kind": "contains", "value": "OK"}],
                    },
                    {
                        "id": CASE_ID,
                        "input": "DROP",
                        "reference": "unused",
                        "assertions": [
                            {"kind": "regex", "value": "yes"},
                            {"kind": "contains", "value": "NEEDLE"},
                        ],
                    },
                    {
                        "id": "soft",
                        "input": "SOFT",
                        "reference": REFERENCE,
                        "assertions": [{"kind": "tfidf_cosine"}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _rules() -> str:
    return json.dumps(
        [
            ["BASE\nKEEP", "KEEP OK"],
            ["CAND\nKEEP", "KEEP OK"],
            ["BASE\nDROP", "yes NEEDLE"],
            ["CAND\nDROP", OUTPUT],
            ["BASE\nSOFT", REFERENCE],
            ["CAND\nSOFT", CANDIDATE_SOFT],
        ]
    )


def test_compare_junit_xml_escapes_counts_and_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = _compare_suite(tmp_path / "suite.json")
    report = tmp_path / "out" / "report.xml"
    code = main(
        [
            "compare",
            str(suite),
            "--provider",
            "mock",
            "--mock-rules",
            _rules(),
            "--baseline-template",
            "BASE\n{input}",
            "--candidate-template",
            "CAND\n{input}",
            "--baseline-config-name",
            BASE_LABEL,
            "--candidate-name",
            CAND_LABEL,
            "--max-regression",
            "0",
            "--junit-xml",
            str(report),
            "--no-history",
        ]
    )
    assert code == 1
    raw = report.read_text(encoding="utf-8")
    assert "&lt;" in raw
    assert "&amp;" in raw
    assert "&quot;" in raw
    assert "id<&" not in raw
    assert "yes < &" not in raw

    root = ET.parse(report).getroot()
    assert root.tag == "testsuites"
    suites = list(root.findall("testsuite"))
    assert [item.get("name") for item in suites] == [BASE_LABEL, CAND_LABEL]
    assert [item.get("classname") for item in suites[0].findall("testcase")] == [SUITE_NAME] * 3

    baseline, candidate = suites
    assert baseline.get("tests") == "3"
    assert baseline.get("failures") == "0"
    assert baseline.get("errors") == "0"
    assert float(baseline.get("time") or "-1") >= 0.0
    assert candidate.get("tests") == "3"
    assert candidate.get("failures") == "1"
    assert candidate.get("errors") == "0"
    assert float(candidate.get("time") or "-1") >= 0.0

    for suite_el in suites:
        assert len(suite_el.findall("testcase")) == int(suite_el.get("tests") or "0")
        failures = [
            item for item in suite_el.findall("testcase") if item.find("failure") is not None
        ]
        assert len(failures) == int(suite_el.get("failures") or "-1")

    base_cases = {item.get("name"): item for item in baseline.findall("testcase")}
    cand_cases = {item.get("name"): item for item in candidate.findall("testcase")}
    assert set(base_cases) == {"keep", CASE_ID, "soft"}
    assert base_cases[CASE_ID].find("failure") is None
    assert base_cases[CASE_ID].find("system-out") is None

    failed = cand_cases[CASE_ID]
    assert failed.get("classname") == SUITE_NAME
    failure = failed.find("failure")
    assert failure is not None
    assert failure.get("message") == "score=0.500000; contains=0.000000"
    assert failure.text == "score=0.500000; contains=0.000000"
    assert failed.find("system-out") is not None
    assert failed.find("system-out").text == OUTPUT
    assert "regressed:" not in (failed.find("system-out").text or "")

    soft_case = Case(
        id="soft",
        input="SOFT",
        reference=REFERENCE,
        assertions=[AssertionSpec(kind="tfidf_cosine")],
    )
    base_score = score_case(soft_case, REFERENCE)[1]
    cand_score, cand_passed = score_case(soft_case, CANDIDATE_SOFT)[1:]
    assert cand_passed is True
    assert cand_score < base_score
    note = f"regressed: baseline {base_score:.6f} -> candidate {cand_score:.6f}"
    soft = cand_cases["soft"]
    assert soft.find("failure") is None
    assert soft.find("system-out") is not None
    assert soft.find("system-out").text == note
    assert base_cases["soft"].find("system-out") is None
    assert base_cases["soft"].find("failure") is None

    keep = cand_cases["keep"]
    assert keep.find("failure") is None
    assert keep.find("system-out") is None


def test_run_junit_xml_keeps_exit_zero_when_a_case_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "one",
                "cases": [
                    {
                        "id": "only",
                        "input": "x",
                        "reference": "y",
                        "assertions": [{"kind": "exact_match"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "run.xml"
    code = main(
        [
            "run",
            str(suite),
            "--provider",
            "echo",
            "--config-name",
            "echo-run",
            "--template",
            "{input}",
            "--out",
            str(tmp_path / "run.json"),
            "--junit-xml",
            str(report),
            "--no-history",
        ]
    )
    assert code == 0
    root = ET.parse(report).getroot()
    suites = list(root.findall("testsuite"))
    assert len(suites) == 1
    assert suites[0].get("name") == "echo-run"
    assert suites[0].get("tests") == "1"
    assert suites[0].get("failures") == "1"
    case = suites[0].find("testcase")
    assert case is not None
    assert case.get("classname") == "one"
    assert case.get("name") == "only"
    failure = case.find("failure")
    assert failure is not None
    assert failure.get("message") == "score=0.000000; exact_match=0.000000"


def test_failure_message_skips_a_passing_kind_and_uses_error() -> None:
    mixed = CaseResult(
        case_id="m",
        output="out",
        scores={"regex": 1.0, "contains": 0.0},
        score=0.5,
        passed=False,
        latency_ms=1.0,
    )
    assert failure_message(mixed) == "score=0.500000; contains=0.000000"
    crashed = CaseResult(
        case_id="c",
        output="",
        scores={},
        score=0.0,
        passed=False,
        latency_ms=2.0,
        error="ProviderError: boom < &",
    )
    assert failure_message(crashed) == "score=0.000000; ProviderError: boom < &"
    empty = CaseResult(
        case_id="e",
        output="",
        scores={},
        score=0.0,
        passed=False,
        latency_ms=0.0,
    )
    assert failure_message(empty) == "score=0.000000; no assertion detail"


def test_junit_strips_xml_controls_and_round_trips_the_rest() -> None:
    result = RunResult(
        suite_name="s",
        config_name="c",
        results=[
            CaseResult(
                case_id="a\x00b",
                output="x\x01y < &",
                scores={"exact_match": 0.0},
                score=0.0,
                passed=False,
                latency_ms=0.0,
            )
        ],
    )
    root = ET.fromstring(render_junit_xml([result]))
    case = root.find("testsuite")
    assert case is not None
    testcase = case.find("testcase")
    assert testcase is not None
    assert testcase.get("name") == "ab"
    system_out = testcase.find("system-out")
    assert system_out is not None
    assert system_out.text == "xy < &"
