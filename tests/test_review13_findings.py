"""Regression tests for bugs found in the independent v0.6.12 review cycle.

Scope is incomplete UTF-8 BOM / decode handling after 0.6.12 fixed suite and
pricing loaders: other user-facing JSON/JSONL readers (stored baselines /
run reports, calibration, power ``--from-report``, history) still rejected
a leading BOM; a double BOM still broke suite/pricing JSON and CSV headers
(``utf-8-sig`` strips only one); and invalid UTF-8 in CSV/pricing raised
bare ``UnicodeDecodeError`` instead of ``SuiteError`` / ``PricingError``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptregress.baseline import load_baseline, save_baseline
from promptregress.calibration import load_calibration_jsonl
from promptregress.gate import evaluate_gate
from promptregress.history import load_history, make_run_record
from promptregress.loading import SuiteError, load_suite
from promptregress.models import CaseResult, Comparison, PairedRow, RunResult
from promptregress.power import load_report_diffs
from promptregress.pricing import PricingError, load_pricing
from promptregress.report import write_json

_BOM = "\ufeff"


def _run() -> RunResult:
    return RunResult(
        suite_name="s",
        config_name="c",
        results=[
            CaseResult(
                case_id="a",
                output="x",
                scores={"contains": 1.0},
                score=1.0,
                passed=True,
                latency_ms=1.0,
            )
        ],
    )


def test_baseline_run_report_with_utf8_bom_loads(tmp_path: Path) -> None:
    """A stored baseline JSON with a UTF-8 BOM must load for ``--baseline-name``.

    0.6.12 taught suite and pricing loaders to use ``utf-8-sig``, but
    ``load_run_report`` (used by ``load_baseline``, ``irt``, and related
    commands) still read plain ``utf-8``. A baseline re-saved from a Windows
    editor then failed ``compare --baseline-name`` with ``Unexpected UTF-8
    BOM`` instead of scoring the candidate. Match the suite/pricing rule.
    """
    path, _overwritten = save_baseline(_run(), "base1", cwd=tmp_path)
    path.write_text(_BOM + path.read_text(encoding="utf-8"), encoding="utf-8")
    loaded = load_baseline("base1", cwd=tmp_path)
    assert loaded.suite_name == "s"
    assert loaded.results[0].case_id == "a"


def test_double_bom_suite_and_pricing_still_load(tmp_path: Path) -> None:
    """A double UTF-8 BOM must not leave a residual ``U+FEFF`` for ``json.loads``.

    ``utf-8-sig`` strips only the first BOM byte sequence. A second BOM
    (common after a ``utf-8-sig`` re-save of text that already began with
    ``U+FEFF``) stayed as a leading character, so JSON suites and pricing
    tables still raised ``Unexpected UTF-8 BOM``. Stripping residual leading
    ``U+FEFF`` after decode fixes that; CSV goes through the same text path
    so the ``id`` header is not ``\\ufeffid``.
    """
    body = json.dumps({"name": "s", "cases": [{"id": "a", "input": "x", "assertions": []}]})
    suite_path = tmp_path / "suite.json"
    suite_path.write_bytes(b"\xef\xbb\xbf\xef\xbb\xbf" + body.encode("utf-8"))
    suite = load_suite(suite_path)
    assert suite.cases[0].id == "a"
    assert suite.cases[0].source_line == 1

    prices_path = tmp_path / "prices.json"
    prices_path.write_bytes(
        b"\xef\xbb\xbf\xef\xbb\xbf"
        + json.dumps({"mock": {"prompt_per_1k": 1.0, "completion_per_1k": 2.0}}).encode()
    )
    prices = load_pricing(prices_path)
    assert prices["mock"].prompt_per_1k == 1.0

    csv_path = tmp_path / "cases.csv"
    csv_path.write_bytes(b"\xef\xbb\xbf\xef\xbb\xbfid,input,reference\na,q,q\n")
    csv_suite = load_suite(csv_path)
    assert [(case.id, case.source_line) for case in csv_suite.cases] == [("a", 2)]


def test_calibration_jsonl_with_utf8_bom_loads(tmp_path: Path) -> None:
    """Calibration JSONL with a leading BOM must load like suite JSONL.

    ``load_calibration_jsonl`` read plain ``utf-8``, so line 1 of a
    Windows-saved labels file failed with the same BOM error suite JSONL
    had before 0.6.12. ``calibrate`` then exited before scoring.
    """
    path = tmp_path / "labels.jsonl"
    path.write_text(
        _BOM
        + json.dumps({"input": "q", "reference": "r", "output": "o", "human_score": 1.0})
        + "\n",
        encoding="utf-8",
    )
    rows = load_calibration_jsonl(path)
    assert len(rows) == 1
    assert rows[0].input == "q"
    assert rows[0].human_score == 1.0


def test_power_compare_report_with_utf8_bom_loads(tmp_path: Path) -> None:
    """``power --from-report`` must accept a compare JSON file with a BOM.

    ``load_report_diffs`` used plain ``utf-8``. A report opened and saved in
    a Windows editor became unreadable, so power analysis exited 2 instead
    of reading per-case diffs. Same ``utf-8-sig`` rule as other JSON loaders.
    """
    run = _run()
    paired = PairedRow(
        case_id="a",
        baseline_output="x",
        candidate_output="x",
        baseline_score=1.0,
        candidate_score=1.0,
        diff=0.0,
        baseline_passed=True,
        candidate_passed=True,
        outcome="tie",
    )
    comparison = Comparison(
        baseline=run,
        candidate=run,
        mean_diff=0.0,
        ci_low=0.0,
        ci_high=0.0,
        mcnemar_p=1.0,
        wins=0,
        losses=0,
        ties=1,
        per_case=[paired],
    )
    path = tmp_path / "compare.json"
    write_json(comparison, path, evaluate_gate(comparison))
    path.write_text(_BOM + path.read_text(encoding="utf-8"), encoding="utf-8")
    assert load_report_diffs(path) == [0.0]


def test_history_jsonl_with_utf8_bom_keeps_first_record(tmp_path: Path) -> None:
    """A leading BOM on history JSONL must not drop the first record.

    ``load_history`` skipped malformed lines. A BOM on the file made line 1
    look like invalid JSON (``Unexpected UTF-8 BOM``), so that run was
    omitted from ``history`` / ``trend`` while later lines loaded. Strip the
    BOM the same way as other JSONL readers so the row is kept.
    """
    path = tmp_path / "history.jsonl"
    record = make_run_record(
        _run(),
        provider="mock",
        model="m",
        template="{input}",
    )
    path.write_text(_BOM + json.dumps(record) + "\n", encoding="utf-8")
    records, notes = load_history(path)
    assert notes == []
    assert len(records) == 1
    assert records[0]["suite"] == "s"
    assert records[0]["command"] == "run"


def test_csv_invalid_utf8_raises_suite_error(tmp_path: Path) -> None:
    """CSV suites with invalid UTF-8 must raise ``SuiteError``, not a codec error.

    JSON/YAML/JSONL go through ``_read_text``, which wraps ``UnicodeError``.
    CSV used ``path.open(encoding=...)`` directly, so a bad byte raised
    ``UnicodeDecodeError`` and contradicted ``load_suite``'s documented
    error type. Reading CSV via ``_read_text`` keeps the same ``SuiteError``.
    """
    path = tmp_path / "suite.csv"
    path.write_bytes(b"id,input,reference\na,q,\xff\n")
    with pytest.raises(SuiteError, match="not valid UTF-8"):
        load_suite(path)


def test_pricing_invalid_utf8_raises_pricing_error(tmp_path: Path) -> None:
    """Malformed UTF-8 in a pricing file must raise ``PricingError``.

    ``load_pricing`` caught ``OSError`` around ``read_text`` but not
    ``UnicodeError``, so a truncated multi-byte sequence raised a bare
    ``UnicodeDecodeError``. Callers that handle ``PricingError`` (and the
    module contract that a bad file raises that type) missed it.
    """
    path = tmp_path / "prices.json"
    path.write_bytes(b'{"m":{"prompt_per_1k":1,"completion_per_1k":2}}\xff')
    with pytest.raises(PricingError, match="not valid UTF-8"):
        load_pricing(path)
