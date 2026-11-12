"""Regression tests for bugs found in the independent v0.6.13 review cycle.

Scope is incomplete BOM handling after 0.6.12–0.6.13: JSONL readers only
stripped a leading file BOM, so a second Windows-saved chunk glued mid-file
still broke suite/calibration JSONL and silently dropped history rows; matrix
YAML still used plain ``utf-8``, so a double BOM corrupted the first config
name or raised a cryptic parse error, and invalid UTF-8 raised a bare
``UnicodeDecodeError`` instead of ``MatrixError``. Run-report and power
``--from-report`` loaders had the same bare decode gap as pre-0.6.13 pricing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptregress.calibration import load_calibration_jsonl
from promptregress.history import load_history, make_run_record
from promptregress.irt import load_run_report
from promptregress.loading import load_suite
from promptregress.matrix import MatrixError, load_matrix
from promptregress.models import CaseResult, RunResult
from promptregress.power import load_report_diffs

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


def test_jsonl_suite_strips_bom_on_each_line(tmp_path: Path) -> None:
    """Concatenated JSONL chunks each carrying a UTF-8 BOM must still load.

    ``_read_text`` strips only a leading file BOM. ``cat`` of two
    Windows-saved JSONL files leaves ``U+FEFF`` at the start of later lines,
    so ``json.loads`` raised ``Unexpected UTF-8 BOM`` on line 2 and
    ``load_suite`` aborted. Strip a residual line-leading BOM the same way
    the file-level reader does so every case loads with the right
    ``source_line``.
    """
    row_a = '{"id": "a", "input": "q", "assertions": []}\n'
    row_b = '{"id": "b", "input": "q", "assertions": []}\n'
    path = tmp_path / "cases.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + row_a.encode() + b"\xef\xbb\xbf" + row_b.encode())
    suite = load_suite(path)
    assert [(case.id, case.source_line) for case in suite.cases] == [("a", 1), ("b", 2)]


def test_calibration_and_history_jsonl_strip_midfile_bom(tmp_path: Path) -> None:
    """Calibration and history JSONL must keep rows after a mid-file BOM.

    The same concatenated-BOM pattern that broke suite JSONL made
    ``load_calibration_jsonl`` fail on line 2 and made ``load_history`` skip
    that line as invalid JSON (silently dropping a run from trend). Per-line
    BOM stripping keeps every object.
    """
    cal_row = json.dumps({"input": "q", "reference": "r", "output": "o", "human_score": 1.0}) + "\n"
    cal_path = tmp_path / "labels.jsonl"
    cal_path.write_bytes(b"\xef\xbb\xbf" + cal_row.encode() + b"\xef\xbb\xbf" + cal_row.encode())
    rows = load_calibration_jsonl(cal_path)
    assert len(rows) == 2
    assert rows[1].human_score == 1.0

    record = make_run_record(_run(), provider="mock", model="m", template="{input}")
    hist_line = json.dumps(record) + "\n"
    hist_path = tmp_path / "history.jsonl"
    hist_path.write_bytes(
        b"\xef\xbb\xbf" + hist_line.encode() + b"\xef\xbb\xbf" + hist_line.encode()
    )
    records, notes = load_history(hist_path)
    assert notes == []
    assert len(records) == 2
    assert records[1]["suite"] == "s"


def test_matrix_double_bom_keeps_config_names(tmp_path: Path) -> None:
    """A double UTF-8 BOM on a matrix YAML must not rename the first config.

    Suite/pricing YAML go through ``utf-8-sig`` + residual BOM strip. Matrix
    still used plain ``utf-8``. PyYAML treats one leading ``U+FEFF`` as a BOM
    but a second sticks on the first key, so ``a`` loaded as ``\\ufeffa`` and
    ranking/labels were wrong. A BOM before a blank first line instead raised
    a cryptic YAML error. Strip BOMs like the other loaders.
    """
    body = (
        "a:\n  provider: echo\n  template: '{input}'\nb:\n  provider: echo\n  template: '{input}'\n"
    )
    path = tmp_path / "matrix.yaml"
    path.write_bytes(b"\xef\xbb\xbf\xef\xbb\xbf" + body.encode("utf-8"))
    configs = load_matrix(path)
    assert [config.name for config in configs] == ["a", "b"]

    blank_first = tmp_path / "blank.yaml"
    blank_first.write_bytes(b"\xef\xbb\xbf\xef\xbb\xbf\n" + body.encode("utf-8"))
    assert [config.name for config in load_matrix(blank_first)] == ["a", "b"]


def test_matrix_invalid_utf8_raises_matrix_error(tmp_path: Path) -> None:
    """Malformed UTF-8 in a matrix file must raise ``MatrixError``.

    Pricing and CSV suites wrap decode errors as their documented types.
    ``load_matrix`` let ``UnicodeDecodeError`` escape, so callers that handle
    ``MatrixError`` (and the CLI path that reports config failures) missed it.
    """
    path = tmp_path / "matrix.yaml"
    path.write_bytes(
        b"a:\n  provider: echo\n  template: '{input}'\n"
        b"b:\n  provider: echo\n  template: '{input}'\n\xff"
    )
    with pytest.raises(MatrixError, match="not valid UTF-8"):
        load_matrix(path)


def test_run_report_power_and_calibration_invalid_utf8_raise_value_error(
    tmp_path: Path,
) -> None:
    """Invalid UTF-8 in run/compare/calibration files must raise ``ValueError``.

    0.6.13 wrapped CSV/pricing decode failures. ``load_run_report``,
    ``load_report_diffs``, and ``load_calibration_jsonl`` still raised bare
    ``UnicodeDecodeError`` (a ``ValueError`` subclass with a codec message)
    before their JSON checks, so ``irt``, ``power --from-report``, and
    ``calibrate`` did not surface the same clear ``not valid UTF-8`` failure.
    """
    run_path = tmp_path / "run.json"
    run_path.write_bytes(
        b'{"suite_name":"s","config_name":"c","results":'
        b'[{"case_id":"a","score":1.0,"passed":true,"latency_ms":1.0}]}\xff'
    )
    with pytest.raises(ValueError, match="not valid UTF-8") as run_exc:
        load_run_report(run_path)
    assert type(run_exc.value) is ValueError

    compare_path = tmp_path / "compare.json"
    compare_path.write_bytes(b'{"per_case":[{"diff":0.0}]}\xff')
    with pytest.raises(ValueError, match="not valid UTF-8") as power_exc:
        load_report_diffs(compare_path)
    assert type(power_exc.value) is ValueError

    cal_path = tmp_path / "labels.jsonl"
    cal_path.write_bytes(b'{"input":"q","reference":"r","output":"o","human_score":1.0}\n\xff')
    with pytest.raises(ValueError, match="not valid UTF-8") as cal_exc:
        load_calibration_jsonl(cal_path)
    assert type(cal_exc.value) is ValueError
