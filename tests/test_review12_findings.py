"""Regression tests for bugs found in the independent v0.6.11 review cycle.

Scope is UTF-8 BOM handling on suite and pricing loaders (JSON/JSONL/CSV
and pricing JSON reject a leading BOM while YAML suites already loaded),
which blocks ``load_suite`` and ``--pricing`` / cost budgets on files
saved from common Windows editors.
"""

from __future__ import annotations

import json
from pathlib import Path

from promptregress.annotations import case_annotation
from promptregress.loading import load_suite
from promptregress.models import CaseResult
from promptregress.pricing import load_pricing

_BOM = "\ufeff"


def test_json_suite_with_utf8_bom_loads_and_keeps_source_lines(tmp_path: Path) -> None:
    """A JSON suite prefixed with a UTF-8 BOM must load, not raise SuiteError.

    ``json.loads`` rejects a leading ``U+FEFF`` (``Unexpected UTF-8 BOM``).
    Suite text was read with plain ``utf-8``, so Windows-saved ``.json``
    files failed ``load_suite`` while the same document without a BOM
    worked, and YAML suites with a BOM already loaded via PyYAML.
    Reading with ``utf-8-sig`` strips the BOM; compose marks still locate
    each case ``id`` for ``--github-annotations``.
    """
    path = tmp_path / "suite.json"
    body = {
        "name": "s",
        "cases": [
            {"id": "a", "input": "x", "assertions": []},
            {"id": "b", "input": "y", "assertions": []},
        ],
    }
    path.write_text(_BOM + json.dumps(body, indent=2) + "\n", encoding="utf-8")
    suite = load_suite(path)
    assert [(case.id, case.source_line) for case in suite.cases] == [("a", 5), ("b", 10)]
    failed = CaseResult(
        case_id="b",
        output="y",
        scores={"contains": 0.0},
        score=0.0,
        passed=False,
        latency_ms=1.0,
        source_line=suite.cases[1].source_line,
    )
    assert ",line=10::" in case_annotation(failed, str(path))


def test_jsonl_and_csv_suites_with_utf8_bom_load(tmp_path: Path) -> None:
    """JSONL and CSV suites with a leading BOM must load like YAML.

    JSONL failed on line 1 with the same ``json.loads`` BOM error. CSV kept
    the BOM on the first header (``\\ufeffid``), so the loader reported
    ``missing id`` even though the file had the right columns. Both must
    strip the BOM and record ``source_line`` for annotations.
    """
    jsonl = tmp_path / "cases.jsonl"
    jsonl.write_text(
        _BOM
        + '{"id": "a", "input": "q", "assertions": []}\n'
        + '{"id": "b", "input": "q", "assertions": []}\n',
        encoding="utf-8",
    )
    jsonl_suite = load_suite(jsonl)
    assert [(case.id, case.source_line) for case in jsonl_suite.cases] == [("a", 1), ("b", 2)]

    csv_path = tmp_path / "cases.csv"
    csv_path.write_text(_BOM + "id,input,reference\na,q,q\nb,r,r\n", encoding="utf-8")
    csv_suite = load_suite(csv_path)
    assert [(case.id, case.source_line) for case in csv_suite.cases] == [("a", 2), ("b", 3)]


def test_pricing_json_with_utf8_bom_loads(tmp_path: Path) -> None:
    """A pricing JSON file with a UTF-8 BOM must load for ``--pricing``.

    Cost budgets reuse ``load_pricing``. The same BOM that broke suite JSON
    also made ``json.loads`` reject a Windows-saved pricing table, so
    ``compare --pricing`` / ``run --pricing`` exited 2 before any cost
    budget could run. ``utf-8-sig`` matches the suite loader.
    """
    path = tmp_path / "prices.json"
    path.write_text(
        _BOM + json.dumps({"mock": {"prompt_per_1k": 1.0, "completion_per_1k": 2.0}}) + "\n",
        encoding="utf-8",
    )
    prices = load_pricing(path)
    assert prices["mock"].prompt_per_1k == 1.0
    assert prices["mock"].completion_per_1k == 2.0
