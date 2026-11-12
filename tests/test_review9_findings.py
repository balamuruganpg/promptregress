"""Regression tests for bugs found in the independent v0.6.8 review cycle.

Scope is compose-mark ``source_line`` attachment for annotations: duplicate
top-level ``cases`` keys must follow last-wins like the loader, and JSON
must not treat nested ``"id"`` keys (metadata / schemas) as case id fields.
"""

from __future__ import annotations

import json
from pathlib import Path

from promptregress.annotations import case_annotation
from promptregress.loading import load_suite
from promptregress.models import CaseResult


def test_yaml_source_line_uses_last_duplicate_cases_key(tmp_path: Path) -> None:
    """Duplicate top-level ``cases`` must annotate the last block, not the first.

    ``safe_load`` keeps the last ``cases`` sequence. Compose-mark attachment
    stopped at the first ``cases`` key; when both sequences had the same
    length, ``source_line`` came from the discarded block and
    ``--github-annotations`` highlighted the wrong lines. Last ``cases``
    wins, matching the loaded suite.
    """
    path = tmp_path / "suite.yaml"
    path.write_text(
        "name: s\n"
        "cases:\n"
        "  - id: old\n"
        "    input: x\n"
        "    assertions: []\n"
        "cases:\n"
        "  - id: new\n"
        "    input: y\n"
        "    assertions: []\n",
        encoding="utf-8",
    )
    suite = load_suite(path)
    assert len(suite.cases) == 1
    case = suite.cases[0]
    assert case.id == "new"
    assert case.source_line == 7

    failed = CaseResult(
        case_id=case.id,
        output="y",
        scores={"contains": 0.0},
        score=0.0,
        passed=False,
        latency_ms=1.0,
        source_line=case.source_line,
    )
    line = case_annotation(failed, str(path))
    assert ",line=7::" in line
    assert ",line=3::" not in line


def test_json_source_line_ignores_nested_id_keys(tmp_path: Path) -> None:
    """JSON annotation ``line`` must be the case ``id`` field, not nested ids.

    Regex attachment scanned for ``"id": "<value>"`` anywhere in the file.
    A later case id that also appeared under an earlier case's ``metadata``
    or assertion ``value`` (for example a ``json_schema`` document) was
    matched first, so ``--github-annotations`` pointed at the nested key.
    Compose-node marks locate only each case object's own ``id`` field.
    """
    path = tmp_path / "suite.json"
    path.write_text(
        "{\n"
        '  "name": "s",\n'
        '  "cases": [\n'
        "    {\n"
        '      "id": "a",\n'
        '      "input": "x",\n'
        '      "assertions": [],\n'
        '      "metadata": {\n'
        '        "id": "b"\n'
        "      }\n"
        "    },\n"
        "    {\n"
        '      "id": "b",\n'
        '      "input": "y",\n'
        '      "assertions": []\n'
        "    }\n"
        "  ]\n"
        "}\n",
        encoding="utf-8",
    )
    suite = load_suite(path)
    assert [(case.id, case.source_line) for case in suite.cases] == [("a", 5), ("b", 13)]

    failed = CaseResult(
        case_id="b",
        output="y",
        scores={"contains": 0.0},
        score=0.0,
        passed=False,
        latency_ms=1.0,
        source_line=suite.cases[1].source_line,
    )
    line = case_annotation(failed, str(path))
    assert ",line=13::" in line
    assert ",line=9::" not in line

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "name": "s",
                "cases": [
                    {
                        "id": "a",
                        "input": "x",
                        "assertions": [{"kind": "json_schema", "value": {"id": "b"}}],
                    },
                    {"id": "b", "input": "y", "assertions": []},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    schema_suite = load_suite(schema_path)
    assert [(case.id, case.source_line) for case in schema_suite.cases] == [
        ("a", 5),
        ("b", 17),
    ]
