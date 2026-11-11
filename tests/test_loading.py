"""Suite loader: JSON, YAML, JSONL, CSV, and the error paths."""

import json
from pathlib import Path

import pytest

from promptregress.loading import SuiteError, load_suite

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_json_fields(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "suite.json",
        json.dumps(
            {
                "name": "demo",
                "description": "A small suite",
                "cases": [
                    {
                        "id": "a",
                        "input": "hello",
                        "reference": "hello",
                        "assertions": [{"kind": "exact_match", "weight": 2}],
                        "tags": ["greet"],
                        "metadata": {"n": 1},
                    }
                ],
            }
        ),
    )
    suite = load_suite(path)
    assert suite.name == "demo"
    assert suite.description == "A small suite"
    case = suite.cases[0]
    assert case.id == "a"
    assert case.input == "hello"
    assert case.reference == "hello"
    assert case.tags == ["greet"]
    assert case.metadata == {"n": 1}
    assert case.assertions[0].kind == "exact_match"
    assert case.assertions[0].weight == 2.0
    assert case.assertions[0].params == {}
    assert case.assertions[0].value is None


def test_load_yaml(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "suite.yaml",
        """
name: yaml-suite
description: from yaml
cases:
  - id: a
    input: hello
    reference: hello
    assertions:
      - kind: contains
        value: hello
    tags: [greet]
    metadata:
      n: 1
""".lstrip(),
    )
    suite = load_suite(path)
    assert suite.name == "yaml-suite"
    assert suite.description == "from yaml"
    assert suite.cases[0].assertions[0].kind == "contains"
    assert suite.cases[0].assertions[0].value == "hello"
    assert suite.cases[0].metadata["n"] == 1


def test_load_yml_suffix(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "suite.yml",
        "name: y\ncases:\n  - id: a\n    input: q\n    assertions: []\n",
    )
    suite = load_suite(path)
    assert suite.name == "y"
    assert suite.cases[0].assertions == []


def test_load_jsonl(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "items.jsonl",
        "\n".join(
            [
                "",
                json.dumps(
                    {
                        "id": "a",
                        "input": "hello",
                        "reference": "hello",
                        "assertions": [{"kind": "exact_match"}],
                    }
                ),
                json.dumps(
                    {
                        "id": "b",
                        "input": "world",
                        "reference": "world",
                        "assertions": [{"kind": "contains", "value": "world"}],
                    }
                ),
                "",
            ]
        ),
    )
    suite = load_suite(path)
    assert suite.name == "items"
    assert suite.description == ""
    assert [case.id for case in suite.cases] == ["a", "b"]
    assert suite.cases[1].assertions[0].kind == "contains"


def test_load_csv(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "rows.csv",
        'id,input,reference,note\na,"hello, world","hello, world",extra\nb,second,second,x\n',
    )
    suite = load_suite(path)
    assert suite.name == "rows"
    assert suite.cases[0].id == "a"
    assert suite.cases[0].input == "hello, world"
    assert suite.cases[0].reference == "hello, world"
    assert suite.cases[0].assertions[0].kind == "exact_match"
    assert suite.cases[0].assertions[0].value == "hello, world"
    assert suite.cases[1].id == "b"


def test_duplicate_id(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "dup.json",
        json.dumps(
            {
                "name": "dup",
                "cases": [
                    {"id": "a", "input": "1"},
                    {"id": "a", "input": "2"},
                ],
            }
        ),
    )
    with pytest.raises(SuiteError, match="Duplicate case id: a"):
        load_suite(path)


def test_empty_suite(tmp_path: Path) -> None:
    path = _write(tmp_path / "empty.json", json.dumps({"name": "empty", "cases": []}))
    with pytest.raises(SuiteError, match="empty"):
        load_suite(path)


def test_malformed_json(tmp_path: Path) -> None:
    path = _write(tmp_path / "bad.json", "{not json")
    with pytest.raises(SuiteError, match="Malformed JSON"):
        load_suite(path)


def test_malformed_yaml(tmp_path: Path) -> None:
    path = _write(tmp_path / "bad.yaml", "cases: [\n")
    with pytest.raises(SuiteError, match="Malformed YAML"):
        load_suite(path)


def test_malformed_jsonl_line(tmp_path: Path) -> None:
    path = _write(tmp_path / "bad.jsonl", '{"id": "a", "input": "q"}\n{nope\n')
    with pytest.raises(SuiteError, match="line 2"):
        load_suite(path)


def test_missing_input(tmp_path: Path) -> None:
    path = _write(tmp_path / "miss.json", json.dumps({"name": "m", "cases": [{"id": "a"}]}))
    with pytest.raises(SuiteError, match="missing input"):
        load_suite(path)


def test_assertion_missing_kind(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "miss.json",
        json.dumps({"name": "m", "cases": [{"id": "a", "input": "q", "assertions": [{}]}]}),
    )
    with pytest.raises(SuiteError, match="missing kind"):
        load_suite(path)


def test_missing_cases_key(tmp_path: Path) -> None:
    path = _write(tmp_path / "miss.json", json.dumps({"name": "m"}))
    with pytest.raises(SuiteError, match="cases"):
        load_suite(path)


def test_top_level_list(tmp_path: Path) -> None:
    path = _write(tmp_path / "list.json", "[]")
    with pytest.raises(SuiteError, match="object"):
        load_suite(path)


def test_csv_missing_column(tmp_path: Path) -> None:
    path = _write(tmp_path / "rows.csv", "id,input\na,hello\n")
    with pytest.raises(SuiteError, match="reference"):
        load_suite(path)


def test_csv_header_only(tmp_path: Path) -> None:
    path = _write(tmp_path / "rows.csv", "id,input,reference\n")
    with pytest.raises(SuiteError, match="empty"):
        load_suite(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="not found"):
        load_suite(tmp_path / "nope.json")


def test_unsupported_suffix(tmp_path: Path) -> None:
    path = _write(tmp_path / "suite.txt", "hello")
    with pytest.raises(SuiteError, match="Unsupported"):
        load_suite(path)


def test_example_suite_has_at_least_12_cases() -> None:
    suite = load_suite(ROOT / "examples" / "support_qa.json")
    assert suite.name == "support_qa"
    assert len(suite.cases) >= 12
    kinds = {assertion.kind for case in suite.cases for assertion in case.assertions}
    assert {
        "exact_match",
        "contains",
        "regex",
        "json_valid",
        "json_schema",
        "numeric_tolerance",
    } <= kinds
    inputs = [case.input for case in suite.cases]
    for left_index, left in enumerate(inputs):
        for right_index, right in enumerate(inputs):
            if left_index != right_index:
                assert left not in right
    bundled = (ROOT / "src" / "promptregress" / "data" / "support_qa.json").read_text(
        encoding="utf-8"
    )
    example = (ROOT / "examples" / "support_qa.json").read_text(encoding="utf-8")
    assert bundled == example
