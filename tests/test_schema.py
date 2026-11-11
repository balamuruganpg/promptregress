"""Suite JSON Schema and the validate command.

The checker covers the draft 2020-12 keywords this schema uses. It is test-only
and is not a general validator. ``promptregress validate`` uses the loader.
"""

import json
import re
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

from promptregress.cli import main
from promptregress.loading import SuiteError, load_suite, suite_schema_path

ROOT = Path(__file__).resolve().parents[1]

_KNOWN = {
    "$comment",
    "$defs",
    "$id",
    "$ref",
    "$schema",
    "additionalProperties",
    "const",
    "description",
    "items",
    "maximum",
    "maxItems",
    "maxLength",
    "maxProperties",
    "minItems",
    "minLength",
    "minimum",
    "minProperties",
    "not",
    "pattern",
    "properties",
    "propertyNames",
    "required",
    "title",
    "type",
}


def _walk(schema: object) -> None:
    if schema is True or schema is False or not isinstance(schema, dict):
        return
    unknown = set(schema) - _KNOWN
    assert not unknown, unknown
    for key, value in schema.items():
        if key in {"properties", "$defs"}:
            assert isinstance(value, dict)
            for sub in value.values():
                _walk(sub)
        elif key in {"items", "additionalProperties", "propertyNames", "not"}:
            _walk(value)


def _type_ok(instance: object, declared: object) -> bool:
    names = declared if isinstance(declared, list) else [declared]
    return any(_one_type(instance, name) for name in names)


def _one_type(instance: object, name: object) -> bool:
    if name == "object":
        return isinstance(instance, dict)
    if name == "array":
        return isinstance(instance, list)
    if name == "string":
        return isinstance(instance, str)
    if name == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if name == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if name == "boolean":
        return isinstance(instance, bool)
    if name == "null":
        return instance is None
    raise AssertionError(name)


def _const_eq(instance: object, expected: object) -> bool:
    if isinstance(instance, bool) or isinstance(expected, bool):
        return isinstance(instance, bool) and isinstance(expected, bool) and instance is expected
    if isinstance(instance, (int, float)) and isinstance(expected, (int, float)):
        return instance == expected
    return instance == expected


def _resolve(root: dict[str, Any], ref: str) -> object:
    assert ref.startswith("#/")
    node: object = root
    for part in ref[2:].split("/"):
        assert isinstance(node, dict)
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def schema_errors(
    schema: object,
    instance: object,
    root: dict[str, Any],
    path: str = "$",
) -> list[str]:
    if schema is True:
        return []
    if schema is False:
        return [f"{path}: rejected"]
    if not isinstance(schema, dict):
        return [f"{path}: schema is not an object"]
    if "$ref" in schema:
        return schema_errors(_resolve(root, str(schema["$ref"])), instance, root, path)
    if "type" in schema and not _type_ok(instance, schema["type"]):
        return [f"{path}: expected {schema['type']}"]
    found: list[str] = []
    if "const" in schema and not _const_eq(instance, schema["const"]):
        found.append(f"{path}: const")
    if "not" in schema and not schema_errors(schema["not"], instance, root, path):
        found.append(f"{path}: excluded")
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            found.append(f"{path}: minLength")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            found.append(f"{path}: maxLength")
        if "pattern" in schema and re.search(str(schema["pattern"]), instance) is None:
            found.append(f"{path}: pattern")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            found.append(f"{path}: minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            found.append(f"{path}: maximum")
    if isinstance(instance, dict):
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            found.append(f"{path}: minProperties")
        if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
            found.append(f"{path}: maxProperties")
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in instance:
                    found.append(f"{path}: missing {key}")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        if "propertyNames" in schema:
            for key in instance:
                found.extend(schema_errors(schema["propertyNames"], key, root, f"{path}/{key}"))
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in properties:
                found.extend(schema_errors(properties[key], value, root, f"{path}/{key}"))
            elif additional is False:
                found.append(f"{path}: unexpected {key}")
            elif additional is not True:
                found.extend(schema_errors(additional, value, root, f"{path}/{key}"))
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            found.append(f"{path}: minItems")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            found.append(f"{path}: maxItems")
        if "items" in schema:
            for index, item in enumerate(instance):
                found.extend(schema_errors(schema["items"], item, root, f"{path}/{index}"))
    return found


def _schema() -> dict[str, Any]:
    payload = json.loads(suite_schema_path().read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    _walk(payload)
    return payload


def test_schema_command_prints_draft_2020_12(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["schema"]) == 0
    text = capsys.readouterr().out
    payload = json.loads(text)
    assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'promptregress = ["py.typed", "data/*.json"]' in pyproject
    packaged = files("promptregress").joinpath("data/suite_schema.json").read_text(encoding="utf-8")
    assert packaged == text


def test_schema_accepts_bundled_example_suites() -> None:
    schema = _schema()
    paths = sorted((ROOT / "examples").glob("*.json"))
    paths.append(ROOT / "src" / "promptregress" / "data" / "support_qa.json")
    assert paths
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert schema_errors(schema, document, schema) == []
        load_suite(path)


def test_schema_checker_covers_weights_and_empty_cases(tmp_path: Path) -> None:
    schema = _schema()
    empty = {"name": "n", "cases": []}
    assert schema_errors(schema, empty, schema)
    negative = {
        "name": "n",
        "scoring": {"weights": {"contains": -1}},
        "cases": [{"id": "a", "input": "q", "assertions": [{"kind": "contains", "value": "q"}]}],
    }
    assert schema_errors(schema, negative, schema)
    zeros = {
        "name": "n",
        "scoring": {"weights": {"contains": 0, "regex": 0}},
        "cases": [{"id": "a", "input": "q", "assertions": [{"kind": "contains", "value": "q"}]}],
    }
    assert schema_errors(schema, zeros, schema)
    # The loader rejects an unused scorer kind. Draft 2020-12 cannot see the
    # assertion kinds from the weight map, so the schema allows this document.
    unused = {
        "name": "n",
        "scoring": {"weights": {"nope": 1}},
        "cases": [{"id": "a", "input": "q", "assertions": [{"kind": "contains", "value": "q"}]}],
    }
    assert schema_errors(schema, unused, schema) == []
    unused_path = tmp_path / "unused.json"
    unused_path.write_text(json.dumps(unused), encoding="utf-8")
    with pytest.raises(SuiteError, match="Unknown scorer"):
        load_suite(unused_path)


def test_validate_command_pass_and_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = ROOT / "examples" / "support_qa.json"
    bad = tmp_path / "bad.json"
    bad.write_text("{}\n", encoding="utf-8")
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text(
        json.dumps({"id": "a", "input": "q", "assertions": [{"kind": "exact_match"}]}) + "\n",
        encoding="utf-8",
    )
    assert main(["validate", str(good), str(jsonl)]) == 0
    ok = capsys.readouterr().out
    assert f"{good}: OK" in ok
    assert f"{jsonl}: OK" in ok
    code = main(["validate", str(good), str(bad)])
    assert code == 1
    mixed = capsys.readouterr().out
    assert f"{good}: OK" in mixed
    assert f"{bad}: error:" in mixed
    assert "cases" in mixed
