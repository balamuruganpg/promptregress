"""Load eval suites from JSON, YAML, JSONL, and CSV."""

import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml

from promptregress.models import AssertionSpec, Case, ScoringConfig, Suite


class SuiteError(ValueError):
    """A suite file is missing, empty, or malformed."""


def suite_schema_path() -> Path:
    """JSON Schema (draft 2020-12) for the JSON/YAML suite document.

    The file is package data (``data/*.json``). ``promptregress schema``
    prints it. ``promptregress validate`` does not apply the schema; it
    loads each suite with :func:`load_suite`.
    """
    return Path(__file__).resolve().parent / "data" / "suite_schema.json"


def load_suite(path: str | Path) -> Suite:
    """Load a suite from ``.json``, ``.yaml``/``.yml``, ``.jsonl``, or ``.csv``.

    JSON and YAML documents are objects with ``name``, optional ``description``,
    optional ``scoring.weights``, and ``cases``. JSONL is one case object per
    line; the suite name is the file stem. CSV requires columns ``id``,
    ``input``, and ``reference`` and attaches an ``exact_match`` assertion
    against the reference.

    Raises ``SuiteError`` on a missing file, an unsupported suffix, malformed
    content (including invalid UTF-8, documents nested past the parser stack,
    and integers the JSON parser refuses), a duplicate case id, or an empty
    suite.
    """
    suite_path = Path(path)
    if not suite_path.is_file():
        raise SuiteError(f"Suite file not found: {suite_path}")
    suffix = suite_path.suffix.lower()
    if suffix == ".json":
        return _load_json(suite_path)
    if suffix in {".yaml", ".yml"}:
        return _load_yaml(suite_path)
    if suffix == ".jsonl":
        return _load_jsonl(suite_path)
    if suffix == ".csv":
        return _load_csv(suite_path)
    raise SuiteError(
        f"Unsupported suite format '{suffix}'. Use .json, .yaml, .yml, .jsonl, or .csv."
    )


def suite_from_mapping(data: Any, default_name: str) -> Suite:
    """Build a suite from a decoded JSON/YAML object."""
    if not isinstance(data, dict):
        raise SuiteError("Suite must be an object with a cases list")
    name = data.get("name", default_name)
    if not isinstance(name, str) or not name.strip():
        raise SuiteError("Suite name must be a non-empty string")
    description_raw = data.get("description", "")
    if description_raw is None:
        description = ""
    elif isinstance(description_raw, str):
        description = description_raw
    else:
        raise SuiteError("Suite description must be a string")
    cases_raw = data.get("cases")
    if not isinstance(cases_raw, list):
        raise SuiteError("Suite is missing a cases list")
    cases = [_parse_case(item, index) for index, item in enumerate(cases_raw, start=1)]
    scoring = _parse_scoring(data.get("scoring"), cases)
    return _finish(name, description, cases, scoring)


def _read_text(path: Path) -> str:
    """Read a suite file as UTF-8, or raise ``SuiteError``."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise SuiteError(f"Suite file {path} is not valid UTF-8: {exc}") from exc


def _load_json(path: Path) -> Suite:
    text = _read_text(path)
    try:
        data = json.loads(text)
    except RecursionError as exc:
        raise SuiteError(f"Malformed JSON in {path}: document is too deeply nested") from exc
    except ValueError as exc:
        # JSONDecodeError, and ValueError when an integer exceeds the
        # interpreter's string-digit limit.
        raise SuiteError(f"Malformed JSON in {path}: {exc}") from exc
    return suite_from_mapping(data, default_name=path.stem)


def _load_yaml(path: Path) -> Suite:
    text = _read_text(path)
    try:
        data = yaml.safe_load(text)
    except RecursionError as exc:
        raise SuiteError(f"Malformed YAML in {path}: document is too deeply nested") from exc
    except yaml.YAMLError as exc:
        raise SuiteError(f"Malformed YAML in {path}: {exc}") from exc
    except ValueError as exc:
        raise SuiteError(f"Malformed YAML in {path}: {exc}") from exc
    return suite_from_mapping(data, default_name=path.stem)


def _load_jsonl(path: Path) -> Suite:
    text = _read_text(path)
    cases: list[Case] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except RecursionError as exc:
            raise SuiteError(
                f"Malformed JSONL in {path} at line {line_number}: document is too deeply nested"
            ) from exc
        except ValueError as exc:
            raise SuiteError(f"Malformed JSONL in {path} at line {line_number}: {exc}") from exc
        cases.append(_parse_case(raw, line_number))
    return _finish(path.stem, "", cases)


def _load_csv(path: Path) -> Suite:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise SuiteError(f"CSV suite {path} is missing a header row")
        fields = {(name or "").strip() for name in fieldnames}
        missing = {"id", "input", "reference"} - fields
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise SuiteError(
                f"CSV suite {path} must have columns id, input, reference; missing {missing_list}"
            )
        cases: list[Case] = []
        for index, row in enumerate(reader, start=1):
            normalized = {(key or "").strip(): value for key, value in row.items()}
            cases.append(_case_from_csv_row(normalized, index))
    return _finish(path.stem, "", cases)


def _case_from_csv_row(row: dict[str, str | None], index: int) -> Case:
    case_id = (row.get("id") or "").strip()
    if not case_id:
        raise SuiteError(f"CSV row {index} is missing id")
    if "input" not in row or row.get("input") is None:
        raise SuiteError(f"CSV row {index} ({case_id}) is missing input")
    question = row.get("input") or ""
    reference = row.get("reference")
    if reference is None:
        reference = ""
    assertion = AssertionSpec(kind="exact_match", value=reference)
    return Case(id=case_id, input=question, reference=reference, assertions=[assertion])


def _finish(
    name: str,
    description: str,
    cases: list[Case],
    scoring: ScoringConfig | None = None,
) -> Suite:
    if not cases:
        raise SuiteError(f"Suite '{name}' is empty")
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise SuiteError(f"Duplicate case id: {case.id}")
        seen.add(case.id)
    return Suite(
        name=name,
        description=description,
        cases=cases,
        scoring=ScoringConfig() if scoring is None else scoring,
    )


def _parse_scoring(raw: Any, cases: list[Case]) -> ScoringConfig:
    """Read ``scoring.weights``. An omitted block is equal weights.

    Each weight multiplies the assertion weights of that scorer kind. Missing
    kinds use 1, so a partial map only changes the kinds it names. Unknown
    kinds, negative weights, and a map whose every weight is 0 are errors.
    """
    if raw is None:
        return ScoringConfig()
    if not isinstance(raw, dict):
        raise SuiteError("Suite scoring must be an object")
    if "weights" not in raw or raw["weights"] is None:
        return ScoringConfig()
    weights_raw = raw["weights"]
    if not isinstance(weights_raw, dict):
        raise SuiteError("scoring.weights must be an object mapping scorer names to numbers")
    weights: dict[str, float] = {}
    for key, value in weights_raw.items():
        if not isinstance(key, str) or not key.strip():
            raise SuiteError("scoring.weights keys must be non-empty scorer names")
        number = _as_number(value, f"scoring weight for {key!r}")
        if number < 0:
            raise SuiteError(
                f"scoring weight for {key!r} is negative ({number}); weights must be non-negative"
            )
        weights[key] = number
    if weights and all(weight == 0.0 for weight in weights.values()):
        raise SuiteError(
            "scoring.weights are all zero; at least one scorer weight must be positive"
        )
    used = {assertion.kind for case in cases for assertion in case.assertions}
    unknown = sorted(set(weights) - used)
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        known = ", ".join(sorted(used)) if used else "(none)"
        label = "scorer" if len(unknown) == 1 else "scorers"
        raise SuiteError(f"Unknown {label} {names} in scoring.weights. Suite scorers: {known}.")
    return ScoringConfig(weights=weights)


def _parse_case(raw: Any, index: int) -> Case:
    if not isinstance(raw, dict):
        raise SuiteError(f"Case {index} is not an object")
    if "id" not in raw:
        raise SuiteError(f"Case {index} is missing id")
    case_id = raw["id"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise SuiteError(f"Case {index} id must be a non-empty string")
    if "input" not in raw:
        raise SuiteError(f"Case {case_id} is missing input")
    question = raw["input"]
    if not isinstance(question, str):
        raise SuiteError(f"Case {case_id} input must be a string")
    if "reference" not in raw or raw["reference"] is None:
        reference: str | None = None
    elif isinstance(raw["reference"], str):
        reference = raw["reference"]
    else:
        raise SuiteError(f"Case {case_id} reference must be a string or null")
    assertions_raw = raw.get("assertions", [])
    if assertions_raw is None:
        assertions_raw = []
    if not isinstance(assertions_raw, list):
        raise SuiteError(f"Case {case_id} assertions must be a list")
    assertions = [
        _parse_assertion(item, case_id, position)
        for position, item in enumerate(assertions_raw, start=1)
    ]
    tags = _parse_tags(raw.get("tags", []), case_id)
    metadata = _parse_metadata(raw.get("metadata", {}), case_id)
    return Case(
        id=case_id,
        input=question,
        reference=reference,
        assertions=assertions,
        tags=tags,
        metadata=metadata,
    )


def _parse_assertion(raw: Any, case_id: str, index: int) -> AssertionSpec:
    if not isinstance(raw, dict):
        raise SuiteError(f"Case {case_id} assertion {index} is not an object")
    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise SuiteError(f"Case {case_id} assertion {index} is missing kind")
    if "weight" in raw and raw["weight"] is not None:
        weight = _as_number(raw["weight"], f"Case {case_id} assertion {index} weight")
    else:
        weight = 1.0
    params_raw = raw.get("params", {})
    if params_raw is None:
        params_raw = {}
    if not isinstance(params_raw, dict):
        raise SuiteError(f"Case {case_id} assertion {index} params must be an object")
    return AssertionSpec(kind=kind, value=raw.get("value"), weight=weight, params=dict(params_raw))


def _parse_tags(raw: Any, case_id: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise SuiteError(f"Case {case_id} tags must be a list of strings")
    return list(raw)


def _parse_metadata(raw: Any, case_id: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SuiteError(f"Case {case_id} metadata must be an object")
    return dict(raw)


def _as_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SuiteError(f"{label} must be a number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise SuiteError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise SuiteError(f"{label} must be finite")
    return number
