"""Load eval suites from JSON, YAML, JSONL, and CSV."""

import csv
import io
import json
import math
from dataclasses import replace
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

    Each case records ``source_line`` when the file locates it: the JSONL
    line, the CSV record's last physical line, or (for JSON and YAML) the
    compose-mark line of that case's ``id`` key so nested ``id`` fields and
    block-scalar text cannot steal another case's line. A case whose id
    field is not found keeps ``source_line`` unset.
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


def strip_utf8_bom_text(text: str, *, backslash_escapes: bool = True) -> str:
    """Strip leading and per-line residual ``U+FEFF`` characters.

    ``utf-8-sig`` removes one leading UTF-8 BOM on decode. A second BOM
    (utf-8-sig re-save) stays as a leading ``U+FEFF``. Concatenating
    Windows-saved chunks leaves ``U+FEFF`` at the start of later lines.
    Both must go before JSON/YAML/CSV parse so keys and case ids are not
    renamed to ``\\ufeff…``.

    Line-leading ``U+FEFF`` inside a quoted region is kept: CSV and YAML
    quoted scalars may continue on the next physical line, and a
    legitimate ZWNBSP (or pasted ``U+FEFF``) at the start of that
    continuation must not be deleted. Quote state must survive escapes
    used by those formats.

    ``backslash_escapes`` selects the quote dialect:

    - ``True`` (JSON / YAML / pretty-printed reports): inside double
      quotes, ``\\`` escapes the next character (so ``\\"`` does not end
      the region). Treating only CSV ``""`` as an escape desynchronises
      on model output that contains a quote, so a later chunk BOM is
      kept (JSON ``loads`` fails) or an in-field ``U+FEFF`` is deleted.
      YAML single-quoted scalars are tracked too (``''`` escapes a
      quote; backslash is literal). An opener ``'`` is recognised only
      after a token boundary so a plain-scalar apostrophe (``it's``)
      does not leave quote-state open and hide a later chunk BOM.
    - ``False`` (CSV): only ``""`` escapes a double quote. A field such
      as ``"C:\\\\temp\\\\"`` ends at the quote after the trailing
      backslash; treating that ``\\"`` as an escape would keep a later
      line-leading BOM and rename the next case id to ``\\ufeff…``.
    """
    text = text.lstrip("\ufeff")
    if "\ufeff" not in text:
        return text
    parts: list[str] = []
    in_double = False
    in_single = False
    for line in text.splitlines(keepends=True):
        if not in_double and not in_single:
            line = line.lstrip("\ufeff")
        index = 0
        length = len(line)
        while index < length:
            char = line[index]
            if in_double:
                # JSON/YAML: backslash escapes the next character.
                if backslash_escapes and char == "\\" and index + 1 < length:
                    index += 2
                    continue
                # CSV: doubled quote is a literal quote.
                if char == '"' and index + 1 < length and line[index + 1] == '"':
                    index += 2
                    continue
                if char == '"':
                    in_double = False
                index += 1
                continue
            if in_single:
                # YAML single-quoted: '' is an escaped quote; \\ is literal.
                if char == "'" and index + 1 < length and line[index + 1] == "'":
                    index += 2
                    continue
                if char == "'":
                    in_single = False
                index += 1
                continue
            if char == '"':
                in_double = True
            elif backslash_escapes and char == "'" and _yaml_single_quote_opener(line, index):
                in_single = True
            index += 1
        parts.append(line)
    return "".join(parts)


def _yaml_single_quote_opener(line: str, index: int) -> bool:
    """True when ``'`` at ``index`` can open a YAML single-quoted scalar.

    Plain scalars often contain apostrophes (``it's``). Those must not
    toggle quote-state, or a later mid-file BOM would be kept. A real
    opener sits after whitespace or a YAML token boundary (``:``, ``-``,
    flow indicators).
    """
    if index == 0:
        return True
    return line[index - 1] in " \t:,-[]{},"


def _read_text(path: Path, *, backslash_escapes: bool = True) -> str:
    """Read a suite file as UTF-8, or raise ``SuiteError``.

    ``utf-8-sig`` strips a leading BOM so JSON/JSONL suites saved from
    Windows editors load the same way YAML already did under plain
    ``utf-8`` (PyYAML tolerates ``U+FEFF``; ``json.loads`` does not).
    A residual leading ``U+FEFF`` (double BOM from a utf-8-sig re-save)
    and mid-file line-leading BOMs (concatenated Windows chunks) are
    stripped too so ``json.loads`` / CSV headers / YAML keys stay clean.

    CSV callers pass ``backslash_escapes=False`` so a trailing backslash
    in a quoted field does not desynchronise BOM stripping.
    """
    try:
        return strip_utf8_bom_text(
            path.read_text(encoding="utf-8-sig"),
            backslash_escapes=backslash_escapes,
        )
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
    suite = suite_from_mapping(data, default_name=path.stem)
    # Same compose-mark attachment as YAML: a regex scan of the raw text
    # false-matches ``"id": …`` inside metadata, schemas, and other nested
    # objects, so annotation lines would point at the wrong case.
    return _attach_compose_source_lines(suite, text)


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
    suite = suite_from_mapping(data, default_name=path.stem)
    # Regex scanning of the raw text false-matches ``id: …`` inside block
    # scalars (case input). Compose-node marks point at the real id fields.
    return _attach_compose_source_lines(suite, text)


def _load_jsonl(path: Path) -> Suite:
    text = _read_text(path)
    cases: list[Case] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        # Per-line residual BOM: concatenating Windows-saved JSONL chunks
        # leaves U+FEFF at the start of later lines after the file-level strip.
        line = line.lstrip("\ufeff")
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
        cases.append(_parse_case(raw, line_number, source_line=line_number))
    return _finish(path.stem, "", cases)


def _load_csv(path: Path) -> Suite:
    # Reuse ``_read_text`` (utf-8-sig + residual BOM strip) so a leading or
    # double BOM does not glue onto the ``id`` header. Plain utf-8 made the
    # first column ``\\ufeffid`` and reported "missing id". CSV must not
    # treat ``\\`` as an escape: a quoted Windows directory path ending in
    # ``\\`` would keep a later chunk BOM on the next row's id.
    reader = csv.DictReader(io.StringIO(_read_text(path, backslash_escapes=False)))
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
        cases.append(_case_from_csv_row(normalized, index, source_line=reader.line_num))
    return _finish(path.stem, "", cases)


def _case_from_csv_row(
    row: dict[str, str | None],
    index: int,
    *,
    source_line: int | None = None,
) -> Case:
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
    return Case(
        id=case_id,
        input=question,
        reference=reference,
        assertions=[assertion],
        source_line=source_line,
    )


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


def _attach_compose_source_lines(suite: Suite, text: str) -> Suite:
    """Set ``source_line`` from compose marks on each case's ``id`` key.

    Used for JSON and YAML. Compose walks the structural tree, so an ``id``
    inside a nested object (metadata, schema) or a literal/folded block
    scalar (case input) is not treated as a case id field. When the tree
    shape does not match the loaded suite, or compose raises (including
    ``RecursionError`` on nesting ``json.loads`` still accepts), lines stay
    unset.
    """
    try:
        root = yaml.compose(text)
    except (yaml.YAMLError, RecursionError):
        # ``json.loads`` tolerates deeper nesting than pure-Python compose.
        # A valid suite with a deep ``json_schema`` (or similar) must still
        # load; skip marks instead of aborting with RecursionError.
        return suite
    marks = _compose_case_id_lines(root)
    if marks is None or len(marks) != len(suite.cases):
        return suite
    located = [
        replace(case, source_line=line) for case, line in zip(suite.cases, marks, strict=True)
    ]
    return replace(suite, cases=located)


def _compose_case_id_lines(root: object) -> list[int | None] | None:
    """1-based line of each case's ``id`` key, or ``None`` when shape mismatches."""
    if not isinstance(root, yaml.MappingNode):
        return None
    cases_node: object | None = None
    # Last ``cases`` key wins, matching ``json.loads`` / ``safe_load``.
    for key, value in root.value:
        if isinstance(key, yaml.ScalarNode) and key.value == "cases":
            cases_node = value
    if not isinstance(cases_node, yaml.SequenceNode):
        return None
    lines: list[int | None] = []
    for case_node in cases_node.value:
        if not isinstance(case_node, yaml.MappingNode):
            lines.append(None)
            continue
        found: int | None = None
        # Last ``id`` key wins, matching ``json.loads`` / ``safe_load``.
        for key, _value in case_node.value:
            if isinstance(key, yaml.ScalarNode) and key.value == "id":
                found = key.start_mark.line + 1
        lines.append(found)
    return lines


def _parse_case(raw: Any, index: int, *, source_line: int | None = None) -> Case:
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
        source_line=source_line,
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
