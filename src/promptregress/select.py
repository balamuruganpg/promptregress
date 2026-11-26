"""Active case selection under a budget.

A full suite is expensive to run on every candidate. ``select`` keeps the
cases that look most likely to regress, using only what saved comparison
files record, and writes those cases as an ordinary suite.

For each case, the observations are the ``per_case`` rows with that id.
``load_comparison`` rebuilds those rows from the stored runs. One comparison
contributes at most one row per id. A comparison that does not contain the
id is left out of that case's counts. It is not treated as a non-event.

Let ``n`` be the number of observations.

- ``flip_rate = n_flips / n``. A flip is a row whose baseline passed and
  whose candidate failed.
- ``candidate_variance`` is the population variance of the candidate scores.
  The mean is ``sum(scores) / n``. The variance is
  ``sum((score - mean) ** 2) / n``. One observation has variance 0. Both
  sums are ``math.fsum`` of the values sorted ascending, so the order of
  the comparison files does not change the result.
- ``variance_component = candidate_variance / 0.25``. ``0.25`` is the
  population variance of a fair coin on ``{0, 1}``, which is the maximum
  population variance of values in ``[0, 1]``. Scores outside ``[0, 1]``
  can push this above 1. It is not clamped.
- ``mean_drop`` is the mean of ``baseline_score - candidate_score`` over
  rows whose candidate score is strictly below the baseline score, and
  ``0`` when there are no such rows. Gains do not cancel a drop. The sum
  is ``math.fsum`` of those drops sorted ascending. ``compare`` stores a
  difference inside ``1e-12`` as ``diff`` 0. This formula reads the stored
  scores, so a smaller difference still counts.
- ``priority = (flip_rate + variance_component + mean_drop) / 3``. The
  priority is not clamped.

Cases with at least one observation rank by priority descending. Equal
priorities break by case id ascending, in code-point order. There is no
tie band. Cases in the suite with no observations have no priority and
rank after every case with history, then by case id ascending. The first
``budget`` cases in that order are written. A budget at least as large as
the suite writes every case in source order.

History ids that are not in the suite are ignored, counted once, and
listed in code-point order.

``--seed`` is stored on the report. The formula and the sort do not read
it. JSON and YAML outputs set ``name`` to the source suite name plus
``-selected`` and copy each selected case object from the parsed source,
including fields the loader ignores. The file is re-serialized, so
whitespace can change. JSONL and CSV copy the selected case rows the same
way. Those formats have no name field; ``load_suite`` names them from the
output path's stem. The output format family matches the source
(``.yaml`` and ``.yml`` are one family).

Selection is only as good as the history behind it. A case that never
regressed in the recorded comparisons may still regress on a new kind of
change. The ranking does not change the gate.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from promptregress.junit_xml import strip_unpaired_surrogates
from promptregress.loading import load_suite, strip_utf8_bom_text
from promptregress.models import Comparison, PairedRow, Suite
from promptregress.risk import load_comparison

VARIANCE_SCALE = 0.25
DEFAULT_SEED = 13
PRIORITY_FORMULA = "(flip_rate + candidate_variance / 0.25 + mean_drop) / 3"

SELECT_LIMIT = (
    "Selection is only as good as the history behind it. A case that never "
    "regressed in the recorded comparisons may still regress on a new kind of change."
)

SELECT_NOTE = (
    f"priority = {PRIORITY_FORMULA}. "
    "flip_rate is the share of recorded comparisons for that case where the "
    "baseline passed and the candidate failed. candidate_variance is the "
    "population variance of the candidate scores: the sum of squared "
    "deviations from their mean, divided by the number of observations. One "
    "observation has variance 0. Both sums are math.fsum of the values sorted "
    "ascending, so comparison order does not change the score. 0.25 is the "
    "population variance of a fair coin on {0, 1}, the maximum population "
    "variance of values in [0, 1]. Scores outside [0, 1] can push "
    "candidate_variance / 0.25 above 1. The priority is not clamped. "
    "mean_drop is the mean of baseline_score - candidate_score over "
    "observations where the candidate score is strictly lower, and 0 when "
    "there are none. Gains do not cancel a drop. The drop sum is math.fsum "
    "of those values sorted ascending. compare stores a difference inside "
    "1e-12 as diff 0; this formula reads the stored scores, so a smaller "
    "difference still counts. An observation is one per_case row. A "
    "comparison that does not contain the case is left out of that case's "
    "counts. Cases in the suite with no observations have no priority and "
    "rank after every case with history, ordered by case id ascending, in "
    "code-point order. Equal priorities break the same way. There is no tie "
    "band on the priority. History ids that are not in the suite are "
    "ignored, counted once, and listed in code-point order. The seed is "
    "stored on the report and is not an input to the formula. "
    f"{SELECT_LIMIT} "
    "The ranking does not change the gate. The text rounds to 6 decimals. "
    "JSON keeps the stored floats."
)


@dataclass(frozen=True)
class CasePriority:
    """One suite case after ranking.

    ``history`` is false when no comparison recorded the id. The float
    components are then ``None``. ``n_observed`` is 0, and the flip and
    regression counts are 0. A case with history stores the formula inputs.
    ``variance_component`` is ``candidate_variance / 0.25``.
    """

    case_id: str
    history: bool
    priority: float | None
    flip_rate: float | None
    candidate_variance: float | None
    variance_component: float | None
    mean_drop: float | None
    n_observed: int
    n_flips: int
    n_regressed: int


@dataclass(frozen=True)
class Selection:
    """The budgeted selection and the counts behind the report.

    ``selected`` is the written order. ``ignored_history_ids`` is sorted
    ascending in code-point order. ``suite_name`` is the source name plus
    ``-selected``. JSON and YAML files store that name. JSONL and CSV do
    not, and ``load_suite`` names those files from the output stem.
    ``source_suite`` and ``output_suite`` are the paths the caller passed.
    """

    source_suite: str
    output_suite: str
    suite_name: str
    budget: int
    seed: int
    n_comparisons: int
    n_suite: int
    n_with_history: int
    n_without_history: int
    ignored_history_ids: tuple[str, ...]
    selected: tuple[CasePriority, ...]
    note: str


def prioritize(
    suite: Suite,
    comparisons: Sequence[Comparison],
    *,
    budget: int,
    seed: int = DEFAULT_SEED,
    source_suite: str = "",
    output_suite: str = "",
) -> Selection:
    """Rank ``suite`` and keep the first ``budget`` cases.

    Does not write a file. ``seed`` is stored and is not read by the
    formula. An empty suite, a budget below 1, no comparisons, a repeated
    id in one comparison, or a non-finite score raises ``ValueError``.
    """
    _check_seed(seed)
    _check_budget(budget)
    comparisons = _check_comparisons(comparisons)
    case_ids = _suite_ids(suite)
    observed, ignored = _observations(comparisons, set(case_ids))
    ranked = tuple(
        sorted(
            (_priority_for(case_id, observed.get(case_id, ())) for case_id in case_ids),
            key=_rank_key,
        )
    )
    n_with_history = sum(1 for item in ranked if item.history)
    return Selection(
        source_suite=source_suite,
        output_suite=output_suite,
        suite_name=f"{suite.name}-selected",
        budget=budget,
        seed=seed,
        n_comparisons=len(comparisons),
        n_suite=len(ranked),
        n_with_history=n_with_history,
        n_without_history=len(ranked) - n_with_history,
        ignored_history_ids=ignored,
        selected=ranked[:budget],
        note=SELECT_NOTE,
    )


def select_suite(
    suite_path: str | Path,
    comparison_paths: Sequence[str | Path],
    *,
    budget: int,
    out_path: str | Path,
    seed: int = DEFAULT_SEED,
) -> Selection:
    """Load a suite and comparison reports, write the selected suite, and return the report.

    Comparison files are loaded with :func:`promptregress.risk.load_comparison`.
    The output is the same format family as the source. Parent directories
    are not created. The source file is not modified. ``--out`` must be a
    different path. Budget below 1, an empty suite, no comparisons, a
    mismatched format, or an unwritable output raises ``ValueError``.
    """
    _check_seed(seed)
    _check_budget(budget)
    if isinstance(comparison_paths, (str, bytes)):
        raise ValueError("comparisons must be a sequence of paths")
    paths = list(comparison_paths)
    if len(paths) == 0:
        raise ValueError("select needs at least one comparison")
    source = Path(suite_path)
    target = Path(out_path)
    suite = load_suite(source)
    comparisons = [load_comparison(path) for path in paths]
    selection = prioritize(
        suite,
        comparisons,
        budget=budget,
        seed=seed,
        source_suite=str(suite_path),
        output_suite=str(out_path),
    )
    if budget >= selection.n_suite:
        priorities = {item.case_id: item for item in selection.selected}
        selection = replace(selection, selected=tuple(priorities[case.id] for case in suite.cases))
    _write_selected(source, target, suite, [item.case_id for item in selection.selected])
    return selection


def format_selection(report: Selection) -> str:
    """Text report. Ends with a newline.

    Floats print with 6 decimals. A case with no history prints ``-`` for
    each component. Ignored history ids print as a comma-separated list,
    or ``none``.
    """
    if report.ignored_history_ids:
        ignored = ", ".join(report.ignored_history_ids)
    else:
        ignored = "none"
    lines = [
        "select",
        f"suite={report.source_suite}",
        f"out={report.output_suite}",
        f"suite_name={report.suite_name}",
        f"budget={report.budget}",
        f"seed={report.seed}",
        f"n_comparisons={report.n_comparisons}",
        f"n_suite={report.n_suite}",
        f"n_selected={len(report.selected)}",
        f"n_with_history={report.n_with_history}",
        f"n_without_history={report.n_without_history}",
        f"n_ignored_history={len(report.ignored_history_ids)}",
        f"ignored_history_ids={ignored}",
        f"note={report.note}",
    ]
    lines.extend(_format_case(item) for item in report.selected)
    return strip_unpaired_surrogates("\n".join(lines) + "\n")


def selection_to_dict(report: Selection) -> dict[str, object]:
    """JSON-ready report. Floats are the stored values. Missing components are null."""
    return {
        "suite": report.source_suite,
        "out": report.output_suite,
        "suite_name": report.suite_name,
        "budget": report.budget,
        "seed": report.seed,
        "n_comparisons": report.n_comparisons,
        "n_suite": report.n_suite,
        "n_selected": len(report.selected),
        "n_with_history": report.n_with_history,
        "n_without_history": report.n_without_history,
        "n_ignored_history": len(report.ignored_history_ids),
        "ignored_history_ids": list(report.ignored_history_ids),
        "note": report.note,
        "selected": [_case_to_dict(item) for item in report.selected],
    }


def _format_case(item: CasePriority) -> str:
    history = "true" if item.history else "false"
    return (
        f"- {item.case_id} history={history} "
        f"priority={_fmt(item.priority)} "
        f"flip_rate={_fmt(item.flip_rate)} "
        f"candidate_variance={_fmt(item.candidate_variance)} "
        f"variance_component={_fmt(item.variance_component)} "
        f"mean_drop={_fmt(item.mean_drop)} "
        f"n_observed={item.n_observed} n_flips={item.n_flips} n_regressed={item.n_regressed}"
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def _case_to_dict(item: CasePriority) -> dict[str, object]:
    return {
        "case_id": item.case_id,
        "history": item.history,
        "priority": item.priority,
        "flip_rate": item.flip_rate,
        "candidate_variance": item.candidate_variance,
        "variance_component": item.variance_component,
        "mean_drop": item.mean_drop,
        "n_observed": item.n_observed,
        "n_flips": item.n_flips,
        "n_regressed": item.n_regressed,
    }


def _check_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")


def _check_budget(budget: int) -> None:
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be an integer >= 1")


def _check_comparisons(comparisons: Sequence[Comparison]) -> list[Comparison]:
    if isinstance(comparisons, (str, bytes)):
        raise ValueError("comparisons must be a sequence of comparisons")
    items = list(comparisons)
    if len(items) == 0:
        raise ValueError("select needs at least one comparison")
    for index, comparison in enumerate(items, start=1):
        if not isinstance(comparison, Comparison):
            raise ValueError(f"comparisons[{index}] must be a Comparison")
    return items


def _suite_ids(suite: Suite) -> list[str]:
    if not isinstance(suite, Suite):
        raise ValueError("suite must be a Suite")
    if not isinstance(suite.name, str) or suite.name == "":
        raise ValueError("suite name must be a non-empty string")
    if len(suite.cases) == 0:
        raise ValueError("suite is empty")
    seen: set[str] = set()
    ids: list[str] = []
    for case in suite.cases:
        case_id = case.id
        if not isinstance(case_id, str) or case_id == "":
            raise ValueError("case id must be a non-empty string")
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        ids.append(case_id)
    return ids


@dataclass(frozen=True)
class _Observation:
    baseline_score: float
    candidate_score: float
    flipped: bool


def _observations(
    comparisons: Sequence[Comparison],
    suite_ids: set[str],
) -> tuple[dict[str, tuple[_Observation, ...]], tuple[str, ...]]:
    grouped: dict[str, list[_Observation]] = {}
    for index, comparison in enumerate(comparisons, start=1):
        rows = comparison.per_case
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise ValueError(f"comparison {index} per_case must be a sequence of paired rows")
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, PairedRow):
                raise ValueError(f"comparison {index} per_case must be paired rows")
            case_id = row.case_id
            if not isinstance(case_id, str) or case_id == "":
                raise ValueError(f"comparison {index} case id must be a non-empty string")
            if case_id in seen:
                raise ValueError(f"comparison {index} repeats case id {case_id!r}")
            seen.add(case_id)
            grouped.setdefault(case_id, []).append(_observation(index, row))
    ignored = tuple(sorted(set(grouped) - suite_ids))
    return {case_id: tuple(rows) for case_id, rows in grouped.items()}, ignored


def _observation(index: int, row: PairedRow) -> _Observation:
    baseline_score = _finite(
        row.baseline_score, f"comparison {index} case {row.case_id!r} baseline score"
    )
    candidate_score = _finite(
        row.candidate_score,
        f"comparison {index} case {row.case_id!r} candidate score",
    )
    if not isinstance(row.baseline_passed, bool) or not isinstance(row.candidate_passed, bool):
        raise ValueError(f"comparison {index} case {row.case_id!r} passed flags must be booleans")
    return _Observation(
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        flipped=row.baseline_passed and not row.candidate_passed,
    )


def _finite(value: object, label: str) -> float:
    message = f"{label} must be a finite number"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(message) from exc
    if not math.isfinite(number):
        raise ValueError(message)
    return number


def _priority_for(case_id: str, rows: tuple[_Observation, ...]) -> CasePriority:
    if not rows:
        return CasePriority(
            case_id=case_id,
            history=False,
            priority=None,
            flip_rate=None,
            candidate_variance=None,
            variance_component=None,
            mean_drop=None,
            n_observed=0,
            n_flips=0,
            n_regressed=0,
        )
    n_flips = sum(1 for row in rows if row.flipped)
    flip_rate = n_flips / len(rows)
    variance = _population_variance([row.candidate_score for row in rows])
    variance_component = variance / VARIANCE_SCALE
    drops = [
        row.baseline_score - row.candidate_score
        for row in rows
        if row.candidate_score < row.baseline_score
    ]
    mean_drop = _mean_of(drops) if drops else 0.0
    priority = (flip_rate + variance_component + mean_drop) / 3.0
    for label, value in (
        ("flip_rate", flip_rate),
        ("candidate_variance", variance),
        ("variance_component", variance_component),
        ("mean_drop", mean_drop),
        ("priority", priority),
    ):
        if not math.isfinite(value):
            raise ValueError(f"case {case_id!r} {label} is not finite")
    return CasePriority(
        case_id=case_id,
        history=True,
        priority=priority,
        flip_rate=flip_rate,
        candidate_variance=variance,
        variance_component=variance_component,
        mean_drop=mean_drop,
        n_observed=len(rows),
        n_flips=n_flips,
        n_regressed=len(drops),
    )


def _population_variance(values: Sequence[float]) -> float:
    """Population variance. Sums are ``math.fsum`` of the values sorted ascending."""
    count = len(values)
    if count == 0:
        raise ValueError("variance needs at least one score")
    mean = _mean_of(values)
    return _mean_of([_square(value - mean) for value in values])


def _mean_of(values: Sequence[float]) -> float:
    """Mean via ``math.fsum`` of the values sorted ascending, divided by the count.

    ``fsum`` raises ``OverflowError`` when the sum leaves the float range.
    The mean of finite values is finite, so an equal series returns that
    value and any other series divides before summing.
    """
    count = len(values)
    if count == 0:
        raise ValueError("mean needs at least one value")
    ordered = sorted(values)
    try:
        return math.fsum(ordered) / count
    except OverflowError:
        first = ordered[0]
        if all(value == first for value in ordered):
            return float(first)
        try:
            return math.fsum(value / count for value in ordered)
        except OverflowError:
            return math.inf


def _square(value: float) -> float:
    """``value ** 2``. A square outside the float range is infinity."""
    try:
        return value**2
    except OverflowError:
        return math.inf


def _rank_key(item: CasePriority) -> tuple[int, float, str]:
    priority = item.priority
    if item.history:
        if priority is None or not math.isfinite(priority):
            raise ValueError(f"case {item.case_id!r} has history but no priority")
        return (0, -priority, item.case_id)
    return (1, 0.0, item.case_id)


def _write_selected(
    source: Path,
    target: Path,
    suite: Suite,
    case_ids: Sequence[str],
) -> None:
    if _same_path(source, target):
        raise ValueError("--out must be a different path from --suite")
    if target.exists() and target.is_dir():
        raise ValueError(f"cannot write suite to {target}: path is a directory")
    family = _require_same_family(source, target)
    text = _render_selected(source, suite, case_ids, family)
    try:
        # newline="" keeps a CR inside a quoted CSV field. The default
        # translator would rewrite it when os.linesep is not "\n".
        target.write_text(text, encoding="utf-8", newline="")
    except UnicodeEncodeError as exc:
        raise ValueError(f"cannot write suite to {target}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"cannot write suite to {target}: {exc}") from exc


def _same_path(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        # A missing or inaccessible path still needs the name comparison.
        pass
    try:
        return left.resolve() == right.resolve()
    except (OSError, RuntimeError):
        return left.absolute() == right.absolute()


def _family(suffix: str) -> str:
    if suffix == ".json":
        return "json"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    raise ValueError(
        f"unsupported suite format {suffix!r}. Use .json, .yaml, .yml, .jsonl, or .csv."
    )


def _require_same_family(source: Path, target: Path) -> str:
    source_family = _family(source.suffix.lower())
    try:
        target_family = _family(target.suffix.lower())
    except ValueError as exc:
        raise ValueError(
            f"output {target} is {target.suffix or '(no suffix)'}; select writes the "
            "same format family as the source suite (.json, .yaml/.yml, .jsonl, or .csv)"
        ) from exc
    if source_family != target_family:
        raise ValueError(
            f"output {target} is {target.suffix}, but source suite {source} is "
            f"{source.suffix}; select writes the same format family "
            "(.yaml and .yml are one family)"
        )
    return target_family


def _read_source(path: Path, *, csv_dialect: bool) -> str:
    try:
        if csv_dialect:
            # Same contract as the CSV loader: newline="" keeps CR and CRLF
            # inside quoted fields. Path.read_text translates those away.
            with path.open(encoding="utf-8-sig", newline="") as handle:
                raw = handle.read()
        else:
            raw = path.read_text(encoding="utf-8-sig")
    except UnicodeError as exc:
        raise ValueError(f"suite file {path} is not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"cannot read suite {path}: {exc}") from exc
    return strip_utf8_bom_text(raw, backslash_escapes=not csv_dialect)


def _render_selected(
    source: Path,
    suite: Suite,
    case_ids: Sequence[str],
    family: str,
) -> str:
    text = _read_source(source, csv_dialect=family == "csv")
    if family == "json":
        return _render_json(text, suite, case_ids)
    if family == "yaml":
        return _render_yaml_suite(text, suite, case_ids)
    if family == "jsonl":
        return _render_jsonl(text, case_ids)
    if family == "csv":
        return _render_csv(text, case_ids)
    raise ValueError(f"unsupported suite format family {family!r}")


def _render_json(text: str, suite: Suite, case_ids: Sequence[str]) -> str:
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"cannot read suite JSON: {exc}") from exc
    document = _selected_document(data, suite, case_ids)
    try:
        body = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot serialize suite: {exc}") from exc
    return body + "\n"


def _render_yaml_suite(text: str, suite: Suite, case_ids: Sequence[str]) -> str:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"cannot read suite YAML: {exc}") from exc
    document = _selected_document(data, suite, case_ids)
    try:
        dumped = yaml.dump(
            document,
            Dumper=_QuotedDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    except yaml.YAMLError as exc:
        raise ValueError(f"cannot serialize suite: {exc}") from exc
    if not isinstance(dumped, str):
        raise ValueError("cannot serialize suite as YAML")
    if not dumped.endswith("\n"):
        return dumped + "\n"
    return dumped


def _selected_document(data: object, suite: Suite, case_ids: Sequence[str]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("suite must be an object with a cases list")
    cases_raw = data.get("cases")
    if not isinstance(cases_raw, list):
        raise ValueError("suite is missing a cases list")
    by_id: dict[str, Any] = {}
    for item in cases_raw:
        if not isinstance(item, dict):
            raise ValueError("suite cases must be objects")
        case_id = item.get("id")
        if not isinstance(case_id, str) or case_id == "":
            raise ValueError("suite case id must be a non-empty string")
        by_id[case_id] = item
    selected: list[Any] = []
    for case_id in case_ids:
        item = by_id.get(case_id)
        if item is None:
            raise ValueError(f"selected case {case_id!r} is not in the source suite")
        selected.append(item)
    document = dict(data)
    document["name"] = f"{suite.name}-selected"
    document["cases"] = selected
    return document


def _render_jsonl(text: str, case_ids: Sequence[str]) -> str:
    by_id: dict[str, Any] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip().lstrip("\ufeff")
        if stripped == "":
            continue
        try:
            item = json.loads(stripped)
        except ValueError as exc:
            raise ValueError(f"malformed JSONL at line {line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"JSONL line {line_number} is not a case object")
        case_id = item.get("id")
        if not isinstance(case_id, str) or case_id == "":
            raise ValueError(f"JSONL line {line_number} case id must be a non-empty string")
        by_id[case_id] = item
    lines: list[str] = []
    for case_id in case_ids:
        item = by_id.get(case_id)
        if item is None:
            raise ValueError(f"selected case {case_id!r} is not in the source suite")
        try:
            lines.append(json.dumps(item, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cannot serialize case {case_id!r}: {exc}") from exc
    # splitlines() in the loader also recognises these Unicode separators.
    body = "\n".join(lines)
    for separator in ("\u2028", "\u2029", "\u0085"):
        body = body.replace(separator, f"\\u{ord(separator):04x}")
    return body + "\n"


def _render_csv(text: str, case_ids: Sequence[str]) -> str:
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = reader.fieldnames
    if fieldnames is None:
        raise ValueError("CSV suite is missing a header row")
    by_id: dict[str, dict[str, str | None]] = {}
    for row in reader:
        normalized = {(key or "").strip(): value for key, value in row.items() if key is not None}
        case_id = (normalized.get("id") or "").strip()
        if case_id == "":
            raise ValueError("CSV row is missing id")
        by_id[case_id] = {key: value for key, value in row.items() if key is not None}
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for case_id in case_ids:
        selected_row = by_id.get(case_id)
        if selected_row is None:
            raise ValueError(f"selected case {case_id!r} is not in the source suite")
        payload: dict[str, str] = {}
        for name in fieldnames:
            cell = selected_row.get(name)
            payload[name] = "" if cell is None else cell
        writer.writerow(payload)
    return buffer.getvalue()


class _QuotedDumper(yaml.SafeDumper):
    """Quote every string so YAML 1.1 does not reread ``yes`` as a bool."""


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


_QuotedDumper.add_representer(str, _represent_str)
