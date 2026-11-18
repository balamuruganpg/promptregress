"""Pareto frontier over prompt and model variants.

A variant is one named run: a mean score, and a mean cost and/or a mean
latency. Score is maximised. Cost and latency are minimised. Variant A
dominates variant B when A is at least as good on every axis in use and
strictly better on at least one. The frontier is the set of variants that
nobody dominates.

Equal values on every axis in use do not dominate each other, so both
variants stay on the frontier. The comparison is exact. There is no
``1e-12`` tie band, unlike ``compare``.

Cost is the run's estimated USD (``cost.usd``). A missing ``cost`` object,
or ``cost.usd`` of null, is a missing cost. Latency is ``RunResult``'s mean
of per-case ``latency_ms``. An empty run contributes score 0 and latency 0,
which is what that aggregate already reports. A cache hit's 0 ms is a real
latency, not a missing one.

When every variant is missing cost, or every variant is missing latency, the
frontier uses the axes that remain and the report names the omitted axis.
Score is always required. A mix, where some variants have an axis and others
do not, raises ``ValueError`` and names the variants and the axis. A
non-finite value, or a bool, raises ``ValueError`` and names the variant and
the axis. Both cost and latency missing is an error: one number is not a
frontier.

``dominated_by`` is the closest variant that dominates this one. That
variant may itself be dominated. Distance is Euclidean after each axis is
oriented so that larger is better and then scaled to ``[0, 1]`` by its
minimum and maximum in the set. An axis whose minimum equals its maximum
contributes 0. Equal distances break by variant name, ascending, using
ordinary code-point order. The same name in two files is an error.

Rows are ordered for display. Frontier rows come first: score descending,
then each minimised axis that is in use, ascending, then name ascending.
Dominated rows follow: ``dominated_by`` ascending, then that same key.
Adding or dropping a variant can change who is dominated and which
dominator is closest, because both the frontier and the scaling are
computed on the set you pass.

The inputs are comparison JSON from ``compare --report-json`` (baseline and
candidate) and run JSON from ``run --out``. Matrix ranking JSON stores a
mean score and does not store per-config cost or latency, so it is rejected.
The frontier describes only these runs. It is not a confidence interval,
and it does not change the gate.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from promptregress.irt import run_from_dict
from promptregress.junit_xml import strip_unpaired_surrogates
from promptregress.loading import strip_utf8_bom_text
from promptregress.models import RunResult

PARETO_NOTE = (
    "A variant is dominated when another is at least as good on every axis "
    "in use and strictly better on one. Score is maximised. Cost is the "
    "estimated USD on the run and is minimised. Latency is the mean case "
    "latency in milliseconds and is minimised. Equal values on every axis "
    "in use do not dominate, so both variants stay on the frontier. There "
    "is no 1e-12 tie band. If cost is missing on every variant, or latency "
    "is, that axis is left out and named in omitted. A null cost.usd counts "
    "as missing cost. A mix of present and missing on one axis is an error. "
    "dominated_by is the closest dominating variant, which may itself be "
    "dominated. Distance is Euclidean after each axis is oriented so larger "
    "is better and scaled to [0, 1] by its minimum and maximum in this set. "
    "An axis with no spread contributes 0. Equal distances break by variant "
    "name, ascending, in code-point order. Frontier rows are listed first, "
    "by score descending, then by each minimised axis in use, ascending, "
    "then by name ascending. Dominated rows follow, by dominated_by "
    "ascending, then by that same key. The table rounds; JSON keeps the "
    "stored floats. The frontier describes only these runs. It is not a "
    "confidence interval, and it does not change the gate."
)

_AXIS_SCORE = "score"
_AXIS_COST = "cost"
_AXIS_LATENCY = "latency"
_MINIMISE = (_AXIS_COST, _AXIS_LATENCY)


@dataclass(frozen=True)
class VariantPoint:
    """One variant's mean score and the optional cost and latency axes.

    ``cost`` is estimated USD. ``None`` means the run did not record a
    numeric cost. ``latency_ms`` is mean case latency. ``None`` means latency
    was not supplied. ``source`` is the file the point was loaded from, used
    in error text. Hand-built points may leave it empty.
    """

    name: str
    score: float
    cost: float | None = None
    latency_ms: float | None = None
    source: str = ""


@dataclass(frozen=True)
class ParetoRow:
    """One variant in display order."""

    name: str
    score: float
    cost: float | None
    latency_ms: float | None
    on_frontier: bool
    dominated_by: str | None


@dataclass(frozen=True)
class ParetoReport:
    """The frontier for one set of variants.

    ``axes`` always starts with ``score`` and then lists whichever of
    ``cost`` and ``latency`` were present on every variant. ``omitted``
    names an axis that was missing on every variant. ``rows`` is the
    display order documented on :data:`PARETO_NOTE`.
    """

    axes: tuple[str, ...]
    omitted: tuple[str, ...]
    rows: tuple[ParetoRow, ...]
    note: str = PARETO_NOTE


def pareto_frontier(points: Sequence[VariantPoint]) -> ParetoReport:
    """Return the non-dominated set and the closest dominator of each loser.

    See the module docstring for dominance, the two-axis fallback, distance,
    and the display order. Raises ``ValueError`` on an empty set, a missing
    or non-finite axis, or a repeated variant name.
    """
    if len(points) == 0:
        raise ValueError("pareto needs at least one variant")
    cleaned = [_validate(point) for point in points]
    _reject_duplicates(cleaned)
    axes, omitted = _choose_axes(cleaned)
    on_frontier = _frontier_mask(cleaned, axes)
    closest = _closest_map(cleaned, axes, on_frontier)
    frontier = sorted(
        (point for index, point in enumerate(cleaned) if on_frontier[index]),
        key=lambda point: _order_key(point, axes),
    )
    dominated = sorted(
        (point for index, point in enumerate(cleaned) if not on_frontier[index]),
        key=lambda point: (closest[point.name], *_order_key(point, axes)),
    )
    rows = tuple(_row(point, on=True, dominated_by=None) for point in frontier) + tuple(
        _row(point, on=False, dominated_by=closest[point.name]) for point in dominated
    )
    return ParetoReport(axes=axes, omitted=omitted, rows=rows)


def dominates(winner: VariantPoint, loser: VariantPoint, axes: Sequence[str]) -> bool:
    """True when ``winner`` is at least as good on every axis and better on one.

    ``axes`` is a subset of ``score``, ``cost``, and ``latency``. Score is
    maximised. Cost and latency are minimised. The comparison is exact.
    """
    if not axes:
        raise ValueError("dominates needs at least one axis")
    strict = False
    for axis in axes:
        left = _axis_value(winner, axis)
        right = _axis_value(loser, axis)
        at_least, better = _at_least_as_good(left, right, maximize=(axis == _AXIS_SCORE))
        if not at_least:
            return False
        strict = strict or better
    return strict


def point_from_run(run: RunResult, *, source: str) -> VariantPoint:
    """Build a point from a loaded run's mean score, cost, and mean latency.

    ``cost.usd`` of null, and a run with no cost estimate, are a missing
    cost. The mean score and mean latency are the run's own aggregates.
    """
    cost = None if run.cost is None or run.cost.usd is None else run.cost.usd
    return _validate(
        VariantPoint(
            name=run.config_name,
            score=run.mean_score,
            cost=cost,
            latency_ms=run.mean_latency_ms,
            source=source,
        )
    )


def load_variant_file(path: str | Path) -> list[VariantPoint]:
    """Load points from one comparison report or one run report.

    A comparison contributes the baseline and then the candidate. A run
    contributes its one config. Matrix ranking JSON raises ``ValueError``
    because it has no per-config cost or latency.
    """
    target = Path(path)
    payload = _read_json_object(target)
    return _points_from_payload(payload, target)


def load_variants(paths: Sequence[str | Path]) -> list[VariantPoint]:
    """Load every file, in the order given.

    Duplicate names are reported by :func:`pareto_frontier`, which sees the
    whole set.
    """
    points: list[VariantPoint] = []
    for path in paths:
        points.extend(load_variant_file(path))
    return points


def pareto_from_files(paths: Sequence[str | Path]) -> ParetoReport:
    """Load the files and compute the frontier."""
    return pareto_frontier(load_variants(paths))


def format_pareto(report: ParetoReport) -> str:
    """Text table. Ends with a newline.

    Score and latency print with 6 decimals. Cost prints with 8, matching
    the other USD reports. A missing axis prints as ``-``.
    """
    omitted = "none" if not report.omitted else ",".join(report.omitted)
    header = ("variant", "score", "cost", "latency", "frontier", "dominated_by")
    body = [_cells(row) for row in report.rows]
    widths = [len(column) for column in header]
    for row in body:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def fmt(row: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()

    lines = [
        "pareto",
        f"n={len(report.rows)}",
        f"axes={','.join(report.axes)}",
        f"omitted={omitted}",
        f"note={report.note}",
        "",
        fmt(header),
        *(fmt(row) for row in body),
    ]
    return strip_unpaired_surrogates("\n".join(lines) + "\n")


def pareto_to_dict(report: ParetoReport) -> dict[str, object]:
    """JSON-ready report. ``variants`` is the display order."""
    return {
        "n": len(report.rows),
        "axes": list(report.axes),
        "omitted": list(report.omitted),
        "note": report.note,
        "variants": [_variant_dict(row) for row in report.rows],
    }


def write_pareto_json(payload: Mapping[str, object], path: str | Path) -> None:
    """Write ``payload`` as indented JSON. Rejects NaN and Infinity."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    target.write_text(strip_unpaired_surrogates(text), encoding="utf-8")


def _variant_dict(row: ParetoRow) -> dict[str, object]:
    dominated = None if row.dominated_by is None else strip_unpaired_surrogates(row.dominated_by)
    return {
        "name": strip_unpaired_surrogates(row.name),
        "score": row.score,
        "cost": row.cost,
        "latency_ms": row.latency_ms,
        "on_frontier": row.on_frontier,
        "dominated_by": dominated,
    }


def _cells(row: ParetoRow) -> tuple[str, str, str, str, str, str]:
    return (
        row.name.replace("\n", " "),
        _format_number(row.score, 6),
        _format_number(row.cost, 8),
        _format_number(row.latency_ms, 6),
        "yes" if row.on_frontier else "no",
        "-" if row.dominated_by is None else row.dominated_by.replace("\n", " "),
    )


def _format_number(value: float | None, digits: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _row(point: VariantPoint, *, on: bool, dominated_by: str | None) -> ParetoRow:
    return ParetoRow(
        name=point.name,
        score=point.score,
        cost=point.cost,
        latency_ms=point.latency_ms,
        on_frontier=on,
        dominated_by=dominated_by,
    )


def _validate(point: VariantPoint) -> VariantPoint:
    where = _where(point)
    if not isinstance(point.name, str) or point.name == "":
        origin = point.source or "pareto"
        raise ValueError(f"{origin}: variant name is empty")
    score = _as_finite(point.score, f"{where}: mean score")
    cost = None if point.cost is None else _as_finite(point.cost, f"{where}: cost")
    latency = (
        None if point.latency_ms is None else _as_finite(point.latency_ms, f"{where}: mean latency")
    )
    return VariantPoint(
        name=point.name,
        score=score,
        cost=cost,
        latency_ms=latency,
        source=point.source,
    )


def _where(point: VariantPoint) -> str:
    if point.source:
        return f"{point.source}: variant {point.name!r}"
    return f"variant {point.name!r}"


def _as_finite(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be a finite number")
    # Signed zero would print as -0.000000. Normalise it.
    return number + 0.0


def _reject_duplicates(points: Sequence[VariantPoint]) -> None:
    seen: dict[str, str] = {}
    for point in points:
        if point.name in seen:
            raise ValueError(
                f"duplicate variant name {point.name!r} ({seen[point.name]} and {_origin(point)})"
            )
        seen[point.name] = _origin(point)


def _origin(point: VariantPoint) -> str:
    return point.source if point.source else "<in memory>"


def _choose_axes(points: Sequence[VariantPoint]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    missing_cost = [point.name for point in points if point.cost is None]
    missing_latency = [point.name for point in points if point.latency_ms is None]
    axes = [_AXIS_SCORE]
    omitted: list[str] = []
    if not missing_cost:
        axes.append(_AXIS_COST)
    elif len(missing_cost) != len(points):
        listed = ", ".join(repr(name) for name in missing_cost)
        raise ValueError(f"missing cost on variant(s) {listed}; other variants have a cost")
    else:
        omitted.append(_AXIS_COST)
    if not missing_latency:
        axes.append(_AXIS_LATENCY)
    elif len(missing_latency) != len(points):
        listed = ", ".join(repr(name) for name in missing_latency)
        raise ValueError(f"missing latency on variant(s) {listed}; other variants have a latency")
    else:
        omitted.append(_AXIS_LATENCY)
    if _AXIS_COST not in axes and _AXIS_LATENCY not in axes:
        raise ValueError("missing cost and latency on every variant")
    return tuple(axes), tuple(omitted)


def _frontier_mask(points: Sequence[VariantPoint], axes: Sequence[str]) -> list[bool]:
    on = [True] * len(points)
    for index, point in enumerate(points):
        for other_index, other in enumerate(points):
            if index == other_index:
                continue
            if dominates(other, point, axes):
                on[index] = False
                break
    return on


def _closest_map(
    points: Sequence[VariantPoint],
    axes: Sequence[str],
    on_frontier: Sequence[bool],
) -> dict[str, str]:
    scales = _scales(points, axes)
    closest: dict[str, str] = {}
    for index, point in enumerate(points):
        if on_frontier[index]:
            continue
        closest[point.name] = _closest_name(point, index, points, axes, scales)
    return closest


def _closest_name(
    point: VariantPoint,
    index: int,
    points: Sequence[VariantPoint],
    axes: Sequence[str],
    scales: Mapping[str, tuple[float, float]],
) -> str:
    """Closest dominator. Equal distances break by name, ascending.

    Distances compare with exact ``==``. Identical coordinates produce
    identical IEEE values, so a tie is a tie and the name decides it.
    """
    best_name: str | None = None
    best_distance: float | None = None
    for other_index, other in enumerate(points):
        if other_index == index:
            continue
        if not dominates(other, point, axes):
            continue
        distance = _distance(point, other, axes, scales)
        if best_name is None or best_distance is None:
            best_name = other.name
            best_distance = distance
            continue
        if distance < best_distance or (distance == best_distance and other.name < best_name):
            best_name = other.name
            best_distance = distance
    if best_name is None:
        raise ValueError(f"variant {point.name!r} is dominated but no dominating variant was found")
    return best_name


def _scales(
    points: Sequence[VariantPoint],
    axes: Sequence[str],
) -> dict[str, tuple[float, float]]:
    found: dict[str, tuple[float, float]] = {}
    for axis in axes:
        values = [_oriented(point, axis) for point in points]
        found[axis] = (min(values), max(values))
    return found


def _distance(
    left: VariantPoint,
    right: VariantPoint,
    axes: Sequence[str],
    scales: Mapping[str, tuple[float, float]],
) -> float:
    total = 0.0
    for axis in axes:
        delta = _coord(left, axis, scales) - _coord(right, axis, scales)
        total += delta * delta
    return math.sqrt(total)


def _coord(
    point: VariantPoint,
    axis: str,
    scales: Mapping[str, tuple[float, float]],
) -> float:
    low, high = scales[axis]
    if high == low:
        return 0.0
    return (_oriented(point, axis) - low) / (high - low)


def _oriented(point: VariantPoint, axis: str) -> float:
    value = _axis_value(point, axis)
    if axis == _AXIS_SCORE:
        return value
    return -value


def _axis_value(point: VariantPoint, axis: str) -> float:
    if axis == _AXIS_SCORE:
        return point.score
    if axis == _AXIS_COST:
        if point.cost is None:
            raise ValueError(f"variant {point.name!r}: missing cost")
        return point.cost
    if axis == _AXIS_LATENCY:
        if point.latency_ms is None:
            raise ValueError(f"variant {point.name!r}: missing latency")
        return point.latency_ms
    raise ValueError(f"unknown axis {axis!r}")


def _at_least_as_good(left: float, right: float, *, maximize: bool) -> tuple[bool, bool]:
    if maximize:
        return left >= right, left > right
    return left <= right, left < right


def _order_key(point: VariantPoint, axes: Sequence[str]) -> tuple[object, ...]:
    parts: list[object] = [-point.score]
    for axis in _MINIMISE:
        if axis in axes:
            parts.append(_axis_value(point, axis))
    parts.append(point.name)
    return tuple(parts)


def _points_from_payload(payload: dict[str, Any], path: Path) -> list[VariantPoint]:
    if "baseline" in payload or "candidate" in payload:
        baseline = payload.get("baseline")
        candidate = payload.get("candidate")
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            raise ValueError(
                f"{path}: comparison report needs object fields 'baseline' and 'candidate'"
            )
        return [
            _point_from_side(baseline, path, "baseline"),
            _point_from_side(candidate, path, "candidate"),
        ]
    if "config_name" in payload and "results" in payload:
        return [_point_from_side(payload, path, None)]
    if "rows" in payload and "best" in payload:
        raise ValueError(
            f"{path}: matrix ranking JSON stores mean_score but not per-config cost or "
            "latency. Pass comparison JSON from compare --report-json or run JSON from "
            "run --out."
        )
    if payload.get("command") in {"run", "compare"} and "metrics" in payload:
        raise ValueError(
            f"{path}: this is a history record of headline metrics, not a comparison "
            "or a run report. Pass compare --report-json or run --out files."
        )
    raise ValueError(
        f"{path}: expected a comparison report (baseline and candidate) "
        "or a run report (config_name and results)"
    )


def _point_from_side(raw: dict[str, Any], path: Path, side: str | None) -> VariantPoint:
    label = str(path) if side is None else f"{path} {side}"
    source = str(path) if side is None else f"{path} ({side})"
    run = run_from_dict(raw, label=label)
    return point_from_run(run, source=source)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Report not found: {path}")
    try:
        text = strip_utf8_bom_text(path.read_text(encoding="utf-8-sig"))
    except UnicodeError as exc:
        raise ValueError(f"{path}: not valid UTF-8: {exc}") from exc
    if text.strip() == "":
        raise ValueError(f"{path}: file is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        if _looks_like_history(text):
            raise ValueError(
                f"{path}: this is a history JSONL file, not a comparison or a run "
                "report. Pass compare --report-json or run --out files."
            ) from exc
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: report must be a JSON object")
    return payload


def _looks_like_history(text: str) -> bool:
    saw_row = False
    for line in text.splitlines():
        raw = line.strip().lstrip("\ufeff")
        if raw == "":
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            return False
        if not isinstance(row, dict):
            return False
        if row.get("command") not in {"run", "compare"} or "metrics" not in row:
            return False
        saw_row = True
    return saw_row
