"""Pass/fail flakiness across stored baselines of one suite.

``promptregress flaky SUITE`` reads ``.promptregress/baselines/*.json`` and
keeps the files whose run ``suite_name`` equals ``SUITE``. The suite
argument is that stored name, not a path. ``--suite`` is the same value.

Runs are ordered by file modification time (nanoseconds), then by baseline
name. The list date is the same UTC second stamp ``baseline list`` prints.
Two files saved in the same nanosecond keep name order.

A case's flip count is the number of pass/fail changes between consecutive
runs. The stored ``passed`` bit is used, not a fresh cut on the score.
``flip_rate`` is ``flip_count / (n_runs - 1)``. Score mean, population
standard deviation (divide by N), min, and max use the scores from the
runs that contain the case.

A case missing from any run of the suite is reported with the number of
runs that contain it. Its flip count and flip rate are omitted. Score
statistics still use the runs where it appears.

Fewer than two stored runs prints why and exits 0, including when
``--fail-on-flaky`` is set. That flag exits 1 when any case that has a
flip rate is at least the given rate. Cases excluded from flip math do
not trip it. The rate must be a finite number in ``[0, 1]``.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from promptregress.baseline import _NAME, baselines_dir, load_baseline
from promptregress.models import RunResult

_DEFINITIONS = {
    "order": "File mtime in nanoseconds, then baseline name.",
    "flip_count": (
        "Pass/fail changes between consecutive runs in that order. The stored passed bit is used."
    ),
    "flip_rate": (
        "flip_count / (n_runs - 1). A case missing from any run of the suite "
        "is excluded, so flip_count and flip_rate are null."
    ),
    "sd": "Population standard deviation of the case scores that are present (divide by N).",
    "fail_on_flaky": (
        "Exit 1 when any included flip rate is >= RATE. "
        "Fewer than 2 stored runs exits 0 and does not apply the flag. "
        "Excluded cases do not trip it."
    ),
}


@dataclass(frozen=True)
class FlakyRun:
    """One stored baseline that belongs to the suite, in analysis order."""

    name: str
    saved_at: str
    mean_score: float


@dataclass(frozen=True)
class FlakyCase:
    """Flips and score spread for one case id across the suite's runs."""

    case_id: str
    n_runs: int
    n_suite_runs: int
    scores: tuple[float, ...]
    passed: tuple[bool, ...]
    flip_count: int | None
    flip_rate: float | None
    mean: float
    sd: float
    minimum: float
    maximum: float
    note: str | None


@dataclass(frozen=True)
class FlakyReport:
    """Flakiness of one suite. ``enough`` is false when fewer than two runs exist."""

    suite_name: str
    runs: tuple[FlakyRun, ...]
    cases: tuple[FlakyCase, ...]
    enough: bool
    message: str


def analyze_baselines(suite_name: str, cwd: str | Path | None = None) -> FlakyReport:
    """Load stored baselines for ``suite_name`` and summarise per-case flips.

    Other suites in the baselines directory are ignored. A legal filename
    whose JSON is not a run report raises :class:`promptregress.baseline.BaselineError`.
    """
    name = suite_name.strip()
    if name == "":
        raise ValueError("flaky requires a non-empty suite name")
    loaded = _load_suite_runs(name, cwd)
    runs = tuple(
        FlakyRun(name=item_name, saved_at=_saved_at(path), mean_score=result.mean_score)
        for item_name, path, result in loaded
    )
    if len(loaded) < 2:
        count = len(loaded)
        noun = "baseline" if count == 1 else "baselines"
        message = (
            f"not enough runs: suite {name!r} has {count} stored {noun} "
            "in .promptregress/baselines (need at least 2)"
        )
        return FlakyReport(
            suite_name=name,
            runs=runs,
            cases=(),
            enough=False,
            message=message,
        )
    return FlakyReport(
        suite_name=name,
        runs=runs,
        cases=tuple(_cases(loaded)),
        enough=True,
        message="",
    )


def format_flaky(report: FlakyReport) -> str:
    """Terminal table, or the not-enough-runs message. Ends with a newline."""
    if not report.enough:
        return report.message + "\n"
    header = ("case", "runs", "flips", "flip_rate", "mean", "sd", "min", "max", "note")
    body = [_case_cells(case) for case in report.cases]
    widths = [len(column) for column in header]
    for row in body:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def fmt(row: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()

    lines = [
        fmt(header),
        *(fmt(row) for row in body),
        (
            "sd is the population standard deviation (divide by N). "
            "flip_rate is pass/fail flips between consecutive runs divided by (runs - 1). "
            "Runs are ordered by file mtime, then name. "
            "A case missing from any run is excluded from flip math."
        ),
    ]
    return "\n".join(lines) + "\n"


def flaky_to_dict(report: FlakyReport) -> dict[str, object]:
    """JSON-ready report. Flip fields are ``None`` when the case was excluded."""
    return {
        "suite_name": report.suite_name,
        "n_runs": len(report.runs),
        "enough": report.enough,
        "message": report.message,
        "definitions": dict(_DEFINITIONS),
        "runs": [
            {"name": run.name, "saved_at": run.saved_at, "mean_score": run.mean_score}
            for run in report.runs
        ],
        "cases": [_case_to_dict(case) for case in report.cases],
    }


def failing_cases(report: FlakyReport, rate: float) -> tuple[FlakyCase, ...]:
    """Cases whose flip rate is at least ``rate``. Excluded cases are omitted.

    An empty result when the report does not have two runs: the flag is not
    applied. ``rate`` must already be finite and in ``[0, 1]``.
    """
    if not report.enough:
        return ()
    return tuple(
        case for case in report.cases if case.flip_rate is not None and case.flip_rate >= rate
    )


def _load_suite_runs(
    suite_name: str,
    cwd: str | Path | None,
) -> list[tuple[str, Path, RunResult]]:
    directory = baselines_dir(cwd)
    if not directory.is_dir():
        return []
    found: list[tuple[str, Path, RunResult]] = []
    for path in directory.glob("*.json"):
        if not path.is_file():
            continue
        name = path.stem
        if _NAME.fullmatch(name) is None:
            continue
        result = load_baseline(name, cwd)
        if result.suite_name != suite_name:
            continue
        found.append((name, path, result))
    found.sort(key=lambda item: (item[1].stat().st_mtime_ns, item[0]))
    return found


def _cases(loaded: Sequence[tuple[str, Path, RunResult]]) -> list[FlakyCase]:
    n_suite = len(loaded)
    series: dict[str, list[tuple[bool, float]]] = {}
    order: list[str] = []
    for _name, _path, result in loaded:
        # Last duplicate id in a run wins, matching RunResult.by_id.
        by_id = result.by_id()
        recorded: set[str] = set()
        for row in result.results:
            if row.case_id in recorded:
                continue
            recorded.add(row.case_id)
            chosen = by_id[row.case_id]
            if row.case_id not in series:
                series[row.case_id] = []
                order.append(row.case_id)
            series[row.case_id].append((chosen.passed, chosen.score))
    cases = [_one_case(case_id, series[case_id], n_suite) for case_id in order]
    cases.sort(key=_sort_key)
    return cases


def _one_case(
    case_id: str,
    points: Sequence[tuple[bool, float]],
    n_suite: int,
) -> FlakyCase:
    passed = tuple(item[0] for item in points)
    scores = tuple(item[1] for item in points)
    present = len(scores)
    mean = sum(scores) / present
    if present == n_suite:
        flips = _flip_count(passed)
        rate: float | None = flips / (n_suite - 1)
        flip_count: int | None = flips
        note = None
    else:
        rate = None
        flip_count = None
        note = f"present in {present} of {n_suite} runs; excluded from flip math"
    return FlakyCase(
        case_id=case_id,
        n_runs=present,
        n_suite_runs=n_suite,
        scores=scores,
        passed=passed,
        flip_count=flip_count,
        flip_rate=rate,
        mean=mean,
        sd=_population_sd(scores),
        minimum=min(scores),
        maximum=max(scores),
        note=note,
    )


def _flip_count(passed: Sequence[bool]) -> int:
    # The second sequence is one shorter on purpose: each step is a consecutive pair.
    return sum(1 for left, right in zip(passed, passed[1:]) if left != right)


def _population_sd(values: Sequence[float]) -> float:
    """Population standard deviation. Divide by N. One value is 0."""
    count = len(values)
    if count == 0:
        return 0.0
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    if variance < 0.0:
        variance = 0.0
    return math.sqrt(variance)


def _sort_key(case: FlakyCase) -> tuple[bool, float, float, str]:
    """Flip rate descending, then sd descending, then case id.

    A case with no flip rate sorts after every case that has one.
    """
    if case.flip_rate is None:
        return (True, 0.0, -case.sd, case.case_id)
    return (False, -case.flip_rate, -case.sd, case.case_id)


def _case_cells(case: FlakyCase) -> tuple[str, ...]:
    if case.flip_count is None or case.flip_rate is None:
        flips = "-"
        rate = "-"
    else:
        flips = str(case.flip_count)
        rate = f"{case.flip_rate:.6f}"
    return (
        case.case_id,
        str(case.n_runs),
        flips,
        rate,
        f"{case.mean:.6f}",
        f"{case.sd:.6f}",
        f"{case.minimum:.6f}",
        f"{case.maximum:.6f}",
        "" if case.note is None else case.note,
    )


def _case_to_dict(case: FlakyCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "n_runs": case.n_runs,
        "n_suite_runs": case.n_suite_runs,
        "scores": list(case.scores),
        "passed": list(case.passed),
        "flip_count": case.flip_count,
        "flip_rate": case.flip_rate,
        "mean": case.mean,
        "sd": case.sd,
        "min": case.minimum,
        "max": case.maximum,
        "note": case.note,
    }


def _saved_at(path: Path) -> str:
    stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
