"""Named baseline runs under ``.promptregress/baselines``.

``promptregress run --save-baseline NAME`` writes one file,
``NAME.json``. The file is the run report from :func:`promptregress.report.run_to_dict`
(the same JSON ``promptregress run --out`` writes). Each case object includes
``output``, so a later compare can score drift without calling the baseline
provider. Derived headlines (``mean_score``, latency percentiles, token totals)
are in the file too; loading recomputes the properties from ``results``.

The list date is the file's modification time in UTC, not a field in the
run JSON. That keeps the stored document the same shape as a run report.
"""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from promptregress.irt import load_run_report
from promptregress.models import RunResult, Suite
from promptregress.report import write_run_json

_NAME = re.compile(r"[A-Za-z0-9._-]+\Z")


class BaselineError(ValueError):
    """A baseline name, file, or suite identity check failed."""


@dataclass(frozen=True)
class BaselineInfo:
    """One stored baseline, for ``promptregress baseline list``."""

    name: str
    saved_at: str
    suite_name: str
    mean_score: float


def validate_baseline_name(name: str) -> str:
    """Return ``name`` when it is a single path segment of ``[A-Za-z0-9._-]+``.

    Slashes and backslashes are rejected, so the name cannot escape the
    baselines directory. ``.`` and ``..`` are not special here: the stored
    file is always ``<name>.json``.
    """
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise BaselineError(
            "baseline name must match [A-Za-z0-9._-]+ and must not contain a path separator"
        )
    return name


def baselines_dir(cwd: str | Path | None = None) -> Path:
    """``.promptregress/baselines`` under ``cwd`` or the process cwd."""
    root = Path.cwd() if cwd is None else Path(cwd)
    return root / ".promptregress" / "baselines"


def baseline_path(name: str, cwd: str | Path | None = None) -> Path:
    """Path of the JSON file for ``name``. Raises when ``name`` is illegal."""
    validate_baseline_name(name)
    return baselines_dir(cwd) / f"{name}.json"


def save_baseline(
    result: RunResult,
    name: str,
    cwd: str | Path | None = None,
) -> tuple[Path, bool]:
    """Write ``result`` as run-report JSON. Returns ``(path, overwritten)``.

    Creates the baselines directory. ``overwritten`` is true when ``path``
    was already a file. The caller prints the warning.
    """
    path = baseline_path(name, cwd)
    overwritten = path.is_file()
    write_run_json(result, path)
    return path, overwritten


def load_baseline(name: str, cwd: str | Path | None = None) -> RunResult:
    """Load a stored run. A missing or unreadable file is a :class:`BaselineError`."""
    path = baseline_path(name, cwd)
    if not path.is_file():
        raise BaselineError(f"baseline {name!r} not found at {path}")
    try:
        return load_run_report(path)
    except (OSError, ValueError) as exc:
        raise BaselineError(str(exc)) from exc


def list_baselines(cwd: str | Path | None = None) -> list[BaselineInfo]:
    """Stored baselines sorted by name. A missing directory is an empty list.

    Files whose stem is not a legal baseline name are skipped. A legal name
    whose JSON is not a run report raises :class:`BaselineError`.
    """
    directory = baselines_dir(cwd)
    if not directory.is_dir():
        return []
    found: list[BaselineInfo] = []
    for path in sorted(directory.glob("*.json")):
        if not path.is_file():
            continue
        name = path.stem
        if _NAME.fullmatch(name) is None:
            continue
        result = load_baseline(name, cwd)
        found.append(
            BaselineInfo(
                name=name,
                saved_at=_saved_at(path),
                suite_name=result.suite_name,
                mean_score=result.mean_score,
            )
        )
    return found


def delete_baseline(name: str, cwd: str | Path | None = None) -> Path:
    """Remove one stored baseline. A missing file is a :class:`BaselineError`."""
    path = baseline_path(name, cwd)
    if not path.is_file():
        raise BaselineError(f"baseline {name!r} not found at {path}")
    path.unlink()
    return path


def ensure_baseline_matches(stored: RunResult, suite: Suite, name: str) -> None:
    """Require the stored suite name and case-id multiset to match ``suite``.

    Order may differ. ``compare`` pairs by id. The error names the suite when
    the names differ, and the first differing case id in current-suite order
    (then stored order) when the id multisets differ.
    """
    problems: list[str] = []
    if stored.suite_name != suite.name:
        problems.append(f"suite name {stored.suite_name!r} does not match {suite.name!r}")
    differing = first_differing_case_id(
        [item.case_id for item in stored.results],
        [case.id for case in suite.cases],
    )
    if differing is not None:
        problems.append(f"first differing case id: {differing}")
    if problems:
        detail = "; ".join(problems)
        raise BaselineError(f"baseline {name!r} does not match the current suite: {detail}")


def first_differing_case_id(stored_ids: Sequence[str], current_ids: Sequence[str]) -> str | None:
    """First case id whose count differs, walking ``current_ids`` then ``stored_ids``.

    ``None`` means the multisets are equal. Duplicate ids count.
    """
    stored_counts = Counter(stored_ids)
    current_counts = Counter(current_ids)
    if stored_counts == current_counts:
        return None
    for case_id in current_ids:
        if stored_counts[case_id] != current_counts[case_id]:
            return case_id
    for case_id in stored_ids:
        if stored_counts[case_id] != current_counts[case_id]:
            return case_id
    return None


def format_baseline_table(rows: Sequence[BaselineInfo]) -> str:
    """Aligned text table: name, date, suite, mean. Ends with a newline."""
    header = ("name", "date", "suite", "mean")
    body = [(row.name, row.saved_at, row.suite_name, f"{row.mean_score:.6f}") for row in rows]
    if not body:
        return "name  date  suite  mean\n(no baselines)\n"
    widths = [len(column) for column in header]
    for row in body:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def fmt(row: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()

    lines = [fmt(header), *(fmt(row) for row in body)]
    return "\n".join(lines) + "\n"


def _saved_at(path: Path) -> str:
    stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
