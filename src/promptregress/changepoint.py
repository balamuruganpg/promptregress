"""One change point in the stored run-history mean scores.

``history`` appends one JSON object per ``run`` or ``compare``. This module
reads those records, in file order, and looks for a single shift in
``metrics.mean_score``. ``compare`` rows store the candidate mean, which is
the same number ``trend`` plots.

Series
------

A row with no ``metrics`` object, or a ``metrics`` object with no
``mean_score``, is not a scored run and is skipped. A present
``mean_score`` that is a bool, a non-number, or a non-finite number raises
``ValueError``. The series is not sorted by timestamp.

Statistic
---------

Let ``x_0, ..., x_{n-1}`` be the scored means and let ``k`` run from 1
through ``n - 1``. The before segment is the first ``k`` scores and the
after segment is the rest. Each segment mean is ``math.fsum`` of that
segment divided by its length. The CUSUM statistic at ``k`` is::

    C_k = sqrt(k * (n - k) / n) * abs(before_mean - after_mean)

``change_index`` is the ``k`` that maximises ``C_k``. The smallest ``k``
wins a tie. ``mean_shift`` is ``after_mean - before_mean``. The reported
statistic is ``C`` at ``change_index``. It is not divided by the series
standard deviation: that factor is the same for every permutation of these
values, so it does not change the argmax or the p-value, and it is
undefined when the series is constant.

Permutation test
----------------

The null treats the scored means as exchangeable. ``random.Random(seed)``
shuffles a copy of the series ``permutations`` times (default 2000, default
seed 13). Each shuffle recomputes the maximum ``C_k`` with the same
tie-break. The generator is local to the call. The global ``random``
instance is not seeded and is not drawn from.

The p-value includes the observed series as one extra case::

    p = (1 + #{permutations with maximum C >= observed C}) / (permutations + 1)

Refusals
--------

``ValueError`` (the CLI exits 2) when the history has no records, when no
row has a ``mean_score``, when fewer than 8 scored runs remain, when every
scored mean is equal, or when a present ``mean_score`` is non-finite.

The report is a shift in the recorded means. It is not proof of cause, and
it does not change the gate.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from promptregress.junit_xml import strip_unpaired_surrogates

MIN_RUNS = 8
DEFAULT_SEED = 13
DEFAULT_PERMUTATIONS = 2000

CHANGEPOINT_NOTE = (
    "The series is metrics.mean_score in history-file order. "
    "change_index is the 0-based index of the first run after the split. "
    "It maximises C_k = sqrt(k*(n-k)/n) * abs(before_mean - after_mean) "
    "for k from 1 to n-1. Each segment mean is math.fsum of that segment "
    "divided by its length. The smallest k wins a tie. "
    "mean_shift is after_mean minus before_mean. "
    "The p-value is (1 + the number of permutations whose maximum C is at "
    "least the observed statistic) / (permutations + 1). "
    "Each permutation shuffles this series with random.Random(seed). "
    "That generator is local to the call. "
    "Text rounds to 6 decimals. JSON keeps the stored floats. "
    "It reports a detected shift in the recorded means. It is not proof of "
    "cause, and it does not change the gate."
)

_ID_FIELDS = (
    "timestamp",
    "command",
    "suite",
    "config_name",
    "candidate_name",
    "baseline_name",
    "provider",
    "model",
)


@dataclass(frozen=True)
class RunRef:
    """One scored history row.

    ``index`` is the 0-based position in the scored series. ``record_index``
    is the 0-based position in the loaded history records, which is larger
    when unscored rows were skipped. Identifier fields are copied from the
    record when they are non-empty strings. A missing field is ``None``.
    """

    index: int
    record_index: int
    timestamp: str | None = None
    command: str | None = None
    suite: str | None = None
    config_name: str | None = None
    candidate_name: str | None = None
    baseline_name: str | None = None
    provider: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class ChangePointReport:
    """The single most likely mean shift in one scored series."""

    n: int
    change_index: int
    before_mean: float
    after_mean: float
    mean_shift: float
    statistic: float
    p_value: float
    permutations: int
    seed: int
    skipped: int
    before_runs: tuple[RunRef, ...]
    after_runs: tuple[RunRef, ...]
    note: str = CHANGEPOINT_NOTE

    @property
    def before_n(self) -> int:
        """Number of scored runs in the before segment."""
        return len(self.before_runs)

    @property
    def after_n(self) -> int:
        """Number of scored runs in the after segment."""
        return len(self.after_runs)


def detect_changepoint(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    permutations: int = DEFAULT_PERMUTATIONS,
) -> ChangePointReport:
    """Find one mean shift in history records and attach a permutation p-value.

    ``records`` is the list :func:`promptregress.history.load_history`
    returns. File order is the series order. See the module docstring for
    the statistic, the permutation scheme, and the refusals.
    """
    seed_value = _seed(seed)
    permutation_count = _permutations(permutations)
    if len(records) == 0:
        raise ValueError("empty history")
    scored: list[tuple[float, RunRef]] = []
    skipped = 0
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {record_index}: row is not an object")
        score = _score_of(record, record_index)
        if score is None:
            skipped += 1
            continue
        scored.append((score, _run_ref(record, index=len(scored), record_index=record_index)))
    if not scored:
        raise ValueError("empty history: no mean_score rows")
    scores = [score for score, _ref in scored]
    change_index, statistic, before_mean, after_mean = _analyze(scores)
    p_value = _permutation_p_value(
        scores,
        statistic,
        seed=seed_value,
        permutations=permutation_count,
    )
    refs = tuple(ref for _score, ref in scored)
    return ChangePointReport(
        n=len(scores),
        change_index=change_index,
        before_mean=before_mean,
        after_mean=after_mean,
        mean_shift=after_mean - before_mean,
        statistic=statistic,
        p_value=p_value,
        permutations=permutation_count,
        seed=seed_value,
        skipped=skipped,
        before_runs=refs[:change_index],
        after_runs=refs[change_index:],
    )


def format_changepoint(report: ChangePointReport) -> str:
    """Text report. Ends with a newline.

    Means, the shift, the statistic, and the p-value print with 6 decimals.
    A run line lists only the identifier fields that record carries.
    """
    lines = [
        "changepoint",
        f"n={report.n}",
        f"change_index={report.change_index}",
        f"before_n={report.before_n}",
        f"after_n={report.after_n}",
        f"skipped={report.skipped}",
        f"before_mean={report.before_mean:.6f}",
        f"after_mean={report.after_mean:.6f}",
        f"mean_shift={report.mean_shift:.6f}",
        f"statistic={report.statistic:.6f}",
        f"p_value={report.p_value:.6f}",
        f"permutations={report.permutations}",
        f"seed={report.seed}",
        f"note={report.note}",
        "before:",
        *(_run_line(ref) for ref in report.before_runs),
        "after:",
        *(_run_line(ref) for ref in report.after_runs),
    ]
    return strip_unpaired_surrogates("\n".join(lines) + "\n")


def changepoint_to_dict(report: ChangePointReport) -> dict[str, object]:
    """JSON-ready report. Run objects keep nulls for identifier fields the row lacks."""
    return {
        "n": report.n,
        "change_index": report.change_index,
        "before_n": report.before_n,
        "after_n": report.after_n,
        "skipped": report.skipped,
        "before_mean": report.before_mean,
        "after_mean": report.after_mean,
        "mean_shift": report.mean_shift,
        "statistic": report.statistic,
        "p_value": report.p_value,
        "permutations": report.permutations,
        "seed": report.seed,
        "note": report.note,
        "before_runs": [_run_dict(ref) for ref in report.before_runs],
        "after_runs": [_run_dict(ref) for ref in report.after_runs],
    }


def write_changepoint_json(payload: Mapping[str, object], path: str | Path) -> None:
    """Write ``payload`` as indented JSON. Rejects NaN and Infinity."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    target.write_text(strip_unpaired_surrogates(text), encoding="utf-8")


def _analyze(scores: Sequence[float]) -> tuple[int, float, float, float]:
    """Return ``(change_index, statistic, before_mean, after_mean)``.

    Refuses a series shorter than :data:`MIN_RUNS` and a constant series.
    ``scores`` must already be finite numbers.
    """
    n = len(scores)
    if n < MIN_RUNS:
        raise ValueError(f"changepoint needs at least {MIN_RUNS} scored runs, got {n}")
    first = scores[0]
    if all(score == first for score in scores):
        raise ValueError("constant series: no variance to test")
    change_index, statistic, before_mean, after_mean = _best_split(scores)
    if not math.isfinite(statistic):
        raise ValueError("change-point statistic is non-finite")
    return change_index, statistic, before_mean, after_mean


def _best_split(scores: Sequence[float]) -> tuple[int, float, float, float]:
    """Maximiser of ``C_k``. The smallest ``k`` wins a tie.

    Returns ``(k, C_k, before_mean, after_mean)``.
    """
    n = len(scores)
    best_k = 1
    best_stat = -1.0
    best_before = 0.0
    best_after = 0.0
    for k in range(1, n):
        before_mean = math.fsum(scores[:k]) / k
        after_mean = math.fsum(scores[k:]) / (n - k)
        stat = math.sqrt((k * (n - k)) / n) * abs(before_mean - after_mean)
        if stat > best_stat:
            best_k = k
            best_stat = stat
            best_before = before_mean
            best_after = after_mean
    return best_k, best_stat, best_before, best_after


def _permutation_p_value(
    scores: Sequence[float],
    observed: float,
    *,
    seed: int,
    permutations: int,
) -> float:
    """Monte Carlo p-value of the maximum CUSUM. See the module docstring."""
    rng = random.Random(seed)
    pool = list(scores)
    exceed = 0
    for _ in range(permutations):
        rng.shuffle(pool)
        _k, stat, _before, _after = _best_split(pool)
        if stat >= observed:
            exceed += 1
    return (exceed + 1) / (permutations + 1)


def _score_of(record: Mapping[str, Any], record_index: int) -> float | None:
    """Return the finite mean, ``None`` when the row has no mean, or raise."""
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or "mean_score" not in metrics:
        return None
    raw = metrics["mean_score"]
    where = _where(record, record_index)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{where}: mean_score must be a finite number")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{where}: mean_score is non-finite")
    return value


def _run_ref(record: Mapping[str, Any], *, index: int, record_index: int) -> RunRef:
    config = record.get("config")
    config_map: Mapping[str, Any] = config if isinstance(config, Mapping) else {}
    return RunRef(
        index=index,
        record_index=record_index,
        timestamp=_text(record.get("timestamp")),
        command=_text(record.get("command")),
        suite=_text(record.get("suite")),
        config_name=_text(config_map.get("config_name")),
        candidate_name=_text(config_map.get("candidate_name")),
        baseline_name=_text(config_map.get("baseline_name")),
        provider=_text(config_map.get("provider")),
        model=_text(config_map.get("model")),
    )


def _text(value: object) -> str | None:
    if isinstance(value, str) and value != "":
        return value
    return None


def _where(record: Mapping[str, Any], record_index: int) -> str:
    stamp = _text(record.get("timestamp"))
    if stamp is None:
        return f"record {record_index}"
    return f"record {record_index} ({stamp})"


def _run_line(ref: RunRef) -> str:
    parts = [f"[{ref.index}]", f"record={ref.record_index}"]
    for name in _ID_FIELDS:
        value = getattr(ref, name)
        if isinstance(value, str):
            parts.append(f"{name}={_one_line(value)}")
    return " ".join(parts)


def _run_dict(ref: RunRef) -> dict[str, object]:
    payload: dict[str, object] = {
        "index": ref.index,
        "record_index": ref.record_index,
    }
    for name in _ID_FIELDS:
        value = getattr(ref, name)
        payload[name] = None if value is None else strip_unpaired_surrogates(value)
    return payload


def _one_line(value: str) -> str:
    cleaned = strip_unpaired_surrogates(value)
    return cleaned.replace("\r", " ").replace("\n", " ")


def _seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    return seed


def _permutations(permutations: int) -> int:
    if isinstance(permutations, bool) or not isinstance(permutations, int) or permutations < 1:
        raise ValueError("permutations must be an integer >= 1")
    return permutations
