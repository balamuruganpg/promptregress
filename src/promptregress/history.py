"""Append-only run history and a mean-score sparkline.

``run`` and ``compare`` append one JSON object per line to
``.promptregress/history.jsonl`` in the working directory. The line records
the time, the suite, a short config summary, and headline metrics. A write
error is the caller's to report. The CLI warns and still returns the run's
own exit code.

``trend`` plots ``metrics.mean_score`` for one suite. The HTML page is a
single file with an inline SVG polyline and no external assets. The ASCII
sparkline maps a score in ``[0, 1]`` onto eight blocks, low to high. Scores
outside that range are clamped. The scale is absolute, so a flat 1.0 is a
row of full blocks rather than a rescaled wiggle.

Stored metrics are typed. ``mean_score``, ``mean_diff``, and
``pairwise_win_rate`` print only when they are an int or a float and not a
bool. ``gate_passed`` prints only when it is a JSON bool (``true`` is
``gate=pass``, ``false`` is ``gate=fail``). A string, null, bool, or other
non-number is not coerced, and the digits ``0`` and ``1`` are not a gate.
The first bad field, in the order mean score, mean diff, pairwise win
rate, then gate, is a ``load_history`` note
``path:line: metrics.<field> must be a finite number`` or
``metrics.gate_passed must be a boolean``. The row stays in the loaded
records. The table omits it only when the summary would be empty. An
integer that does not fit in a float is a hard error with that same
``path:line`` wording, not a skipped line. Infinity stays a value: history
prints it, trend skips it, and changepoint reports it as non-finite.
"""

import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeGuard

from promptregress.gate import _fmt_vs_zero
from promptregress.junit_xml import strip_unpaired_surrogates
from promptregress.loading import strip_utf8_bom_text
from promptregress.models import Comparison, RunResult

_SPARK = "▁▂▃▄▅▆▇█"


def utc_now() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_history_path(cwd: str | Path | None = None) -> Path:
    """``.promptregress/history.jsonl`` under ``cwd`` or the process cwd."""
    root = Path.cwd() if cwd is None else Path(cwd)
    return root / ".promptregress" / "history.jsonl"


def make_run_record(
    result: RunResult,
    *,
    provider: str,
    model: str,
    template: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Headline record for one ``run``."""
    return {
        "timestamp": utc_now() if timestamp is None else timestamp,
        "command": "run",
        "suite": result.suite_name,
        "config": {
            "config_name": result.config_name,
            "provider": provider,
            "model": model,
            "template": template,
        },
        "metrics": {
            "mean_score": result.mean_score,
            "pass_rate": result.pass_rate,
            "mean_latency_ms": result.mean_latency_ms,
            "p95_latency_ms": result.p95_latency_ms,
            "total_tokens": result.total_tokens,
            "n_cases": result.n_cases,
        },
    }


def make_compare_record(
    comparison: Comparison,
    *,
    provider: str,
    model: str,
    baseline_template: str,
    candidate_template: str,
    gate_passed: bool,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Headline record for one ``compare``.

    ``mean_score`` is the candidate mean, which is what ``trend`` plots.
    ``pairwise_win_rate`` is omitted when the compare did not run a judge.
    """
    metrics: dict[str, Any] = {
        "mean_score": comparison.candidate.mean_score,
        "baseline_mean_score": comparison.baseline.mean_score,
        "pass_rate": comparison.candidate.pass_rate,
        "mean_diff": comparison.mean_diff,
        "ci_low": comparison.ci_low,
        "ci_high": comparison.ci_high,
        "wins": comparison.wins,
        "losses": comparison.losses,
        "ties": comparison.ties,
        "gate_passed": gate_passed,
        "mean_latency_ms": comparison.candidate.mean_latency_ms,
        "p95_latency_ms": comparison.candidate.p95_latency_ms,
        "total_tokens": comparison.candidate.total_tokens,
        "n_cases": comparison.candidate.n_cases,
    }
    if comparison.pairwise is not None:
        metrics["pairwise_win_rate"] = comparison.pairwise.win_rate
    return {
        "timestamp": utc_now() if timestamp is None else timestamp,
        "command": "compare",
        "suite": comparison.baseline.suite_name,
        "config": {
            "provider": provider,
            "model": model,
            "baseline_name": comparison.baseline.config_name,
            "candidate_name": comparison.candidate.config_name,
            "baseline_template": baseline_template,
            "candidate_template": candidate_template,
        },
        "metrics": metrics,
    }


def append_history(record: Mapping[str, Any], path: str | Path | None = None) -> Path:
    """Append one JSON object as a single line. Creates the parent directory.

    Raises ``OSError`` when the directory or the file cannot be written.
    The caller decides whether that is fatal. This function does not catch
    it, so a library user can still fail closed.
    """
    target = default_history_path() if path is None else Path(path)
    payload = dict(record)
    line = json.dumps(payload, ensure_ascii=False)
    if "\n" in line or "\r" in line:
        raise ValueError("history record must encode as one line")
    target.parent.mkdir(parents=True, exist_ok=True)
    # A previous append may have been cut short mid-line (a full disk, a
    # killed process). Terminate that partial line first, so this record
    # starts on its own line instead of gluing onto the fragment — a glued
    # line is invalid JSON and the reader would drop both rows.
    needs_separator = False
    if target.exists() and target.stat().st_size > 0:
        with target.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_separator = handle.read(1) != b"\n"
    with target.open("a", encoding="utf-8") as handle:
        if needs_separator:
            handle.write("\n")
        handle.write(line + "\n")
    return target


def load_history(path: str | Path | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Read history rows. A missing file is an empty history, not an error.

    Malformed lines are skipped. Their descriptions are returned as notes.
    ``OSError`` propagates. Invalid UTF-8 raises ``ValueError`` with a
    ``not valid UTF-8`` message (same shape as calibration / run reports).

    A present metric of the wrong type is kept and noted (see the module
    docstring). The first bad field wins. An integer metric that does not
    fit in a float raises ``ValueError`` ``path:line: metrics.<field> must
    be a finite number`` instead of skipping the row.
    """
    target = default_history_path() if path is None else Path(path)
    if not target.exists():
        return [], []
    # utf-8-sig (+ residual / per-line BOM strip): a leading BOM on the
    # first JSONL line would otherwise be skipped as "invalid JSON" and
    # drop that row. Per-line strip keeps rows from concatenated chunks.
    try:
        text = strip_utf8_bom_text(target.read_text(encoding="utf-8-sig"))
    except UnicodeError as exc:
        raise ValueError(f"{target}: not valid UTF-8: {exc}") from exc
    records: list[dict[str, Any]] = []
    notes: list[str] = []
    # Split on "\n" only: str.splitlines() also breaks on U+2028, U+2029,
    # and U+0085, which json.dumps(ensure_ascii=False) can emit raw inside
    # a string value — such a row would be cut in two and dropped as
    # invalid JSON. A trailing "\r" (CRLF file) is stripped per line.
    for line_number, raw in enumerate(text.split("\n"), start=1):
        if raw.endswith("\r"):
            raw = raw[:-1]
        raw = raw.lstrip("\ufeff")
        if raw.strip() == "":
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            notes.append(f"{target}:{line_number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(payload, dict):
            notes.append(f"{target}:{line_number}: row is not an object")
            continue
        _reject_metric_overflow(payload, target, line_number)
        problem = _metrics_type_problem(payload)
        if problem is not None:
            notes.append(f"{target}:{line_number}: {problem}")
        records.append(payload)
    return records, notes


def format_history_table(records: Sequence[Mapping[str, Any]]) -> str:
    """Aligned text table. Ends with a newline. Empty history says so."""
    header = ("timestamp", "command", "suite", "summary")
    visible = [record for record in records if _show_history_row(record)]
    body = [
        (
            _cell(record.get("timestamp", "")),
            _cell(record.get("command", "")),
            _cell(record.get("suite", "")),
            _summary(record),
        )
        for record in visible
    ]
    if not body:
        return "timestamp  command  suite  summary\n(no runs)\n"
    widths = [len(column) for column in header]
    for row in body:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def fmt(row: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()

    lines = [fmt(header), *(fmt(row) for row in body)]
    return "\n".join(lines) + "\n"


def suite_points(
    records: Sequence[Mapping[str, Any]],
    suite: str,
) -> list[tuple[str, float]]:
    """``(timestamp, mean_score)`` rows for ``suite``, in file order.

    Rows without a finite ``metrics.mean_score`` are skipped. An integer
    that does not fit in a float raises ``ValueError``.
    """
    points: list[tuple[str, float]] = []
    for record in records:
        if record.get("suite") != suite:
            continue
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            continue
        score = metrics.get("mean_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        try:
            value = float(score)
        except OverflowError as exc:
            stamp = record.get("timestamp")
            where = f"record ({stamp})" if isinstance(stamp, str) and stamp != "" else "record"
            raise ValueError(f"{where}: mean_score must be a finite number") from exc
        if not math.isfinite(value):
            continue
        stamp = record.get("timestamp")
        label = stamp if isinstance(stamp, str) else ""
        points.append((label, value))
    return points


def ascii_sparkline(values: Sequence[float]) -> str:
    """Map each score onto ``▁▂▃▄▅▆▇█`` using a fixed ``[0, 1]`` scale.

    ``0`` is the lowest block and ``1`` is the highest. Non-finite values
    become ``·``. An empty sequence is an empty string.
    """
    chars: list[str] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            chars.append("·")
            continue
        number = float(value)
        if not math.isfinite(number):
            chars.append("·")
            continue
        clamped = min(1.0, max(0.0, number))
        index = 7 if clamped >= 1.0 else int(clamped * 8.0)
        chars.append(_SPARK[index])
    return "".join(chars)


def format_trend(suite: str, values: Sequence[float]) -> str:
    """Terminal trend: the suite, the count, and the ASCII sparkline."""
    lines = [
        f"suite={suite}",
        f"n={len(values)}",
        ascii_sparkline(values),
    ]
    return "\n".join(lines) + "\n"


def render_trend_html(suite: str, points: Sequence[tuple[str, float]]) -> str:
    """Self-contained HTML page with an inline SVG sparkline.

    The document does not load scripts, stylesheets, images, or fonts.
    The vertical scale is fixed at 0 to 1.
    """
    values = [score for _, score in points]
    title = _escape(suite)
    rows = []
    for stamp, score in points:
        rows.append(f'          <tr><td>{_escape(stamp)}</td><td class="num">{score:.6f}</td></tr>')
    body = "\n".join(rows)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>PromptRegress trend — {title}</title>\n"
        "  <style>\n"
        "    :root { color-scheme: light; --ink: #1c1915; --muted: #5c564c;\n"
        "      --line: #e4ddd2; --paper: #faf7f2; --card: #fffdf9; --mark: #1f6b45; }\n"
        "    * { box-sizing: border-box; }\n"
        "    body { margin: 0; padding: 2rem 1.25rem 3rem;\n"
        '      font: 16px/1.5 "Segoe UI", Helvetica, Arial, sans-serif;\n'
        "      color: var(--ink); background: var(--paper); }\n"
        "    main { max-width: 720px; margin: 0 auto; }\n"
        "    h1 { font-size: 1.75rem; line-height: 1.2; margin: 0 0 0.25rem; }\n"
        "    .lede { color: var(--muted); margin: 0 0 1rem; }\n"
        "    section { background: var(--card); border: 1px solid var(--line);\n"
        "      border-radius: 10px; padding: 1rem 1.1rem 1.15rem; margin: 1rem 0; }\n"
        "    table { width: 100%; border-collapse: collapse; }\n"
        "    th, td { text-align: left; padding: 0.4rem 0.5rem;\n"
        "      border-bottom: 1px solid var(--line); }\n"
        "    th { font-size: 0.75rem; letter-spacing: 0.03em; text-transform: uppercase;\n"
        "      color: var(--muted); }\n"
        "    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }\n"
        "    svg { width: 100%; height: auto; display: block; }\n"
        "    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <main>\n"
        "    <header>\n"
        "      <h1>PromptRegress trend</h1>\n"
        f'      <p class="lede">Mean score for <code>{title}</code>. '
        "The scale is 0 to 1.</p>\n"
        "    </header>\n"
        "    <section>\n"
        "      <h2>Mean score</h2>\n"
        f"{_sparkline_svg(values)}\n"
        f"      <p>ASCII: <code>{_escape(ascii_sparkline(values))}</code></p>\n"
        "    </section>\n"
        "    <section>\n"
        "      <h2>History</h2>\n"
        "      <table>\n"
        "        <thead>\n"
        "          <tr>\n"
        '            <th scope="col">Timestamp</th>\n'
        '            <th scope="col" class="num">Mean score</th>\n'
        "          </tr>\n"
        "        </thead>\n"
        "        <tbody>\n"
        f"{body}\n"
        "        </tbody>\n"
        "      </table>\n"
        "    </section>\n"
        "  </main>\n"
        "</body>\n"
        "</html>\n"
    )


def write_trend_html(
    suite: str,
    points: Sequence[tuple[str, float]],
    path: str | Path,
) -> None:
    """Write :func:`render_trend_html` to ``path``, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        strip_unpaired_surrogates(render_trend_html(suite, points)),
        encoding="utf-8",
    )


def _sparkline_svg(values: Sequence[float]) -> str:
    width = 320.0
    height = 64.0
    pad = 4.0
    label = _escape(ascii_sparkline(values))

    def y_of(value: float) -> float:
        clamped = min(1.0, max(0.0, value))
        return pad + (height - 2.0 * pad) * (1.0 - clamped)

    guides = (
        f'<line x1="{pad:.1f}" y1="{y_of(1):.2f}" x2="{width - pad:.1f}" '
        f'y2="{y_of(1):.2f}" stroke="#e4ddd2" stroke-width="1"/>'
        f'<line x1="{pad:.1f}" y1="{y_of(0):.2f}" x2="{width - pad:.1f}" '
        f'y2="{y_of(0):.2f}" stroke="#e4ddd2" stroke-width="1"/>'
    )
    if not values:
        shape = ""
    elif len(values) == 1:
        shape = f'<circle cx="{width / 2:.2f}" cy="{y_of(values[0]):.2f}" r="3.5" fill="#1f6b45"/>'
    else:
        coords: list[str] = []
        last = len(values) - 1
        for index, value in enumerate(values):
            x_pos = pad + (width - 2.0 * pad) * index / last
            coords.append(f"{x_pos:.2f},{y_of(value):.2f}")
        shape = (
            f'<polyline fill="none" stroke="#1f6b45" stroke-width="2" points="{" ".join(coords)}"/>'
        )
    return (
        f'      <svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="mean score {label}">'
        f"{guides}{shape}</svg>"
    )


_NUMBER_FIELDS = ("mean_score", "mean_diff", "pairwise_win_rate")


def _is_json_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _reject_metric_overflow(record: Mapping[str, Any], target: Path, line_number: int) -> None:
    """Raise when a numeric metric does not fit in a float."""
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return
    for field in _NUMBER_FIELDS:
        if field not in metrics:
            continue
        value = metrics[field]
        if not _is_json_number(value):
            continue
        try:
            float(value)
        except OverflowError as exc:
            raise ValueError(
                f"{target}:{line_number}: metrics.{field} must be a finite number"
            ) from exc


def _metrics_type_problem(record: Mapping[str, Any]) -> str | None:
    """First mistyped metric, or ``None`` when every present field is typed."""
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None
    for field in _NUMBER_FIELDS:
        if field not in metrics:
            continue
        if not _is_json_number(metrics[field]):
            return f"metrics.{field} must be a finite number"
    if "gate_passed" in metrics and not isinstance(metrics["gate_passed"], bool):
        return "metrics.gate_passed must be a boolean"
    return None


def _render_metric(value: object, field: str) -> str | None:
    """``field=digits`` for a real number. A wrong type renders nothing."""
    if not _is_json_number(value):
        return None
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"metrics.{field} must be a finite number") from exc
    return f"{field}={_fmt_vs_zero(number)}"


def _summary(record: Mapping[str, Any]) -> str:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return ""
    parts: list[str] = []
    for field in ("mean_score", "mean_diff"):
        if field not in metrics:
            continue
        rendered = _render_metric(metrics[field], field)
        if rendered is not None:
            parts.append(rendered)
    if isinstance(metrics.get("gate_passed"), bool):
        parts.append("gate=pass" if metrics["gate_passed"] else "gate=fail")
    if "pairwise_win_rate" in metrics:
        rendered = _render_metric(metrics["pairwise_win_rate"], "pairwise_win_rate")
        if rendered is not None:
            parts.append(rendered)
    return " ".join(parts)


def _show_history_row(record: Mapping[str, Any]) -> bool:
    """Hide a row only when its summary is empty because a metric is mistyped."""
    return bool(_summary(record)) or _metrics_type_problem(record) is None


def _cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ")


def _escape(text: str) -> str:
    cleaned = strip_unpaired_surrogates(text)
    return (
        cleaned.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
