"""JSON, Markdown, and HTML writers for a paired comparison."""

import html
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from promptregress.calibration import JUDGE_AGREEMENT_DEFINITION
from promptregress.cluster import FailureCluster
from promptregress.drift import DRIFT_DEFINITION
from promptregress.gate import GateVerdict, evaluate_gate
from promptregress.models import (
    CaseResult,
    Comparison,
    CostEstimate,
    DriftSummary,
    JudgeAgreement,
    PairedRow,
    PairwiseSummary,
    RunResult,
    SliceSummary,
)
from promptregress.pricing import cost_delta


def write_json(
    comparison: Comparison,
    path: str | Path,
    verdict: GateVerdict | None = None,
    *,
    failure_clusters: Sequence[FailureCluster] | None = None,
    embedder: str | None = None,
) -> None:
    """Write the comparison and gate verdict as indented JSON.

    When ``verdict`` is omitted, the default gate is evaluated
    (``max_regression=0``, pass-rate floor 0, significance not required).
    ``failure_clusters`` is included only when the caller passes a list
    (including an empty list). ``None`` leaves the report unchanged.
    """
    resolved = verdict if verdict is not None else evaluate_gate(comparison)
    payload = comparison_to_dict(
        comparison,
        resolved,
        failure_clusters=failure_clusters,
        embedder=embedder,
    )
    _write_text(path, json.dumps(payload, indent=2) + "\n")


def write_markdown(
    comparison: Comparison,
    path: str | Path,
    verdict: GateVerdict | None = None,
    *,
    failure_clusters: Sequence[FailureCluster] | None = None,
    embedder: str | None = None,
) -> None:
    """Write a Markdown summary, per-case win/loss table, and gate verdict.

    When ``verdict`` is omitted, the default gate is evaluated.
    """
    resolved = verdict if verdict is not None else evaluate_gate(comparison)
    _write_text(
        path,
        render_markdown(
            comparison,
            resolved,
            failure_clusters=failure_clusters,
            embedder=embedder,
        ),
    )


def write_html(
    comparison: Comparison,
    path: str | Path,
    verdict: GateVerdict | None = None,
    *,
    failure_clusters: Sequence[FailureCluster] | None = None,
    embedder: str | None = None,
) -> None:
    """Write one self-contained HTML report.

    The file inlines its CSS and does not reference scripts, stylesheets,
    images, or fonts from the network. When ``verdict`` is omitted, the
    default gate is evaluated (``max_regression=0``, pass-rate floor 0,
    significance not required).
    """
    resolved = verdict if verdict is not None else evaluate_gate(comparison)
    _write_text(
        path,
        render_html(
            comparison,
            resolved,
            failure_clusters=failure_clusters,
            embedder=embedder,
        ),
    )


def write_run_json(result: RunResult, path: str | Path) -> None:
    """Write one run (not a comparison) as indented JSON."""
    _write_text(path, json.dumps(run_to_dict(result), indent=2) + "\n")


def format_text(
    comparison: Comparison,
    verdict: GateVerdict | None = None,
    *,
    failure_clusters: Sequence[FailureCluster] | None = None,
    embedder: str | None = None,
) -> str:
    """Plain-text summary used by the CLI. Ends with a newline."""
    resolved = verdict if verdict is not None else evaluate_gate(comparison)
    level = int(round((1.0 - comparison.alpha) * 100))
    lines = [
        f"suite={comparison.baseline.suite_name}",
        f"baseline={comparison.baseline.config_name}",
        f"candidate={comparison.candidate.config_name}",
        f"baseline_mean={comparison.baseline.mean_score:.6f}",
        f"candidate_mean={comparison.candidate.mean_score:.6f}",
        f"baseline_pass_rate={comparison.baseline.pass_rate:.6f}",
        f"candidate_pass_rate={comparison.candidate.pass_rate:.6f}",
        *_text_usage(comparison.baseline, "baseline"),
        *_text_usage(comparison.candidate, "candidate"),
        *_text_cost(comparison),
        f"mean_diff={comparison.mean_diff:.6f}",
        f"ci_low={comparison.ci_low:.6f}",
        f"ci_high={comparison.ci_high:.6f}",
        f"ci_level={level}",
        f"ci_method={comparison.ci_method}",
        f"mcnemar_p={comparison.mcnemar_p:.6f}",
        f"mcnemar_b={comparison.mcnemar_b}",
        f"mcnemar_c={comparison.mcnemar_c}",
        f"wilcoxon_p={_num(comparison.wilcoxon_p)}",
        f"ttest_p={_num(comparison.ttest_p)}",
        f"cohens_dz={_fmt_stat(comparison.cohens_dz)}",
        f"effect_size={comparison.effect_size_label}",
        f"wins={comparison.wins}",
        f"losses={comparison.losses}",
        f"ties={comparison.ties}",
        *_text_pairwise(comparison),
        *_text_judge_agreement(comparison),
        *_text_drift(comparison),
        *_text_stability(comparison),
        f"gate_passed={'true' if resolved.passed else 'false'}",
        "reasons:",
    ]
    lines.extend(f"- {reason}" for reason in resolved.reasons)
    if comparison.details:
        lines.append("details:")
        lines.extend(f"- {note}" for note in comparison.details)
    if comparison.slices:
        lines.append("slices:")
        for item in comparison.slices:
            lines.append(
                f"- {item.tag} n={item.n} mean_diff={item.mean_diff:.6f} "
                f"ci_low={item.ci_low:.6f} ci_high={item.ci_high:.6f} "
                f"wins={item.wins} losses={item.losses} ties={item.ties}"
            )
    if failure_clusters is not None:
        lines.extend(_text_failure_modes(failure_clusters, embedder))
    lines.append("per_case:")
    for row in comparison.per_case:
        lines.append(
            f"- {row.case_id} baseline={row.baseline_score:.6f} "
            f"candidate={row.candidate_score:.6f} diff={row.diff:.6f} "
            f"baseline_passed={'true' if row.baseline_passed else 'false'} "
            f"candidate_passed={'true' if row.candidate_passed else 'false'} "
            f"{row.outcome}"
        )
    return "\n".join(lines) + "\n"


def render_markdown(
    comparison: Comparison,
    verdict: GateVerdict,
    *,
    failure_clusters: Sequence[FailureCluster] | None = None,
    embedder: str | None = None,
) -> str:
    """Return the Markdown report, including a trailing newline."""
    level = int(round((1.0 - comparison.alpha) * 100))
    lines = [
        "# PromptRegress comparison",
        "",
        f"Suite: `{comparison.baseline.suite_name}`",
        "",
        "| Config | Mean score | Pass rate | Total latency (ms) |",
        "| --- | ---: | ---: | ---: |",
        _run_table_row(comparison.baseline),
        _run_table_row(comparison.candidate),
        "",
        *_markdown_usage(comparison),
        *_markdown_cost(comparison),
        "## Paired statistics",
        "",
        "| Statistic | Value |",
        "| --- | ---: |",
        f"| Mean diff (candidate - baseline) | {comparison.mean_diff:.6f} |",
        f"| {level}% CI low | {comparison.ci_low:.6f} |",
        f"| {level}% CI high | {comparison.ci_high:.6f} |",
        f"| CI method | {comparison.ci_method} |",
        f"| McNemar p-value | {comparison.mcnemar_p:.6f} |",
        f"| McNemar b (baseline pass, candidate fail) | {comparison.mcnemar_b} |",
        f"| McNemar c (baseline fail, candidate pass) | {comparison.mcnemar_c} |",
        f"| Wilcoxon W | {_fmt_stat(comparison.wilcoxon_stat)} |",
        f"| Wilcoxon p-value | {_fmt_stat(comparison.wilcoxon_p)} |",
        f"| Paired t | {_fmt_stat(comparison.ttest_stat)} |",
        f"| t degrees of freedom | {_fmt_stat(comparison.ttest_df)} |",
        f"| Paired t p-value | {_fmt_stat(comparison.ttest_p)} |",
        f"| Cohen's dz | {_fmt_stat(comparison.cohens_dz)} |",
        f"| Effect size | {comparison.effect_size_label} |",
        f"| Wins (candidate score higher) | {comparison.wins} |",
        f"| Losses (candidate score lower) | {comparison.losses} |",
        f"| Ties | {comparison.ties} |",
        "",
        *_slice_markdown(comparison),
        *_markdown_pairwise(comparison),
        *_markdown_judge_agreement(comparison),
        *_markdown_drift(comparison),
        *_markdown_stability(comparison),
        "## Per-case results",
        "",
        "| Case | Baseline score | Candidate score | Diff | Baseline passed | "
        "Candidate passed | Outcome |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in comparison.per_case:
        lines.append(_case_row(row))
    lines.extend(_markdown_changed_outputs(comparison))
    status = "PASS" if verdict.passed else "FAIL"
    lines.extend(["", "## Gate", "", f"**{status}**", ""])
    lines.extend(f"- {reason}" for reason in verdict.reasons)
    lines.append("")
    if failure_clusters is not None:
        lines.extend(_markdown_failure_modes(failure_clusters, embedder))
    return "\n".join(lines)


def render_html(
    comparison: Comparison,
    verdict: GateVerdict,
    *,
    failure_clusters: Sequence[FailureCluster] | None = None,
    embedder: str | None = None,
) -> str:
    """Return a self-contained HTML report, including a trailing newline.

    CSS is inline. The document does not load external assets.
    """
    level = int(round((1.0 - comparison.alpha) * 100))
    status = "pass" if verdict.passed else "fail"
    label = "PASS" if verdict.passed else "FAIL"
    suite = _html(comparison.baseline.suite_name)
    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1">',
        f"  <title>PromptRegress comparison — {suite}</title>",
        "  <style>",
        _HTML_CSS.rstrip("\n"),
        "  </style>",
        "</head>",
        "<body>",
        "  <main>",
        "    <header>",
        "      <h1>PromptRegress comparison</h1>",
        f'      <p class="lede">Suite <code>{suite}</code></p>',
        "    </header>",
        "    <section>",
        "      <h2>Summary</h2>",
        "      <table>",
        "        <thead>",
        "          <tr>",
        '            <th scope="col">Config</th>',
        '            <th scope="col" class="num">Mean score</th>',
        '            <th scope="col" class="num">Pass rate</th>',
        '            <th scope="col" class="num">Total latency (ms)</th>',
        "          </tr>",
        "        </thead>",
        "        <tbody>",
        f"          {_html_run_row(comparison.baseline)}",
        f"          {_html_run_row(comparison.candidate)}",
        "        </tbody>",
        "      </table>",
        "    </section>",
        *_html_usage_section(comparison),
        *_html_cost_section(comparison),
        "    <section>",
        "      <h2>Paired statistics</h2>",
        "      <table>",
        "        <thead>",
        "          <tr>",
        '            <th scope="col">Statistic</th>',
        '            <th scope="col" class="num">Value</th>',
        "          </tr>",
        "        </thead>",
        "        <tbody>",
        _html_stat_row("Mean diff (candidate - baseline)", _num(comparison.mean_diff)),
        _html_stat_row(f"{level}% CI low", _num(comparison.ci_low)),
        _html_stat_row(f"{level}% CI high", _num(comparison.ci_high)),
        _html_stat_row("CI method", comparison.ci_method),
        _html_stat_row("McNemar p-value", _num(comparison.mcnemar_p)),
        _html_stat_row(
            "McNemar b (baseline pass, candidate fail)",
            str(comparison.mcnemar_b),
        ),
        _html_stat_row(
            "McNemar c (baseline fail, candidate pass)",
            str(comparison.mcnemar_c),
        ),
        _html_stat_row("Wilcoxon W", _fmt_stat(comparison.wilcoxon_stat)),
        _html_stat_row("Wilcoxon p-value", _fmt_stat(comparison.wilcoxon_p)),
        _html_stat_row("Paired t", _fmt_stat(comparison.ttest_stat)),
        _html_stat_row("t degrees of freedom", _fmt_stat(comparison.ttest_df)),
        _html_stat_row("Paired t p-value", _fmt_stat(comparison.ttest_p)),
        _html_stat_row("Cohen's dz", _fmt_stat(comparison.cohens_dz)),
        _html_stat_row("Effect size", comparison.effect_size_label),
        _html_stat_row("Wins (candidate score higher)", str(comparison.wins)),
        _html_stat_row("Losses (candidate score lower)", str(comparison.losses)),
        _html_stat_row("Ties", str(comparison.ties)),
        "        </tbody>",
        "      </table>",
        "    </section>",
        *_html_slice_section(comparison),
        *_html_pairwise_section(comparison),
        *_html_judge_agreement_section(comparison),
        *_html_drift_section(comparison),
        *_html_stability_section(comparison),
        "    <section>",
        "      <h2>Gate</h2>",
        f'      <p><span class="verdict {status}">{label}</span></p>',
        "      <ul>",
    ]
    lines.extend(f"        <li>{_html(reason)}</li>" for reason in verdict.reasons)
    lines.extend(
        [
            "      </ul>",
            "    </section>",
            "    <section>",
            "      <h2>Per-case results</h2>",
            "      <table>",
            "        <thead>",
            "          <tr>",
            '            <th scope="col">Case</th>',
            '            <th scope="col" class="num">Baseline score</th>',
            '            <th scope="col" class="num">Candidate score</th>',
            '            <th scope="col" class="num">Diff</th>',
            '            <th scope="col">Baseline passed</th>',
            '            <th scope="col">Candidate passed</th>',
            '            <th scope="col">Outcome</th>',
            '            <th scope="col">Baseline output</th>',
            '            <th scope="col">Candidate output</th>',
            "          </tr>",
            "        </thead>",
            "        <tbody>",
        ]
    )
    lines.extend(_html_case_row(row) for row in comparison.per_case)
    lines.extend(
        [
            "        </tbody>",
            "      </table>",
            "    </section>",
        ]
    )
    lines.extend(_html_diff_section(comparison))
    if failure_clusters is not None:
        lines.extend(_html_failure_modes(failure_clusters, embedder))
    lines.extend(
        [
            "  </main>",
            "</body>",
            "</html>",
            "",
        ]
    )
    return "\n".join(lines)


_HTML_CSS = """
    :root {
      color-scheme: light;
      --ink: #1c1915;
      --muted: #5c564c;
      --line: #e4ddd2;
      --paper: #faf7f2;
      --card: #fffdf9;
      --pass: #1f6b45;
      --pass-bg: #e7f5ec;
      --fail: #8d2b2b;
      --fail-bg: #fdecec;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 2rem 1.25rem 3rem;
      font: 16px/1.5 "Segoe UI", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background: var(--paper);
    }
    main { max-width: 1080px; margin: 0 auto; }
    h1 { font-size: 1.75rem; line-height: 1.2; margin: 0 0 0.25rem; }
    h2 { font-size: 1.1rem; margin: 0 0 0.75rem; }
    .lede { color: var(--muted); margin: 0; }
    section {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 1rem 1.1rem 1.15rem;
      margin: 1rem 0;
    }
    table { width: 100%; border-collapse: collapse; }
    th, td {
      text-align: left;
      padding: 0.4rem 0.5rem;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }
    th {
      font-size: 0.75rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      color: var(--muted);
    }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    .verdict {
      display: inline-block;
      font-weight: 700;
      letter-spacing: 0.04em;
      padding: 0.15rem 0.55rem;
      border-radius: 999px;
    }
    .verdict.pass { color: var(--pass); background: var(--pass-bg); }
    .verdict.fail { color: var(--fail); background: var(--fail-bg); }
    .outcome-loss { color: var(--fail); font-weight: 600; }
    .outcome-win { color: var(--pass); font-weight: 600; }
    .outcome-tie { color: var(--muted); }
    ul { margin: 0.75rem 0 0; padding-left: 1.2rem; }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.92em;
    }
    .output { max-width: 18rem; word-break: break-word; color: var(--muted); }
    details.diff-row { margin-top: 0.75rem; }
    details.diff-row > summary { cursor: pointer; font-weight: 600; }
    .diff-columns {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
      margin-top: 0.55rem;
    }
    .diff-label {
      color: var(--muted);
      font-size: 0.75rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }
    .diff-columns pre {
      margin: 0.3rem 0 0;
      padding: 0.55rem 0.65rem;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.85rem;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 6px;
    }
"""


def comparison_to_dict(
    comparison: Comparison,
    verdict: GateVerdict,
    *,
    failure_clusters: Sequence[FailureCluster] | None = None,
    embedder: str | None = None,
) -> dict[str, Any]:
    """JSON-ready mapping of a comparison and its gate verdict.

    ``failure_clustering`` is added only when ``failure_clusters`` is not
    ``None``. An empty list records that clustering ran and found no
    score regressions.
    """
    payload = {
        "suite_name": comparison.baseline.suite_name,
        "baseline": run_to_dict(comparison.baseline),
        "candidate": run_to_dict(comparison.candidate),
        "mean_diff": comparison.mean_diff,
        "ci_low": comparison.ci_low,
        "ci_high": comparison.ci_high,
        "alpha": comparison.alpha,
        "ci_method": comparison.ci_method,
        "mcnemar_p": comparison.mcnemar_p,
        "mcnemar_b": comparison.mcnemar_b,
        "mcnemar_c": comparison.mcnemar_c,
        "wilcoxon_stat": _json_number(comparison.wilcoxon_stat),
        "wilcoxon_p": _json_number(comparison.wilcoxon_p),
        "ttest_stat": _json_number(comparison.ttest_stat),
        "ttest_df": _json_number(comparison.ttest_df),
        "ttest_p": _json_number(comparison.ttest_p),
        "cohens_dz": _json_number(comparison.cohens_dz),
        "effect_size_label": comparison.effect_size_label,
        "wins": comparison.wins,
        "losses": comparison.losses,
        "ties": comparison.ties,
        "slices": [_slice_to_dict(item) for item in comparison.slices],
        "details": list(comparison.details),
        "per_case": [_row_to_dict(row) for row in comparison.per_case],
        "gate": {"passed": verdict.passed, "reasons": list(verdict.reasons)},
    }
    if comparison.pairwise is not None:
        payload["pairwise"] = _pairwise_to_dict(comparison.pairwise)
    if comparison.judge_agreement:
        payload["judge_agreement"] = [
            _judge_agreement_to_dict(item) for item in comparison.judge_agreement
        ]
    if comparison.drift is not None:
        payload["drift"] = _drift_to_dict(comparison.drift)
    cost_payload = _comparison_cost_dict(comparison)
    if cost_payload is not None:
        payload["cost"] = cost_payload
    if failure_clusters is not None:
        payload["failure_clustering"] = {
            "embedder": embedder,
            "clusters": [_failure_cluster_to_dict(item) for item in failure_clusters],
        }
    return payload


def run_to_dict(result: RunResult) -> dict[str, Any]:
    """JSON-ready mapping of one run, including derived aggregates."""
    payload: dict[str, Any] = {
        "suite_name": result.suite_name,
        "config_name": result.config_name,
        "mean_score": result.mean_score,
        "pass_rate": result.pass_rate,
        "total_latency_ms": result.total_latency_ms,
        "mean_latency_ms": result.mean_latency_ms,
        "p95_latency_ms": result.p95_latency_ms,
        "total_prompt_tokens": result.total_prompt_tokens,
        "total_completion_tokens": result.total_completion_tokens,
        "total_tokens": result.total_tokens,
        "n_cached": result.n_cached,
        "stability": _stability_to_dict(result),
        "results": [_case_result_to_dict(item) for item in result.results],
    }
    if result.cost is not None:
        payload["cost"] = _cost_to_dict(result.cost)
    return payload


def _run_table_row(result: RunResult) -> str:
    return (
        f"| {_cell(result.config_name)} | {result.mean_score:.6f} | "
        f"{result.pass_rate:.6f} | {result.total_latency_ms:.6f} |"
    )


def _score_changed(row: PairedRow) -> bool:
    """True when the paired scores are not a tie."""
    return row.outcome != "tie"


def _output_differs(row: PairedRow) -> bool:
    """True when the score changed or the two outputs are not the same text."""
    return _score_changed(row) or row.baseline_output != row.candidate_output


def _drift_by_case(comparison: Comparison) -> dict[str, float]:
    summary = comparison.drift
    if summary is None:
        return {}
    return {item.case_id: item.drift for item in summary.per_case}


def _quote_block(text: str) -> list[str]:
    """Markdown blockquote. An empty output is a single blank quoted line."""
    if text == "":
        return [">"]
    return ["> " + line for line in text.splitlines()]


def _markdown_changed_outputs(comparison: Comparison) -> list[str]:
    """Quoted baseline and candidate outputs for score changes only.

    Output-only differences stay out of the Markdown report so the section
    stays short. The HTML report includes those rows.
    """
    changed = [row for row in comparison.per_case if _score_changed(row)]
    if not changed:
        return []
    lines = ["", "### Changed outputs", ""]
    for row in changed:
        lines.append(f"#### `{_cell(row.case_id)}`")
        lines.append("")
        lines.append("Baseline:")
        lines.append("")
        lines.extend(_quote_block(row.baseline_output))
        lines.append("")
        lines.append("Candidate:")
        lines.append("")
        lines.extend(_quote_block(row.candidate_output))
        lines.append("")
    return lines


def _html_diff_section(comparison: Comparison) -> list[str]:
    """Collapsible side-by-side outputs for changed scores or changed text.

    Unchanged cases stay in the per-case table and are omitted here. Drift
    is shown only when the comparison recorded it for that case.
    """
    rows = [row for row in comparison.per_case if _output_differs(row)]
    if not rows:
        return []
    drifts = _drift_by_case(comparison)
    lines = [
        "    <section>",
        "      <h2>Output diff</h2>",
        "      <p>Cases whose score changed or whose output differs.</p>",
    ]
    for row in rows:
        summary = f"{_html(row.case_id)} · delta {_num(row.diff)}"
        if row.case_id in drifts:
            summary += f" · drift {_num(drifts[row.case_id])}"
        lines.extend(
            [
                '      <details class="diff-row">',
                f"        <summary>{summary}</summary>",
                '        <div class="diff-columns">',
                "          <div>",
                '            <div class="diff-label">Baseline</div>',
                f"            <pre>{_html(row.baseline_output)}</pre>",
                "          </div>",
                "          <div>",
                '            <div class="diff-label">Candidate</div>',
                f"            <pre>{_html(row.candidate_output)}</pre>",
                "          </div>",
                "        </div>",
                "      </details>",
            ]
        )
    lines.append("    </section>")
    return lines


def _case_row(row: PairedRow) -> str:
    baseline_passed = "true" if row.baseline_passed else "false"
    candidate_passed = "true" if row.candidate_passed else "false"
    return (
        f"| {_cell(row.case_id)} | {row.baseline_score:.6f} | "
        f"{row.candidate_score:.6f} | {row.diff:.6f} | {baseline_passed} | "
        f"{candidate_passed} | {row.outcome} |"
    )


def _row_to_dict(row: PairedRow) -> dict[str, Any]:
    return {
        "case_id": row.case_id,
        "baseline_output": row.baseline_output,
        "candidate_output": row.candidate_output,
        "baseline_score": row.baseline_score,
        "candidate_score": row.candidate_score,
        "diff": row.diff,
        "baseline_passed": row.baseline_passed,
        "candidate_passed": row.candidate_passed,
        "outcome": row.outcome,
        "tags": list(row.tags),
    }


def _case_result_to_dict(result: CaseResult) -> dict[str, Any]:
    payload = {
        "case_id": result.case_id,
        "output": result.output,
        "scores": dict(result.scores),
        "score": result.score,
        "passed": result.passed,
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "cached": result.cached,
        "error": result.error,
        "tags": list(result.tags),
        "sample_scores": list(result.sample_scores),
        "sample_outputs": list(result.sample_outputs),
        "score_sd": result.score_sd,
        "n_samples": result.n_samples,
    }
    if result.judge_scores:
        payload["judge_scores"] = [[label, score] for label, score in result.judge_scores]
    return payload


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _html(text: object) -> str:
    return html.escape(str(text), quote=True)


def _num(value: float) -> str:
    return f"{value:.6f}"


def _fmt_stat(value: float) -> str:
    if not math.isfinite(value):
        if value > 0:
            return "inf"
        if value < 0:
            return "-inf"
        return "nan"
    return f"{value:.6f}"


def _json_number(value: float) -> float | None:
    if math.isfinite(value):
        return value
    return None


def _slice_to_dict(item: SliceSummary) -> dict[str, Any]:
    return {
        "tag": item.tag,
        "n": item.n,
        "mean_diff": item.mean_diff,
        "ci_low": item.ci_low,
        "ci_high": item.ci_high,
        "wins": item.wins,
        "losses": item.losses,
        "ties": item.ties,
    }


def _failure_cluster_to_dict(cluster: FailureCluster) -> dict[str, Any]:
    return {
        "label_terms": list(cluster.label_terms),
        "case_ids": list(cluster.case_ids),
        "size": cluster.size,
        "mean_score_drop": cluster.mean_score_drop,
    }


def _text_failure_modes(
    clusters: Sequence[FailureCluster],
    embedder: str | None,
) -> list[str]:
    lines = ["failure_modes:", f"embedder={embedder or ''}"]
    if not clusters:
        lines.append("- none")
        return lines
    for index, cluster in enumerate(clusters, start=1):
        terms = ",".join(cluster.label_terms)
        cases = ",".join(cluster.case_ids)
        lines.append(
            f"- {index} size={cluster.size} mean_score_drop={cluster.mean_score_drop:.6f} "
            f"label={terms} cases={cases}"
        )
    return lines


def _markdown_failure_modes(
    clusters: Sequence[FailureCluster],
    embedder: str | None,
) -> list[str]:
    lines = ["## Failure modes", "", f"Embedder: `{embedder or ''}`", ""]
    if not clusters:
        lines.extend(["No score regressions to cluster.", ""])
        return lines
    lines.extend(
        [
            "| Cluster | Size | Mean score drop | Label | Cases |",
            "| ---: | ---: | ---: | --- | --- |",
        ]
    )
    for index, cluster in enumerate(clusters, start=1):
        terms = ", ".join(cluster.label_terms)
        cases = ", ".join(cluster.case_ids)
        lines.append(
            f"| {index} | {cluster.size} | {cluster.mean_score_drop:.6f} | "
            f"{_cell(terms)} | {_cell(cases)} |"
        )
    lines.append("")
    return lines


def _html_failure_modes(
    clusters: Sequence[FailureCluster],
    embedder: str | None,
) -> list[str]:
    lines = [
        "    <section>",
        "      <h2>Failure modes</h2>",
        f"      <p>Embedder <code>{_html(embedder or '')}</code></p>",
    ]
    if not clusters:
        lines.extend(
            [
                "      <p>No score regressions to cluster.</p>",
                "    </section>",
            ]
        )
        return lines
    lines.extend(
        [
            "      <table>",
            "        <thead>",
            "          <tr>",
            '            <th scope="col" class="num">Cluster</th>',
            '            <th scope="col" class="num">Size</th>',
            '            <th scope="col" class="num">Mean score drop</th>',
            '            <th scope="col">Label</th>',
            '            <th scope="col">Cases</th>',
            "          </tr>",
            "        </thead>",
            "        <tbody>",
        ]
    )
    for index, cluster in enumerate(clusters, start=1):
        terms = ", ".join(cluster.label_terms)
        cases = ", ".join(cluster.case_ids)
        lines.append(
            "          <tr>"
            f'<td class="num">{index}</td>'
            f'<td class="num">{cluster.size}</td>'
            f'<td class="num">{_num(cluster.mean_score_drop)}</td>'
            f"<td>{_html(terms)}</td>"
            f"<td>{_html(cases)}</td>"
            "</tr>"
        )
    lines.extend(["        </tbody>", "      </table>", "    </section>"])
    return lines


def _slice_markdown(comparison: Comparison) -> list[str]:
    if not comparison.slices:
        return []
    level = int(round((1.0 - comparison.alpha) * 100))
    lines = [
        "## Slices",
        "",
        f"| Tag | n | Mean diff | {level}% CI low | {level}% CI high | Wins | Losses | Ties |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in comparison.slices:
        lines.append(
            f"| {_cell(item.tag)} | {item.n} | {item.mean_diff:.6f} | "
            f"{item.ci_low:.6f} | {item.ci_high:.6f} | {item.wins} | "
            f"{item.losses} | {item.ties} |"
        )
    lines.append("")
    return lines


def _html_slice_section(comparison: Comparison) -> list[str]:
    if not comparison.slices:
        return []
    level = int(round((1.0 - comparison.alpha) * 100))
    lines = [
        "    <section>",
        "      <h2>Slices</h2>",
        "      <table>",
        "        <thead>",
        "          <tr>",
        '            <th scope="col">Tag</th>',
        '            <th scope="col" class="num">n</th>',
        '            <th scope="col" class="num">Mean diff</th>',
        f'            <th scope="col" class="num">{level}% CI low</th>',
        f'            <th scope="col" class="num">{level}% CI high</th>',
        '            <th scope="col" class="num">Wins</th>',
        '            <th scope="col" class="num">Losses</th>',
        '            <th scope="col" class="num">Ties</th>',
        "          </tr>",
        "        </thead>",
        "        <tbody>",
    ]
    for item in comparison.slices:
        lines.append(
            "          <tr>"
            f"<td>{_html(item.tag)}</td>"
            f'<td class="num">{item.n}</td>'
            f'<td class="num">{_num(item.mean_diff)}</td>'
            f'<td class="num">{_num(item.ci_low)}</td>'
            f'<td class="num">{_num(item.ci_high)}</td>'
            f'<td class="num">{item.wins}</td>'
            f'<td class="num">{item.losses}</td>'
            f'<td class="num">{item.ties}</td>'
            "</tr>"
        )
    lines.extend(["        </tbody>", "      </table>", "    </section>"])
    return lines


def _html_bool(value: bool) -> str:
    return "true" if value else "false"


def _html_excerpt(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) > limit:
        compact = compact[: limit - 3] + "..."
    return _html(compact)


def _html_stat_row(label: str, value: str) -> str:
    return f'          <tr><td>{_html(label)}</td><td class="num">{_html(value)}</td></tr>'


def _html_run_row(result: RunResult) -> str:
    return (
        "<tr>"
        f"<td>{_html(result.config_name)}</td>"
        f'<td class="num">{_num(result.mean_score)}</td>'
        f'<td class="num">{_num(result.pass_rate)}</td>'
        f'<td class="num">{_num(result.total_latency_ms)}</td>'
        "</tr>"
    )


def _html_case_row(row: PairedRow) -> str:
    return (
        "          <tr>"
        f"<td>{_html(row.case_id)}</td>"
        f'<td class="num">{_num(row.baseline_score)}</td>'
        f'<td class="num">{_num(row.candidate_score)}</td>'
        f'<td class="num">{_num(row.diff)}</td>'
        f"<td>{_html_bool(row.baseline_passed)}</td>"
        f"<td>{_html_bool(row.candidate_passed)}</td>"
        f'<td class="outcome-{_html(row.outcome)}">{_html(row.outcome)}</td>'
        f'<td class="output">{_html_excerpt(row.baseline_output)}</td>'
        f'<td class="output">{_html_excerpt(row.candidate_output)}</td>'
        "</tr>"
    )


def _text_usage(result: RunResult, label: str) -> list[str]:
    return [
        f"{label}_mean_latency_ms={result.mean_latency_ms:.6f}",
        f"{label}_p95_latency_ms={result.p95_latency_ms:.6f}",
        f"{label}_total_tokens={result.total_tokens}",
        f"{label}_cached_cases={result.n_cached}",
    ]


def _text_pairwise(comparison: Comparison) -> list[str]:
    summary = comparison.pairwise
    if summary is None:
        return []
    return [
        f"pairwise_win_rate={summary.win_rate:.6f}",
        f"pairwise_ci_low={summary.ci_low:.6f}",
        f"pairwise_ci_high={summary.ci_high:.6f}",
        f"pairwise_wins={summary.wins}",
        f"pairwise_losses={summary.losses}",
        f"pairwise_ties={summary.ties}",
    ]


def _text_judge_agreement(comparison: Comparison) -> list[str]:
    if not comparison.judge_agreement:
        return []
    lines = ["judge_agreement:", JUDGE_AGREEMENT_DEFINITION]
    for item in comparison.judge_agreement:
        means = ",".join(
            f"{label}:{mean:.6f}"
            for label, mean in zip(item.labels, item.per_judge_mean, strict=True)
        )
        lines.append(
            f"- side={item.side} config={item.config_name} n_cases={item.n_cases} "
            f"n_judges={item.n_judges} mean_pairwise_abs={item.mean_pairwise_abs:.6f} "
            f"fleiss_kappa={_fmt_kappa(item.fleiss_kappa)} "
            f"pass_threshold={item.pass_threshold:.6f} per_judge_mean={means}"
        )
        lines.extend(f"  note: {note}" for note in item.notes)
    return lines


def _fmt_kappa(value: float | None) -> str:
    if value is None:
        return "undefined"
    return _fmt_stat(value)


def _markdown_usage(comparison: Comparison) -> list[str]:
    return [
        "## Latency and usage",
        "",
        "Latency is the provider call. A cache hit is 0 ms and is marked cached.",
        "Token totals sum the per-case usage fields.",
        "",
        "| Config | Mean latency (ms) | P95 latency (ms) | Total tokens | Cached cases |",
        "| --- | ---: | ---: | ---: | ---: |",
        _usage_row(comparison.baseline),
        _usage_row(comparison.candidate),
        "",
    ]


def _usage_row(result: RunResult) -> str:
    return (
        f"| {_cell(result.config_name)} | {result.mean_latency_ms:.6f} | "
        f"{result.p95_latency_ms:.6f} | {result.total_tokens} | {result.n_cached} |"
    )


def _markdown_pairwise(comparison: Comparison) -> list[str]:
    summary = comparison.pairwise
    if summary is None:
        return []
    level = int(round((1.0 - summary.alpha) * 100))
    lines = [
        "## Pairwise judge",
        "",
        "Each case is judged twice, with the outputs swapped. The score is 1",
        "when the candidate wins both orders, 0.5 on a split or a tie, and 0",
        "when it loses both. The interval is a bootstrap CI for that mean.",
        "It does not change the gate.",
        "",
        f"Rubric: `{_cell(summary.rubric)}`",
        "",
        "| Statistic | Value |",
        "| --- | ---: |",
        f"| Mean pairwise score | {summary.win_rate:.6f} |",
        f"| {level}% CI low | {summary.ci_low:.6f} |",
        f"| {level}% CI high | {summary.ci_high:.6f} |",
        f"| CI method | {summary.ci_method} |",
        "| Wins (both orders) | " + str(summary.wins) + " |",
        "| Losses (both orders) | " + str(summary.losses) + " |",
        "| Splits and ties | " + str(summary.ties) + " |",
        "",
        "| Case | Score | Outcome | Baseline first | Candidate first | Error |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for case in summary.cases:
        error = "" if case.error is None else _cell(case.error)
        lines.append(
            f"| {_cell(case.case_id)} | {case.score:.1f} | {case.outcome} | "
            f"{case.order_baseline_first} | {case.order_candidate_first} | {error} |"
        )
    lines.append("")
    return lines


def _markdown_judge_agreement(comparison: Comparison) -> list[str]:
    if not comparison.judge_agreement:
        return []
    lines = [
        "## Judge ensemble agreement",
        "",
        JUDGE_AGREEMENT_DEFINITION,
        "",
        "| Side | Config | Cases | Judges | Mean pairwise abs | Fleiss' kappa | Per-judge mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in comparison.judge_agreement:
        means = "; ".join(
            f"{_cell(label)} {mean:.6f}"
            for label, mean in zip(item.labels, item.per_judge_mean, strict=True)
        )
        lines.append(
            f"| {item.side} | {_cell(item.config_name)} | {item.n_cases} | {item.n_judges} | "
            f"{item.mean_pairwise_abs:.6f} | {_fmt_kappa(item.fleiss_kappa)} | {means} |"
        )
    notes = [note for item in comparison.judge_agreement for note in item.notes]
    if notes:
        lines.extend(["", *[f"- {_cell(note)}" for note in notes]])
    lines.append("")
    return lines


def _html_usage_section(comparison: Comparison) -> list[str]:
    return [
        "    <section>",
        "      <h2>Latency and usage</h2>",
        "      <p>Latency is the provider call. A cache hit is 0 ms and is marked",
        "      cached. Token totals sum the per-case usage fields.</p>",
        "      <table>",
        "        <thead>",
        "          <tr>",
        '            <th scope="col">Config</th>',
        '            <th scope="col" class="num">Mean latency (ms)</th>',
        '            <th scope="col" class="num">P95 latency (ms)</th>',
        '            <th scope="col" class="num">Total tokens</th>',
        '            <th scope="col" class="num">Cached cases</th>',
        "          </tr>",
        "        </thead>",
        "        <tbody>",
        f"          {_html_usage_row(comparison.baseline)}",
        f"          {_html_usage_row(comparison.candidate)}",
        "        </tbody>",
        "      </table>",
        "    </section>",
    ]


def _html_usage_row(result: RunResult) -> str:
    return (
        "<tr>"
        f"<td>{_html(result.config_name)}</td>"
        f'<td class="num">{_num(result.mean_latency_ms)}</td>'
        f'<td class="num">{_num(result.p95_latency_ms)}</td>'
        f'<td class="num">{result.total_tokens}</td>'
        f'<td class="num">{result.n_cached}</td>'
        "</tr>"
    )


def _html_pairwise_section(comparison: Comparison) -> list[str]:
    summary = comparison.pairwise
    if summary is None:
        return []
    level = int(round((1.0 - summary.alpha) * 100))
    lines = [
        "    <section>",
        "      <h2>Pairwise judge</h2>",
        "      <p>Each case is judged twice, with the outputs swapped. The score is",
        "      1 when the candidate wins both orders, 0.5 on a split or a tie, and",
        "      0 when it loses both. The interval is a bootstrap CI for that mean.",
        "      It does not change the gate.</p>",
        f"      <p>Rubric <code>{_html(summary.rubric)}</code></p>",
        "      <table>",
        "        <thead>",
        "          <tr>",
        '            <th scope="col">Statistic</th>',
        '            <th scope="col" class="num">Value</th>',
        "          </tr>",
        "        </thead>",
        "        <tbody>",
        _html_stat_row("Mean pairwise score", _num(summary.win_rate)),
        _html_stat_row(f"{level}% CI low", _num(summary.ci_low)),
        _html_stat_row(f"{level}% CI high", _num(summary.ci_high)),
        _html_stat_row("CI method", summary.ci_method),
        _html_stat_row("Wins (both orders)", str(summary.wins)),
        _html_stat_row("Losses (both orders)", str(summary.losses)),
        _html_stat_row("Splits and ties", str(summary.ties)),
        "        </tbody>",
        "      </table>",
        "      <table>",
        "        <thead>",
        "          <tr>",
        '            <th scope="col">Case</th>',
        '            <th scope="col" class="num">Score</th>',
        '            <th scope="col">Outcome</th>',
        '            <th scope="col">Baseline first</th>',
        '            <th scope="col">Candidate first</th>',
        '            <th scope="col">Error</th>',
        "          </tr>",
        "        </thead>",
        "        <tbody>",
    ]
    for case in summary.cases:
        lines.append(
            "          <tr>"
            f"<td>{_html(case.case_id)}</td>"
            f'<td class="num">{case.score:.1f}</td>'
            f"<td>{_html(case.outcome)}</td>"
            f"<td>{_html(case.order_baseline_first)}</td>"
            f"<td>{_html(case.order_candidate_first)}</td>"
            f"<td>{_html(case.error or '')}</td>"
            "</tr>"
        )
    lines.extend(["        </tbody>", "      </table>", "    </section>"])
    return lines


def _html_judge_agreement_section(comparison: Comparison) -> list[str]:
    if not comparison.judge_agreement:
        return []
    lines = [
        "    <section>",
        "      <h2>Judge ensemble agreement</h2>",
        f"      <p>{_html(JUDGE_AGREEMENT_DEFINITION)}</p>",
        "      <table>",
        "        <thead>",
        "          <tr>",
        '            <th scope="col">Side</th>',
        '            <th scope="col">Config</th>',
        '            <th scope="col" class="num">Cases</th>',
        '            <th scope="col" class="num">Judges</th>',
        '            <th scope="col" class="num">Mean pairwise abs</th>',
        '            <th scope="col" class="num">Fleiss\' kappa</th>',
        '            <th scope="col">Per-judge mean</th>',
        "          </tr>",
        "        </thead>",
        "        <tbody>",
    ]
    for item in comparison.judge_agreement:
        means = "; ".join(
            f"{label} {mean:.6f}"
            for label, mean in zip(item.labels, item.per_judge_mean, strict=True)
        )
        lines.append(
            "          <tr>"
            f"<td>{_html(item.side)}</td>"
            f"<td>{_html(item.config_name)}</td>"
            f'<td class="num">{item.n_cases}</td>'
            f'<td class="num">{item.n_judges}</td>'
            f'<td class="num">{_num(item.mean_pairwise_abs)}</td>'
            f'<td class="num">{_html(_fmt_kappa(item.fleiss_kappa))}</td>'
            f"<td>{_html(means)}</td>"
            "</tr>"
        )
    lines.extend(["        </tbody>", "      </table>"])
    notes = [note for item in comparison.judge_agreement for note in item.notes]
    if notes:
        lines.append("      <ul>")
        lines.extend(f"        <li>{_html(note)}</li>" for note in notes)
        lines.append("      </ul>")
    lines.extend(["    </section>"])
    return lines


def _judge_agreement_to_dict(item: JudgeAgreement) -> dict[str, Any]:
    return {
        "side": item.side,
        "config_name": item.config_name,
        "n_cases": item.n_cases,
        "n_judges": item.n_judges,
        "mean_pairwise_abs": item.mean_pairwise_abs,
        "fleiss_kappa": item.fleiss_kappa,
        "pass_threshold": item.pass_threshold,
        "per_judge_mean": [
            {"label": label, "mean": mean}
            for label, mean in zip(item.labels, item.per_judge_mean, strict=True)
        ],
        "notes": list(item.notes),
        "definition": JUDGE_AGREEMENT_DEFINITION,
    }


def _pairwise_to_dict(summary: PairwiseSummary) -> dict[str, Any]:
    return {
        "rubric": summary.rubric,
        "wins": summary.wins,
        "losses": summary.losses,
        "ties": summary.ties,
        "win_rate": summary.win_rate,
        "ci_low": summary.ci_low,
        "ci_high": summary.ci_high,
        "alpha": summary.alpha,
        "ci_method": summary.ci_method,
        "n_boot": summary.n_boot,
        "seed": summary.seed,
        "cases": [
            {
                "case_id": case.case_id,
                "outcome": case.outcome,
                "score": case.score,
                "order_baseline_first": case.order_baseline_first,
                "order_candidate_first": case.order_candidate_first,
                "error": case.error,
            }
            for case in summary.cases
        ],
    }


def _write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _fmt_usd(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.8f}"


def _multi_sample(result: RunResult) -> bool:
    return any(item.n_samples > 1 for item in result.results)


def _least_stable(result: RunResult, limit: int = 5) -> list[CaseResult]:
    ranked = sorted(result.results, key=lambda item: (-item.score_sd, item.case_id))
    return ranked[:limit]


def _stability_to_dict(result: RunResult) -> dict[str, Any]:
    return {
        "mean_within_case_sd": result.mean_within_case_sd,
        "least_stable": [
            {"case_id": item.case_id, "score_sd": item.score_sd} for item in _least_stable(result)
        ],
    }


def _cost_to_dict(estimate: CostEstimate) -> dict[str, Any]:
    return {"model": estimate.model, "usd": estimate.usd, "note": estimate.note}


def _comparison_cost_dict(comparison: Comparison) -> dict[str, Any] | None:
    baseline = comparison.baseline.cost
    candidate = comparison.candidate.cost
    if baseline is None and candidate is None:
        return None
    if baseline is None or candidate is None:
        return {
            "baseline": None if baseline is None else _cost_to_dict(baseline),
            "candidate": None if candidate is None else _cost_to_dict(candidate),
            "delta_usd": None,
            "note": "cost was estimated for only one run",
        }
    delta, note = cost_delta(baseline, candidate)
    return {
        "baseline": _cost_to_dict(baseline),
        "candidate": _cost_to_dict(candidate),
        "delta_usd": delta,
        "note": note,
    }


def _drift_to_dict(summary: DriftSummary) -> dict[str, Any]:
    return {
        "embedder": summary.embedder,
        "definition": DRIFT_DEFINITION,
        "mean_drift": summary.mean_drift,
        "max_drift": summary.max_drift,
        "top": [{"case_id": item.case_id, "drift": item.drift} for item in summary.top],
        "per_case": [{"case_id": item.case_id, "drift": item.drift} for item in summary.per_case],
    }


def _text_cost(comparison: Comparison) -> list[str]:
    payload = _comparison_cost_dict(comparison)
    if payload is None:
        return []
    baseline = payload["baseline"]
    candidate = payload["candidate"]
    lines = [
        "baseline_cost_usd=" + _fmt_usd(None if baseline is None else baseline["usd"]),
        "candidate_cost_usd=" + _fmt_usd(None if candidate is None else candidate["usd"]),
        "cost_delta_usd=" + _fmt_usd(payload["delta_usd"]),
    ]
    if payload["note"]:
        lines.append(f"cost_note={payload['note']}")
    return lines


def _text_drift(comparison: Comparison) -> list[str]:
    summary = comparison.drift
    if summary is None:
        return []
    lines = [
        f"drift_embedder={summary.embedder}",
        f"drift_definition={DRIFT_DEFINITION}",
        f"mean_drift={summary.mean_drift:.6f}",
        f"max_drift={summary.max_drift:.6f}",
        "top_drift:",
    ]
    if not summary.top:
        lines.append("- none")
        return lines
    for item in summary.top:
        lines.append(f"- {item.case_id} {item.drift:.6f}")
    return lines


def _text_stability(comparison: Comparison) -> list[str]:
    if not (_multi_sample(comparison.baseline) or _multi_sample(comparison.candidate)):
        return []
    lines = ["stability:"]
    for result in (comparison.baseline, comparison.candidate):
        label = result.config_name
        lines.append(f"{label}_mean_within_case_sd={result.mean_within_case_sd:.6f}")
        lines.append(f"{label}_least_stable:")
        ranked = _least_stable(result)
        if not ranked:
            lines.append("- none")
            continue
        for item in ranked:
            lines.append(f"- {item.case_id} {item.score_sd:.6f}")
    return lines


def _markdown_cost(comparison: Comparison) -> list[str]:
    payload = _comparison_cost_dict(comparison)
    if payload is None:
        return []
    lines = [
        "## Estimated cost",
        "",
        "USD from the pricing file: prompt tokens / 1000 * prompt_per_1k",
        "+ completion tokens / 1000 * completion_per_1k.",
        "The delta is candidate minus baseline. An unknown model is not a number.",
        "Zero token counts cost 0.",
        "",
        "| Config | Model | USD | Note |",
        "| --- | --- | ---: | --- |",
    ]
    for label, estimate in (
        ("baseline", payload["baseline"]),
        ("candidate", payload["candidate"]),
    ):
        if estimate is None:
            lines.append(f"| {label} |  | unknown |  |")
            continue
        note = "" if estimate["note"] is None else _cell(str(estimate["note"]))
        lines.append(
            f"| {label} | {_cell(str(estimate['model']))} | {_fmt_usd(estimate['usd'])} | {note} |"
        )
    lines.append(f"| delta (candidate - baseline) |  | {_fmt_usd(payload['delta_usd'])} |  |")
    if payload["note"]:
        lines.extend(["", _cell(str(payload["note"]))])
    lines.append("")
    return lines


def _markdown_drift(comparison: Comparison) -> list[str]:
    summary = comparison.drift
    if summary is None:
        return []
    lines = [
        "## Semantic drift",
        "",
        DRIFT_DEFINITION + ".",
        "Identical output strings have drift 0.",
        f"Embedder: `{summary.embedder}`",
        "",
        "| Statistic | Value |",
        "| --- | ---: |",
        f"| Mean drift | {summary.mean_drift:.6f} |",
        f"| Max drift | {summary.max_drift:.6f} |",
        "",
        "| Case | Drift |",
        "| --- | ---: |",
    ]
    if not summary.top:
        lines.append("| none |  |")
    for item in summary.top:
        lines.append(f"| {_cell(item.case_id)} | {item.drift:.6f} |")
    lines.append("")
    return lines


def _markdown_stability(comparison: Comparison) -> list[str]:
    if not (_multi_sample(comparison.baseline) or _multi_sample(comparison.candidate)):
        return []
    lines = [
        "## Stability",
        "",
        "Within-case sd is the population standard deviation of that case's",
        "sample scores (divide by N). The suite figure is the mean of those",
        "sds. The lists are the five least stable cases, highest sd first,",
        "then case id.",
        "",
        "| Config | Mean within-case sd |",
        "| --- | ---: |",
        f"| {_cell(comparison.baseline.config_name)} | "
        f"{comparison.baseline.mean_within_case_sd:.6f} |",
        f"| {_cell(comparison.candidate.config_name)} | "
        f"{comparison.candidate.mean_within_case_sd:.6f} |",
        "",
    ]
    for result in (comparison.baseline, comparison.candidate):
        lines.extend(
            [
                f"### {_cell(result.config_name)}",
                "",
                "| Case | SD |",
                "| --- | ---: |",
            ]
        )
        ranked = _least_stable(result)
        if not ranked:
            lines.append("| none |  |")
        for item in ranked:
            lines.append(f"| {_cell(item.case_id)} | {item.score_sd:.6f} |")
        lines.append("")
    return lines


def _html_cost_section(comparison: Comparison) -> list[str]:
    payload = _comparison_cost_dict(comparison)
    if payload is None:
        return []
    lines = [
        "    <section>",
        "      <h2>Estimated cost</h2>",
        "      <p>USD from the pricing file: prompt tokens / 1000 * prompt_per_1k",
        "      + completion tokens / 1000 * completion_per_1k.",
        "      The delta is candidate minus baseline. An unknown model is not a number.",
        "      Zero token counts cost 0.</p>",
        "      <table>",
        "        <thead>",
        "          <tr>",
        '            <th scope="col">Config</th>',
        '            <th scope="col">Model</th>',
        '            <th scope="col" class="num">USD</th>',
        '            <th scope="col">Note</th>',
        "          </tr>",
        "        </thead>",
        "        <tbody>",
    ]
    for label, estimate in (
        ("baseline", payload["baseline"]),
        ("candidate", payload["candidate"]),
    ):
        if estimate is None:
            lines.append(
                f'          <tr><td>{label}</td><td></td><td class="num">unknown</td><td></td></tr>'
            )
            continue
        note = "" if estimate["note"] is None else _html(str(estimate["note"]))
        lines.append(
            "          <tr>"
            f"<td>{label}</td>"
            f"<td>{_html(str(estimate['model']))}</td>"
            f'<td class="num">{_html(_fmt_usd(estimate["usd"]))}</td>'
            f"<td>{note}</td>"
            "</tr>"
        )
    lines.append(
        "          <tr><td>delta (candidate - baseline)</td><td></td>"
        f'<td class="num">{_html(_fmt_usd(payload["delta_usd"]))}</td><td></td></tr>'
    )
    lines.extend(["        </tbody>", "      </table>"])
    if payload["note"]:
        lines.append(f"      <p>{_html(str(payload['note']))}</p>")
    lines.append("    </section>")
    return lines


def _html_drift_section(comparison: Comparison) -> list[str]:
    summary = comparison.drift
    if summary is None:
        return []
    lines = [
        "    <section>",
        "      <h2>Semantic drift</h2>",
        f"      <p>{_html(DRIFT_DEFINITION)}. Identical output strings have drift 0.</p>",
        f"      <p>Embedder <code>{_html(summary.embedder)}</code></p>",
        "      <table>",
        "        <thead>",
        "          <tr>",
        '            <th scope="col">Statistic</th>',
        '            <th scope="col" class="num">Value</th>',
        "          </tr>",
        "        </thead>",
        "        <tbody>",
        _html_stat_row("Mean drift", _num(summary.mean_drift)),
        _html_stat_row("Max drift", _num(summary.max_drift)),
        "        </tbody>",
        "      </table>",
        "      <table>",
        "        <thead>",
        "          <tr>",
        '            <th scope="col">Case</th>',
        '            <th scope="col" class="num">Drift</th>',
        "          </tr>",
        "        </thead>",
        "        <tbody>",
    ]
    if not summary.top:
        lines.append('          <tr><td>none</td><td class="num"></td></tr>')
    for item in summary.top:
        lines.append(
            "          <tr>"
            f"<td>{_html(item.case_id)}</td>"
            f'<td class="num">{_num(item.drift)}</td>'
            "</tr>"
        )
    lines.extend(["        </tbody>", "      </table>", "    </section>"])
    return lines


def _html_stability_section(comparison: Comparison) -> list[str]:
    if not (_multi_sample(comparison.baseline) or _multi_sample(comparison.candidate)):
        return []
    lines = [
        "    <section>",
        "      <h2>Stability</h2>",
        "      <p>Within-case sd is the population standard deviation of that case's",
        "      sample scores (divide by N). The suite figure is the mean of those",
        "      sds. The lists are the five least stable cases, highest sd first,",
        "      then case id.</p>",
        "      <table>",
        "        <thead>",
        "          <tr>",
        '            <th scope="col">Config</th>',
        '            <th scope="col" class="num">Mean within-case sd</th>',
        "          </tr>",
        "        </thead>",
        "        <tbody>",
        "          <tr>"
        f"<td>{_html(comparison.baseline.config_name)}</td>"
        f'<td class="num">{_num(comparison.baseline.mean_within_case_sd)}</td>'
        "</tr>",
        "          <tr>"
        f"<td>{_html(comparison.candidate.config_name)}</td>"
        f'<td class="num">{_num(comparison.candidate.mean_within_case_sd)}</td>'
        "</tr>",
        "        </tbody>",
        "      </table>",
    ]
    for result in (comparison.baseline, comparison.candidate):
        lines.extend(
            [
                f"      <h3>{_html(result.config_name)}</h3>",
                "      <table>",
                "        <thead>",
                "          <tr>",
                '            <th scope="col">Case</th>',
                '            <th scope="col" class="num">SD</th>',
                "          </tr>",
                "        </thead>",
                "        <tbody>",
            ]
        )
        ranked = _least_stable(result)
        if not ranked:
            lines.append('          <tr><td>none</td><td class="num"></td></tr>')
        for item in ranked:
            lines.append(
                "          <tr>"
                f"<td>{_html(item.case_id)}</td>"
                f'<td class="num">{_num(item.score_sd)}</td>'
                "</tr>"
            )
        lines.extend(["        </tbody>", "      </table>"])
    lines.append("    </section>")
    return lines
