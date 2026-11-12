"""Command-line interface for promptregress."""

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from promptregress import __version__
from promptregress.ablate import ablate, ablation_to_dict, format_ablation, write_ablation_json
from promptregress.annotations import emit_annotations, github_annotations_enabled
from promptregress.audit import audit_paths, audit_to_dict, format_audit
from promptregress.baseline import (
    delete_baseline,
    ensure_baseline_matches,
    format_baseline_table,
    list_baselines,
    load_baseline,
    save_baseline,
    validate_baseline_name,
)
from promptregress.cache import ResponseCache
from promptregress.calibration import (
    calibrate_examples,
    calibration_to_dict,
    format_calibration,
    load_calibration_jsonl,
)
from promptregress.changepoint import (
    changepoint_to_dict,
    detect_changepoint,
    format_changepoint,
    write_changepoint_json,
)
from promptregress.cluster import FailureCluster, cluster_failures, make_embedder
from promptregress.doctor import run_doctor
from promptregress.drift import measure_drift, with_drift
from promptregress.flaky import analyze_baselines, failing_cases, flaky_to_dict, format_flaky
from promptregress.gate import GateVerdict, evaluate_gate, evaluate_run_budgets
from promptregress.history import (
    append_history,
    format_history_table,
    format_trend,
    load_history,
    make_compare_record,
    make_run_record,
    suite_points,
    write_trend_html,
)
from promptregress.irt import fit_runs, format_irt, irt_to_dict, load_run_report
from promptregress.junit_xml import strip_unpaired_surrogates, write_compare_junit, write_run_junit
from promptregress.loading import SuiteError, load_suite, suite_schema_path
from promptregress.matrix import (
    MatrixConfig,
    load_matrix,
    render_matrix_markdown,
    run_matrix,
    write_matrix_json,
    write_matrix_markdown,
)
from promptregress.models import Comparison, CostEstimate, PairwiseSummary, RunResult, Suite
from promptregress.pairwise import pairwise_judge, with_pairwise
from promptregress.pareto import (
    format_pareto,
    pareto_from_files,
    pareto_to_dict,
    write_pareto_json,
)
from promptregress.power import (
    POWER_GRID,
    achieved_power,
    format_power,
    load_report_diffs,
    power_grid,
    power_to_dict,
)
from promptregress.pricing import ModelPrice, estimate_cost, load_pricing
from promptregress.providers import (
    AnthropicProvider,
    EchoProvider,
    MockProvider,
    OpenAICompatibleProvider,
    Provider,
    ProviderError,
    provider_model_name,
)
from promptregress.report import format_text, write_html, write_json, write_markdown, write_run_json
from promptregress.risk import (
    RiskRefusal,
    format_refusal,
    format_risk,
    labeled_cases,
    load_comparison,
    rank_cases,
    refusal_to_dict,
    risk_to_dict,
    train,
    write_risk_json,
)
from promptregress.runner import run_suite
from promptregress.scorers import ScoreError
from promptregress.stats import StatsError, cohens_dz, compare_runs
from promptregress.synth import OPERATOR_NAMES, synth_to_path

DEMO_BASELINE_TEMPLATE = "Answer completely.\n{input}"
DEMO_CANDIDATE_TEMPLATE = "Answer briefly.\n{input}"

# Short answers omit the details the bundled assertions require.
REGRESSED_ANSWERS = {
    "password_reset": "Use the reset link.",
    "refund_policy": "We offer refunds.",
    "shipping_time": "Shipping varies.",
}

_INIT_SUITE: dict[str, Any] = {
    "name": "example",
    "description": ("Small suite created by promptregress init. Replace the cases with your own."),
    "cases": [
        {
            "id": "password_reset",
            "input": "How do I reset my password?",
            "reference": "Go to Settings, choose Security, then Reset password.",
            "assertions": [{"kind": "contains", "value": "Settings"}],
            "tags": ["account"],
            "metadata": {},
        },
        {
            "id": "support_hours",
            "input": "What are your support hours?",
            "reference": "Support is available Monday to Friday, 09:00 to 18:00 UTC.",
            "assertions": [
                {"kind": "contains", "value": "Monday"},
                {"kind": "regex", "value": r"\d{2}:\d{2}"},
            ],
            "tags": ["support"],
            "metadata": {},
        },
    ],
}

_INIT_CONFIG = """\
# PromptRegress project file written by `promptregress init`.
# The CLI takes flags. These values are a starting point you can copy:
#
#   promptregress compare suite.json \\
#     --provider echo \\
#     --baseline-template "{input}" \\
#     --candidate-template "Answer briefly.\\n{input}" \\
#     --max-regression 0 \\
#     --report-md report.md \\
#     --report-json report.json \\
#     --ci-method percentile

provider: echo
baseline_template: "{input}"
candidate_template: "Answer briefly.\\n{input}"
max_regression: 0.0
min_pass_rate: 0.0
ci_method: percentile
"""


def bundled_suite_path() -> Path:
    """Path to the support suite shipped inside the package."""
    return Path(__file__).resolve().parent / "data" / "support_qa.json"


def demo_rules(suite: Suite) -> list[tuple[str, str]]:
    """Mock rules that answer fully, except three cases under "Answer briefly."

    Brief-specific rules are listed first so they win over the question text,
    which is a substring of both prompts. Full-answer rules are longest-first
    so a short question cannot shadow a longer one.
    """
    rules: list[tuple[str, str]] = []
    for case in suite.cases:
        short = REGRESSED_ANSWERS.get(case.id)
        if short is not None:
            rules.append((f"Answer briefly.\n{case.input}", short))
    full = [(case.input, case.reference or "") for case in suite.cases]
    full.sort(key=lambda item: len(item[0]), reverse=True)
    rules.extend(full)
    return rules


def run_demo() -> tuple[Comparison, GateVerdict]:
    """Run the bundled offline demo and evaluate the gate at max_regression 0.

    The candidate template starts with ``Answer briefly.`` MockProvider then
    drops required details on at least three cases. The returned gate fails
    when the bootstrap interval still allows a score drop.
    """
    suite = load_suite(bundled_suite_path())
    provider = MockProvider(
        rules=demo_rules(suite),
        default="I do not know.",
        name="mock",
        model="demo",
    )
    baseline = run_suite(
        suite,
        provider,
        "baseline",
        prompt_template=DEMO_BASELINE_TEMPLATE,
    )
    candidate = run_suite(
        suite,
        provider,
        "candidate",
        prompt_template=DEMO_CANDIDATE_TEMPLATE,
    )
    comparison = compare_runs(baseline, candidate)
    verdict = evaluate_gate(comparison, max_regression=0.0)
    return comparison, verdict


def format_demo(comparison: Comparison, verdict: GateVerdict) -> str:
    """Demo header plus the plain-text comparison. Ends with a newline."""
    header = "\n".join(
        [
            "PromptRegress demo",
            "provider=mock",
            f"baseline_template={DEMO_BASELINE_TEMPLATE!r}",
            f"candidate_template={DEMO_CANDIDATE_TEMPLATE!r}",
            "max_regression=0",
            "",
        ]
    )
    return header + format_text(comparison, verdict)


def build_parser() -> argparse.ArgumentParser:
    """Return the ``promptregress`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="promptregress",
        description="Regression testing for LLM prompts and models.",
    )
    parser.add_argument("--version", action="version", version=f"promptregress {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init",
        help="Write an example suite and config into a directory",
        description=(
            "Create DIR if needed and write suite.json and promptregress.yaml. "
            "Existing files of those names are overwritten. The YAML is a starting "
            "point; the CLI reads flags, not that file."
        ),
    )
    init_parser.add_argument("directory", metavar="dir", help="Directory to create")

    run_parser = subparsers.add_parser(
        "run",
        help="Run one suite with one provider",
        description=(
            "Score one suite and write a run JSON. Exits 0 after scoring, "
            "including when cases fail. Exits 1 when --max-cost-usd or "
            "--max-p95-latency-ms is set and the run exceeds that budget. "
            "Use compare for the regression gate."
        ),
    )
    run_parser.add_argument(
        "suite",
        help="Suite file (.json, .yaml, .yml, .jsonl, or .csv)",
    )
    _add_provider_flags(run_parser)
    run_parser.add_argument("--config-name", required=True, help="Name recorded on this run")
    run_parser.add_argument(
        "--out",
        required=True,
        help="Write the run JSON here (the same document --save-baseline stores)",
    )
    _add_junit_flag(run_parser)
    _add_annotation_flag(run_parser)
    _add_budget_flags(run_parser)
    run_parser.add_argument(
        "--template",
        default="{input}",
        help=(
            "Prompt template. {input} and {reference} are replaced; "
            "other braces are left as written (default: {input})"
        ),
    )
    run_parser.add_argument(
        "--save-baseline",
        default=None,
        metavar="NAME",
        help="Write this run to .promptregress/baselines/NAME.json",
    )
    _add_samples_and_pricing(run_parser)
    _add_history_flag(run_parser)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare a candidate with a baseline template or a stored baseline",
        description=(
            "Score a candidate prompt template against a baseline template, or "
            "against a run stored with --save-baseline. --baseline-name does not "
            "call the baseline provider. Exits 1 when the gate fails and 2 on a "
            "bad suite or arguments."
        ),
    )
    compare_parser.add_argument(
        "suite",
        help="Suite file (.json, .yaml, .yml, .jsonl, or .csv)",
    )
    _add_provider_flags(compare_parser)
    compare_parser.add_argument(
        "--baseline-template",
        default=None,
        help=(
            "Baseline prompt template. {input} and {reference} are replaced. "
            "Required unless --baseline-name loads a stored run"
        ),
    )
    compare_parser.add_argument(
        "--candidate-template",
        required=True,
        help="Candidate prompt template. {input} and {reference} are replaced",
    )
    compare_parser.add_argument(
        "--max-regression",
        type=float,
        default=0.0,
        help=(
            "Fail when -ci_low, the bootstrap upper bound on the score drop, "
            "is greater than this value (default: 0)"
        ),
    )
    compare_parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help=(
            "Fail when the candidate pass rate is below this value (default: 0). "
            "This is not a baseline pass rate"
        ),
    )
    compare_parser.add_argument(
        "--max-mean-drift",
        type=float,
        default=None,
        help="Fail when mean semantic drift exceeds this value (unset: no drift gate)",
    )
    _add_budget_flags(compare_parser)
    _add_annotation_flag(compare_parser)
    compare_parser.add_argument(
        "--require-significant",
        action="store_true",
        help="Also fail unless ci_low is above 0",
    )
    _add_samples_and_pricing(compare_parser)
    compare_parser.add_argument("--report-md", default=None, help="Write a Markdown report here")
    compare_parser.add_argument("--report-json", default=None, help="Write a JSON report here")
    compare_parser.add_argument(
        "--report-html",
        default=None,
        help="Write a self-contained HTML report here",
    )
    _add_junit_flag(compare_parser)
    compare_parser.add_argument(
        "--baseline-config-name",
        default="baseline",
        help="Config name recorded when the baseline template is executed (default: baseline)",
    )
    compare_parser.add_argument(
        "--baseline-name",
        default=None,
        metavar="NAME",
        help=(
            "Load .promptregress/baselines/NAME.json as the baseline and do not "
            "call the baseline provider"
        ),
    )
    compare_parser.add_argument(
        "--candidate-name",
        default="candidate",
        help="Config name recorded for the candidate run (default: candidate)",
    )
    compare_parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Bootstrap seed. Also used for pairwise judging and clustering (default: 13)",
    )
    compare_parser.add_argument(
        "--ci-method",
        choices=("percentile", "bca"),
        default="percentile",
        help="Bootstrap CI for the mean difference (default: percentile)",
    )
    compare_parser.add_argument(
        "--cluster-failures",
        action="store_true",
        help="Cluster score regressions and add a failure-mode section to reports",
    )
    compare_parser.add_argument(
        "--embedder",
        choices=("hash", "tfidf", "sentence-transformer"),
        default="hash",
        help="Embedder for semantic drift and --cluster-failures (default: hash)",
    )
    compare_parser.add_argument(
        "--pairwise-judge",
        action="store_true",
        help="Judge baseline vs candidate in both orders and report a win rate",
    )
    compare_parser.add_argument(
        "--judge-provider",
        choices=("mock", "echo", "openai"),
        default=None,
        help=(
            "Provider for --pairwise-judge. Required with that flag. "
            "A suite llm_judge ensemble sets its own providers and does not use this flag"
        ),
    )
    compare_parser.add_argument(
        "--judge-model",
        default=None,
        help="Model name for the pairwise judge",
    )
    compare_parser.add_argument(
        "--judge-rubric",
        default=None,
        help="Rubric for --pairwise-judge. Required with that flag",
    )
    compare_parser.add_argument(
        "--judge-mock-rules",
        default=None,
        help=(
            "JSON list of [needle, response] pairs for the pairwise judge; "
            "response may be a string or a list of strings"
        ),
    )
    compare_parser.add_argument(
        "--judge-mock-default",
        default="tie",
        help="Pairwise mock reply when no rule matches (default: tie)",
    )
    compare_parser.add_argument(
        "--judge-api-key",
        default=None,
        help="API key for the pairwise judge. Falls back to --api-key",
    )
    compare_parser.add_argument(
        "--judge-base-url",
        default=None,
        help="Base URL for the pairwise judge. Falls back to --base-url",
    )
    _add_history_flag(compare_parser)

    matrix_parser = subparsers.add_parser(
        "matrix",
        help="Rank two or more named configs from a YAML matrix",
        description=(
            "Run every named config and rank by mean score. This is a ranking, "
            "not a gate. The file is a YAML list of configs with name, provider, "
            "and template, or a mapping whose keys are the names."
        ),
    )
    matrix_parser.add_argument(
        "suite",
        help="Suite file (.json, .yaml, .yml, .jsonl, or .csv)",
    )
    matrix_parser.add_argument("matrix", help="YAML list or mapping of named configs")
    matrix_parser.add_argument(
        "--mock-rules",
        default=None,
        help=(
            "JSON list of [needle, response] pairs; response may be a string "
            "or a list of strings. Used when a config does not set mock_rules"
        ),
    )
    matrix_parser.add_argument(
        "--mock-default",
        default="",
        help="Mock reply when a config does not set mock_default (default: empty string)",
    )
    matrix_parser.add_argument(
        "--api-key",
        default=None,
        help="API key for openai configs that omit api_key (else PROMPTREGRESS_API_KEY)",
    )
    matrix_parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for openai configs that omit base_url (else PROMPTREGRESS_BASE_URL)",
    )
    matrix_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds for openai configs that omit timeout (default: 60)",
    )
    matrix_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker threads for case scoring (default: 1). Report order stays the suite order",
    )
    matrix_parser.add_argument(
        "--cache",
        default=None,
        help="Response-cache directory. Omit to disable the cache",
    )
    matrix_parser.add_argument("--report-md", default=None, help="Write the Markdown ranking here")
    matrix_parser.add_argument("--report-json", default=None, help="Write the JSON ranking here")
    matrix_parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Bootstrap seed (default: 13)",
    )
    matrix_parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Bootstrap resamples for each config versus the best (default: 2000)",
    )
    matrix_parser.add_argument(
        "--ci-method",
        choices=("percentile", "bca"),
        default="percentile",
        help="Bootstrap CI for the difference from the best config (default: percentile)",
    )
    matrix_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="CI alpha (default: 0.05)",
    )

    ablate_parser = subparsers.add_parser(
        "ablate",
        help="Measure each ## section's effect on the mean score",
        description=(
            "Score --template, then score it again with each ## section removed. "
            "Prints the ablated mean, the mean paired delta versus the full prompt, "
            "a bootstrap confidence interval for that delta, and the cases whose "
            "score changed. Text before the first heading is kept. "
            "{input} and {reference} are replaced after a section is removed. "
            "Exits 2 when the suite or template is rejected. This is a measurement, "
            "not a gate."
        ),
    )
    ablate_parser.add_argument(
        "--suite",
        required=True,
        help="Suite file (.json, .yaml, .yml, .jsonl, or .csv)",
    )
    ablate_parser.add_argument(
        "--template",
        required=True,
        help=(
            "Candidate template. A section heading is a line that starts with "
            "'##', then a space or tab, then a name. {input} and {reference} "
            "are replaced after sections are removed"
        ),
    )
    _add_provider_flags(ablate_parser)
    ablate_parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help=(
            "Score each case this many times and use the mean. "
            "When this is greater than 1, passed is true when the mean is at least 0.5 "
            "(default: 1)"
        ),
    )
    ablate_parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Bootstrap seed (default: 13)",
    )
    ablate_parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Bootstrap resamples for each section's delta (default: 2000)",
    )
    ablate_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Two-sided CI alpha. 0.05 is a 95%% interval (default: 0.05)",
    )
    ablate_parser.add_argument(
        "--ci-method",
        choices=("percentile", "bca"),
        default="percentile",
        help="Bootstrap CI for each section delta (default: percentile)",
    )
    ablate_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of the text report",
    )
    ablate_parser.add_argument("--out", default=None, help="Write the ablation JSON here")

    history_parser = subparsers.add_parser(
        "history",
        help="Print the local run history table",
        description=(
            "Print runs that run and compare appended. Those commands skip the "
            "file when --no-history is set."
        ),
    )
    history_parser.add_argument(
        "--file",
        default=None,
        help="History JSONL path (default: .promptregress/history.jsonl)",
    )

    trend_parser = subparsers.add_parser(
        "trend",
        help="ASCII and HTML sparkline of mean score for one suite",
        description=(
            "Print an ASCII sparkline and write an HTML page for one suite's "
            "mean score in the history file. SUITE is the suite name stored on "
            "the history rows, not a file path."
        ),
    )
    trend_parser.add_argument(
        "suite",
        help="Suite name as stored in the history file, not a file path",
    )
    trend_parser.add_argument("--out", required=True, help="Write the standalone HTML page here")
    trend_parser.add_argument(
        "--file",
        default=None,
        help="History JSONL path (default: .promptregress/history.jsonl)",
    )

    changepoint_parser = subparsers.add_parser(
        "changepoint",
        help="Detect one mean-score shift in the run history",
        description=(
            "Read the same history file as history and test the per-run "
            "metrics.mean_score series, in file order, for one mean shift. "
            "The change index maximises a CUSUM statistic. A seeded "
            "permutation test assigns a p-value. The generator is local to "
            "this command. "
            "It reports a detected shift in the recorded means. It is not "
            "proof of cause, and it does not change the gate. "
            "Exits 0 after a report. Exits 2 when the history is empty, fewer "
            "than 8 scored runs remain, the series is constant, or a "
            "mean_score is non-finite."
        ),
    )
    changepoint_parser.add_argument(
        "--file",
        default=None,
        help="History JSONL path (default: .promptregress/history.jsonl)",
    )
    changepoint_parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help=(
            "Seed for the permutation generator (default: 13). "
            "The generator is local to this command"
        ),
    )
    changepoint_parser.add_argument(
        "--permutations",
        type=int,
        default=2000,
        help="Number of permutations of the series (default: 2000)",
    )
    changepoint_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of the text report",
    )
    changepoint_parser.add_argument(
        "--out",
        default=None,
        help="Write the change-point JSON here",
    )

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="List or delete stored baseline runs",
        description=(
            "List or delete JSON files in .promptregress/baselines. "
            "Names match [A-Za-z0-9._-]+ and are not file paths."
        ),
    )
    baseline_commands = baseline_parser.add_subparsers(dest="baseline_command", required=True)
    baseline_commands.add_parser(
        "list",
        help="List stored baselines (name, date, suite, mean)",
        description=(
            "Print stored baselines sorted by name. "
            "An empty directory prints the header and (no baselines)."
        ),
    )
    delete_parser = baseline_commands.add_parser(
        "delete",
        help="Delete a stored baseline",
        description="Delete .promptregress/baselines/NAME.json.",
    )
    delete_parser.add_argument(
        "name",
        help="Baseline name matching [A-Za-z0-9._-]+, not a file path",
    )

    flaky_parser = subparsers.add_parser(
        "flaky",
        help="Report pass/fail flips across stored baselines of one suite",
        description=(
            "Analyse stored baselines in .promptregress/baselines for one suite name. "
            "Runs are ordered by file mtime, then name. Fewer than 2 runs prints why "
            "and exits 0. --fail-on-flaky RATE exits 1 when any case flip rate is "
            "at least RATE. A case missing from a run is excluded from flip math."
        ),
    )
    flaky_parser.add_argument(
        "suite",
        nargs="?",
        default=None,
        help="Stored suite name (the suite field on the run, not a file path)",
    )
    flaky_parser.add_argument(
        "--suite",
        dest="suite_name",
        default=None,
        help=(
            "Stored suite name. If this and the positional argument are both set, they must match"
        ),
    )
    flaky_parser.add_argument("--out", default=None, help="Write the per-case JSON here")
    flaky_parser.add_argument(
        "--fail-on-flaky",
        type=float,
        default=None,
        help=(
            "Exit 1 when any case flip rate is >= RATE (0 to 1). "
            "Not applied when fewer than 2 runs are stored"
        ),
    )

    subparsers.add_parser(
        "schema",
        help="Print the suite JSON Schema (draft 2020-12)",
        description=(
            "Print the draft 2020-12 JSON Schema for a JSON or YAML suite. "
            "JSONL and CSV are separate loader formats and are not described here."
        ),
    )
    validate_parser = subparsers.add_parser(
        "validate",
        help="Load each suite and print OK or the error",
        description=(
            "Load each suite. Print PATH: OK, or PATH: error: ... Exits 1 if any file is invalid."
        ),
    )
    validate_parser.add_argument(
        "suites",
        nargs="+",
        metavar="FILE",
        help="Suite file (.json, .yaml, .yml, .jsonl, or .csv)",
    )
    audit_parser = subparsers.add_parser(
        "audit",
        help="Lint suite files for silent passes and bad scorers",
        description=(
            "Load each suite and report lint findings. A file that fails to load "
            "is an error finding, not a crash. Errors: no assertions, an unknown "
            "scorer kind, a duplicate case id. Warnings: identical input text, "
            "an empty reference or value on a scorer that needs one. "
            "Exits 1 when any finding is an error. Warnings alone exit 0."
        ),
    )
    audit_parser.add_argument(
        "suites",
        nargs="+",
        metavar="SUITE",
        help="Suite file (.json, .yaml, .yml, .jsonl, or .csv)",
    )
    audit_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of the grouped text report",
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Score a JSONL file with llm_judge and compare it to human scores",
        description=(
            "Run llm_judge on each JSONL row and compare those scores with "
            "human_score. Prints Pearson, Spearman, MAE, Cohen's kappa, and a "
            "split-conformal false-pass threshold at --conformal-alpha."
        ),
    )
    calibrate_parser.add_argument(
        "data",
        help="JSONL of {input, reference, output, human_score}",
    )
    calibrate_parser.add_argument(
        "--rubric",
        required=True,
        help="Grading instructions for llm_judge",
    )
    calibrate_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Pass cut for Cohen's kappa. A score at or above this is a pass (default: 0.5)",
    )
    calibrate_parser.add_argument(
        "--conformal-alpha",
        type=float,
        default=0.1,
        help=(
            "Split-conformal false-pass level in (0, 1). The report includes "
            "the score threshold whose finite-sample bound is at most this "
            "alpha (default: 0.1)"
        ),
    )
    calibrate_parser.add_argument(
        "--judge-provider",
        required=True,
        choices=("mock", "echo", "openai"),
        help="Judge that scores each JSONL row",
    )
    calibrate_parser.add_argument("--judge-model", default=None, help="Model name for the judge")
    calibrate_parser.add_argument(
        "--mock-rules",
        default=None,
        help="JSON list of [needle, response] pairs for --judge-provider mock",
    )
    calibrate_parser.add_argument(
        "--mock-default",
        default="",
        help="Mock reply when no rule matches (default: empty string)",
    )
    calibrate_parser.add_argument(
        "--api-key",
        default=None,
        help="API key for --judge-provider openai (else PROMPTREGRESS_API_KEY)",
    )
    calibrate_parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for --judge-provider openai (else PROMPTREGRESS_BASE_URL)",
    )
    calibrate_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds for --judge-provider openai (default: 60)",
    )
    calibrate_parser.add_argument("--out", default=None, help="Write the calibration JSON here")

    irt_parser = subparsers.add_parser(
        "irt",
        help="Fit a 2PL model to two or more saved run reports",
        description=(
            "Fit a two-parameter logistic model to two or more run JSON files. "
            "One report is an error."
        ),
    )
    irt_parser.add_argument("reports", nargs="+", help="Run JSON files from promptregress run")
    irt_parser.add_argument("--out", default=None, help="Write the IRT JSON here")
    irt_parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Start-jitter seed (default: 13)",
    )
    irt_parser.add_argument(
        "--iterations",
        type=int,
        default=800,
        help="Gradient steps (default: 800)",
    )
    irt_parser.add_argument(
        "--response",
        choices=("score", "passed"),
        default="score",
        help=(
            "score uses the case score in [0, 1] as a soft label; "
            "passed uses 1 or 0 (default: score)"
        ),
    )

    risk_parser = subparsers.add_parser(
        "risk",
        help="Rank cases by learned regression risk from past comparisons",
        description=(
            "Train a logistic regression on comparison JSON reports from "
            "compare --report-json, oldest first. Features are the baseline "
            "side of each case (score, latency, input length, baseline output "
            "length, tag count, and common scorer kinds). Prints the training "
            "size and holdout AUC. With two or more files, the last file is "
            "the holdout and is not used to fit coefficients. With one file, "
            "the last 20% of cases are the holdout. --target ranks that "
            "report's baseline cases. Exits 1 when there are fewer than 20 "
            "labeled cases, only one class, or a one-class split. Exits 2 "
            "on a bad report. The score does not change the gate."
        ),
    )
    risk_parser.add_argument(
        "comparisons",
        nargs="+",
        metavar="COMPARISON",
        help=(
            "Comparison JSON from compare --report-json, oldest first. "
            "With two or more files, the last file is the holdout"
        ),
    )
    risk_parser.add_argument(
        "--target",
        default=None,
        help=(
            "Comparison JSON whose baseline cases are ranked by descending risk. "
            "Omit to print only the training report"
        ),
    )
    risk_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many of the riskiest target cases to print (default: 10)",
    )
    risk_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of the text report",
    )
    risk_parser.add_argument("--out", default=None, help="Write the risk JSON here")

    synth_parser = subparsers.add_parser(
        "synth",
        help="Write adversarial surface variants of a suite",
        description=(
            "Read a suite and write a new suite. Each case input is perturbed "
            "by the selected operators (whitespace and case, punctuation, a "
            "small synonym map, instruction wrappers, a distractor sentence, "
            "a few homoglyphs, and thousands-separator numbers). References "
            "and assertions are copied. The output is an ordinary suite: run "
            "and compare score it with no extra flags. The input file is not "
            "modified. Exits 2 on an unknown operator, an empty suite, or an "
            "unwritable --out path. These edits test surface form. They do "
            "not create new knowledge tests, and keeping the original answer "
            "is assumed, not proven."
        ),
    )
    synth_parser.add_argument("--suite", required=True, help="Suite file to read")
    synth_parser.add_argument(
        "--out",
        required=True,
        help="New suite file to write (.json, .yaml, .yml, .jsonl, or .csv)",
    )
    synth_parser.add_argument(
        "--operators",
        default=None,
        help=(
            "Comma-separated operator names. Default is every operator: "
            + ", ".join(OPERATOR_NAMES)
        ),
    )
    synth_parser.add_argument(
        "--per-case",
        type=int,
        default=1,
        help="Variants per source case per operator (default: 1)",
    )
    synth_parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help=(
            "Seed mixed into each variant (default: 13). Operators that do "
            "not choose among options ignore it"
        ),
    )

    pareto_parser = subparsers.add_parser(
        "pareto",
        help="Pareto frontier of variants from saved comparison and run reports",
        description=(
            "Read comparison JSON from compare --report-json and run JSON from "
            "run --out. A comparison contributes its baseline and its candidate. "
            "Prints each variant's mean score, estimated cost, and mean latency, "
            "whether it is on the Pareto frontier, and which variant dominates it. "
            "Score is maximised. Cost and latency are minimised. A variant is "
            "dominated when another is at least as good on every axis in use and "
            "strictly better on one. If cost is missing on every variant, or "
            "latency is, that axis is left out and the report names it. Matrix "
            "ranking JSON is rejected: it has no per-config cost or latency. "
            "--json prints the report as JSON. --out writes that JSON. Exits 0 "
            "after the report. Exits 2 on a bad file. This does not change the gate."
        ),
    )
    pareto_parser.add_argument(
        "reports",
        nargs="+",
        metavar="FILE",
        help=(
            "Comparison JSON from compare --report-json, or run JSON from run --out. "
            "A comparison contributes its baseline and its candidate"
        ),
    )
    pareto_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of the text table",
    )
    pareto_parser.add_argument("--out", default=None, help="Write the Pareto JSON here")

    power_parser = subparsers.add_parser(
        "power",
        help="Required sample size and achieved power for a paired comparison",
        description=(
            "Print the sample size a paired effect needs, or the achieved power "
            "of a saved comparison. Pass exactly one of --effect and --from-report."
        ),
    )
    power_parser.add_argument(
        "--effect",
        type=float,
        default=None,
        help="Cohen's dz to plan for. Pass this or --from-report, not both",
    )
    power_parser.add_argument(
        "--from-report",
        default=None,
        help=(
            "Comparison JSON from --report-json. Uses per-case diffs for achieved "
            "power and |dz|. Pass this or --effect, not both"
        ),
    )
    power_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Two-sided alpha (default: 0.05)",
    )
    power_parser.add_argument("--out", default=None, help="Write the power JSON here")
    power_parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Seed for the normal simulation (default: 13)",
    )

    subparsers.add_parser(
        "doctor",
        help="Print diagnostics and a one-case mock smoke run",
        description="Print diagnostics and a one-case mock smoke run.",
    )
    subparsers.add_parser(
        "demo",
        help="Run the bundled offline regression demo",
        description=(
            "Run the bundled support_qa suite with the mock provider. "
            "The candidate template drops three cases. Exits 1 when the gate fails."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns 0, 1 from the gate, or 2 on a usage/runtime error."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return _cmd_init(str(args.directory))
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "compare":
            return _cmd_compare(args)
        if args.command == "calibrate":
            return _cmd_calibrate(args)
        if args.command == "irt":
            return _cmd_irt(args)
        if args.command == "risk":
            return _cmd_risk(args)
        if args.command == "synth":
            return _cmd_synth(args)
        if args.command == "pareto":
            return _cmd_pareto(args)
        if args.command == "power":
            return _cmd_power(args)
        if args.command == "matrix":
            return _cmd_matrix(args)
        if args.command == "ablate":
            return _cmd_ablate(args)
        if args.command == "history":
            return _cmd_history(args)
        if args.command == "trend":
            return _cmd_trend(args)
        if args.command == "changepoint":
            return _cmd_changepoint(args)
        if args.command == "baseline":
            return _cmd_baseline(args)
        if args.command == "flaky":
            return _cmd_flaky(args)
        if args.command == "schema":
            return _cmd_schema()
        if args.command == "validate":
            return _cmd_validate(args)
        if args.command == "audit":
            return _cmd_audit(args)
        if args.command == "doctor":
            return _cmd_doctor()
        if args.command == "demo":
            return _cmd_demo()
    except (
        SuiteError,
        StatsError,
        ScoreError,
        ProviderError,
        OSError,
        ValueError,
        ImportError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"error: unknown command {args.command}", file=sys.stderr)
    return 2


def _add_history_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not append this command to .promptregress/history.jsonl",
    )


def _add_annotation_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--github-annotations",
        action="store_true",
        help=(
            "Write a GitHub Actions ::error command to stderr for each failed "
            "case. Also on when GITHUB_ACTIONS=true. A gate failure with no "
            "failed case writes one summary ::error line"
        ),
    )


def _add_budget_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help=(
            "Fail when estimated cost exceeds this many USD. compare uses the "
            "candidate. Requires --pricing for a numeric estimate. Unset: no cost check"
        ),
    )
    parser.add_argument(
        "--max-p95-latency-ms",
        type=float,
        default=None,
        help=(
            "Fail when p95 case latency exceeds this many milliseconds. "
            "compare uses the candidate. Unset: no latency check"
        ),
    )


def _add_junit_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--junit-xml",
        default=None,
        metavar="FILE",
        help=(
            "Write JUnit XML. One testsuite per executed run, named with the "
            "config label. A failed case is a failure. On compare, a candidate "
            "case that still passes but scored lower than the baseline is a "
            "system-out note, not a failure. This flag does not change the exit code"
        ),
    )


def _add_samples_and_pricing(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help=(
            "Score each case this many times and use the mean. "
            "When this is greater than 1, passed is true when the mean is at least 0.5 "
            "(default: 1)"
        ),
    )
    parser.add_argument(
        "--pricing",
        default=None,
        help="YAML or JSON map of model to {prompt_per_1k, completion_per_1k} in USD",
    )


def _add_provider_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        required=True,
        choices=("mock", "echo", "openai", "anthropic"),
        help=(
            "mock matches substring rules, echo returns the prompt, "
            "openai calls an OpenAI-compatible chat API, "
            "anthropic calls the Anthropic Messages API"
        ),
    )
    parser.add_argument(
        "--mock-rules",
        default=None,
        help=(
            "JSON list of [needle, response] pairs; response may be a string "
            "or a list of strings. The first matching needle wins. "
            "Without --mock-vary a list response uses its first string"
        ),
    )
    parser.add_argument(
        "--mock-vary",
        action="store_true",
        help="Cycle mock-rule alternate outputs by sample index (opt-in)",
    )
    parser.add_argument(
        "--mock-default",
        default="",
        help="Mock reply when no rule matches (default: empty string)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key for --provider openai (else PROMPTREGRESS_API_KEY) "
            "or anthropic (else ANTHROPIC_API_KEY)"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Base URL for --provider openai "
            "(else PROMPTREGRESS_BASE_URL, default https://api.openai.com/v1)"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name. openai: else PROMPTREGRESS_MODEL, default gpt-4o-mini. "
            "anthropic: default claude-sonnet-4-6"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds for --provider openai or anthropic (default: 60)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker threads for case scoring (default: 1). Report order stays the suite order",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Response-cache directory. Omit to disable the cache",
    )


def _cmd_init(directory: str) -> int:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    suite_path = root / "suite.json"
    config_path = root / "promptregress.yaml"
    suite_path.write_text(json.dumps(_INIT_SUITE, indent=2) + "\n", encoding="utf-8")
    config_path.write_text(_INIT_CONFIG, encoding="utf-8")
    print(f"Wrote {suite_path}")
    print(f"Wrote {config_path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    baseline_name = None if args.save_baseline is None else str(args.save_baseline)
    if baseline_name is not None:
        validate_baseline_name(baseline_name)
    suite = load_suite(str(args.suite))
    provider = _build_provider(args)
    cache = _cache_from_args(args)
    prices = _pricing_from_args(args)
    result = _attach_cost(
        run_suite(
            suite,
            provider,
            str(args.config_name),
            prompt_template=str(args.template),
            cache=cache,
            max_workers=int(args.workers),
            samples=_samples_from_args(args),
        ),
        provider,
        prices,
    )
    write_run_json(result, str(args.out))
    if args.junit_xml:
        write_run_junit(str(args.junit_xml), result)
    budgets = evaluate_run_budgets(
        result,
        max_cost_usd=_optional_budget(args, "max_cost_usd"),
        max_p95_latency_ms=_optional_budget(args, "max_p95_latency_ms"),
    )
    print(_run_summary(result))
    if budgets.reasons:
        print("reasons:")
        for reason in budgets.reasons:
            print(f"- {strip_unpaired_surrogates(reason)}")
    if not args.no_history:
        _write_history(
            make_run_record(
                result,
                provider=str(args.provider),
                model=provider_model_name(provider),
                template=str(args.template),
            )
        )
    if baseline_name is not None:
        path, overwritten = save_baseline(result, baseline_name)
        if overwritten:
            print(f"warning: overwriting baseline {baseline_name}", file=sys.stderr)
        print(f"saved baseline {baseline_name} to {path}")
    if _annotations_enabled(args):
        emit_annotations(result.results, str(args.suite), budgets)
    return 0 if budgets.passed else 1


def _cmd_compare(args: argparse.Namespace) -> int:
    suite = load_suite(str(args.suite))
    stored_name = None if args.baseline_name is None else str(args.baseline_name)
    baseline_template = None if args.baseline_template is None else str(args.baseline_template)
    if stored_name is None and baseline_template is None:
        raise ValueError(
            "--baseline-template is required unless --baseline-name loads a stored baseline"
        )
    # Resolve a stored baseline before the provider exists, so a bad name,
    # a missing file, or a suite mismatch never constructs or calls it.
    stored: RunResult | None = None
    if stored_name is not None:
        stored = load_baseline(stored_name)
        ensure_baseline_matches(stored, suite, stored_name)
    provider = _build_provider(args)
    cache = _cache_from_args(args)
    workers = int(args.workers)
    samples = _samples_from_args(args)
    prices = _pricing_from_args(args)
    if stored is not None:
        baseline = stored
    else:
        if baseline_template is None:
            raise ValueError(
                "--baseline-template is required unless --baseline-name loads a stored baseline"
            )
        baseline = _attach_cost(
            run_suite(
                suite,
                provider,
                str(args.baseline_config_name),
                prompt_template=baseline_template,
                cache=cache,
                max_workers=workers,
                samples=samples,
            ),
            provider,
            prices,
        )
    candidate = _attach_cost(
        run_suite(
            suite,
            provider,
            str(args.candidate_name),
            prompt_template=str(args.candidate_template),
            cache=cache,
            max_workers=workers,
            samples=samples,
        ),
        provider,
        prices,
    )
    comparison = compare_runs(
        baseline,
        candidate,
        seed=int(args.seed),
        ci_method=str(args.ci_method),
    )
    embedder_name = str(args.embedder)
    embedder = make_embedder(embedder_name)
    comparison = with_drift(
        comparison,
        measure_drift(comparison, embedder, embedder_name=embedder_name),
    )
    if args.pairwise_judge:
        comparison = with_pairwise(comparison, _pairwise_from_args(args, suite, comparison))
    verdict = evaluate_gate(
        comparison,
        max_regression=float(args.max_regression),
        min_candidate_pass_rate=float(args.min_pass_rate),
        require_significant=bool(args.require_significant),
        max_mean_drift=None if args.max_mean_drift is None else float(args.max_mean_drift),
        max_cost_usd=_optional_budget(args, "max_cost_usd"),
        max_p95_latency_ms=_optional_budget(args, "max_p95_latency_ms"),
    )
    failure_clusters: list[FailureCluster] | None = None
    reported_embedder: str | None = None
    if args.cluster_failures:
        reported_embedder = embedder_name
        losses = [row for row in comparison.per_case if row.outcome == "loss"]
        failure_clusters = cluster_failures(
            losses,
            embedder=embedder,
            seed=int(args.seed),
        )
    if args.report_json:
        write_json(
            comparison,
            str(args.report_json),
            verdict,
            failure_clusters=failure_clusters,
            embedder=reported_embedder,
        )
    if args.report_md:
        write_markdown(
            comparison,
            str(args.report_md),
            verdict,
            failure_clusters=failure_clusters,
            embedder=reported_embedder,
        )
    if args.report_html:
        write_html(
            comparison,
            str(args.report_html),
            verdict,
            failure_clusters=failure_clusters,
            embedder=reported_embedder,
        )
    if args.junit_xml:
        write_compare_junit(str(args.junit_xml), comparison)
    print(
        format_text(
            comparison,
            verdict,
            failure_clusters=failure_clusters,
            embedder=reported_embedder,
        ),
        end="",
    )
    if not args.no_history:
        _write_history(
            make_compare_record(
                comparison,
                provider=str(args.provider),
                model=provider_model_name(provider),
                baseline_template="" if baseline_template is None else baseline_template,
                candidate_template=str(args.candidate_template),
                gate_passed=verdict.passed,
            )
        )
    if _annotations_enabled(args):
        emit_annotations(comparison.candidate.results, str(args.suite), verdict)
    return 0 if verdict.passed else 1


def _cmd_matrix(args: argparse.Namespace) -> int:
    suite = load_suite(str(args.suite))
    configs = load_matrix(str(args.matrix))
    entries = [(config, _provider_for_matrix(config, args)) for config in configs]
    result = run_matrix(
        suite,
        entries,
        cache=_cache_from_args(args),
        max_workers=int(args.workers),
        n_boot=int(args.n_boot),
        seed=int(args.seed),
        alpha=float(args.alpha),
        ci_method=str(args.ci_method),
    )
    if args.report_json:
        write_matrix_json(result, str(args.report_json))
    if args.report_md:
        write_matrix_markdown(result, str(args.report_md))
    print(render_matrix_markdown(result), end="")
    return 0


def _cmd_ablate(args: argparse.Namespace) -> int:
    suite = load_suite(str(args.suite))
    provider = _build_provider(args)
    report = ablate(
        suite,
        provider,
        str(args.template),
        cache=_cache_from_args(args),
        max_workers=int(args.workers),
        samples=_samples_from_args(args),
        n_boot=int(args.n_boot),
        seed=int(args.seed),
        alpha=float(args.alpha),
        ci_method=str(args.ci_method),
    )
    payload = ablation_to_dict(report)
    if args.out:
        write_ablation_json(payload, str(args.out))
    if args.json:
        print(json.dumps(payload, indent=2, allow_nan=False))
    else:
        print(format_ablation(report), end="")
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    if args.baseline_command == "list":
        print(format_baseline_table(list_baselines()), end="")
        return 0
    if args.baseline_command == "delete":
        delete_baseline(str(args.name))
        print(f"deleted baseline {args.name}")
        return 0
    print(f"error: unknown baseline command {args.baseline_command}", file=sys.stderr)
    return 2


def _cmd_flaky(args: argparse.Namespace) -> int:
    suite_name = _flaky_suite_name(args)
    rate = _flaky_rate(getattr(args, "fail_on_flaky", None))
    report = analyze_baselines(suite_name)
    print(format_flaky(report), end="")
    if args.out:
        _write_json(str(args.out), flaky_to_dict(report))
    tripped = () if rate is None else failing_cases(report, rate)
    for case in tripped:
        flip_rate = case.flip_rate
        if flip_rate is None or rate is None:
            continue
        print(f"flaky: {case.case_id} flip_rate={flip_rate:.6f} >= {rate:.6f}")
    return 1 if tripped else 0


def _flaky_suite_name(args: argparse.Namespace) -> str:
    positional = None if args.suite is None else str(args.suite).strip()
    flagged = None if args.suite_name is None else str(args.suite_name).strip()
    if positional == "":
        positional = None
    if flagged == "":
        flagged = None
    if positional is not None and flagged is not None and positional != flagged:
        raise ValueError(f"suite argument {positional!r} does not match --suite {flagged!r}")
    chosen = positional if positional is not None else flagged
    if chosen is None:
        raise ValueError("flaky requires a suite name (positional argument or --suite)")
    return chosen


def _flaky_rate(raw: object) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("--fail-on-flaky must be a finite number in [0, 1]")
    rate = float(raw)
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError("--fail-on-flaky must be a finite number in [0, 1]")
    return rate


def _cmd_schema() -> int:
    print(suite_schema_path().read_text(encoding="utf-8"), end="")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    failed = False
    for raw in args.suites:
        path = str(raw)
        try:
            load_suite(path)
        except SuiteError as exc:
            print(f"{path}: error: {exc}")
            failed = True
            continue
        print(f"{path}: OK")
    return 1 if failed else 0


def _cmd_audit(args: argparse.Namespace) -> int:
    report = audit_paths([str(path) for path in args.suites])
    if args.json:
        print(json.dumps(audit_to_dict(report), indent=2))
    else:
        print(format_audit(report), end="")
    return 1 if report.error_count else 0


def _cmd_history(args: argparse.Namespace) -> int:
    records, notes = load_history(None if args.file is None else str(args.file))
    for note in notes:
        print(f"warning: {note}", file=sys.stderr)
    print(format_history_table(records), end="")
    return 0


def _cmd_changepoint(args: argparse.Namespace) -> int:
    records, notes = load_history(None if args.file is None else str(args.file))
    for note in notes:
        print(f"warning: {note}", file=sys.stderr)
    report = detect_changepoint(
        records,
        seed=args.seed,
        permutations=args.permutations,
    )
    payload = changepoint_to_dict(report)
    if args.out:
        write_changepoint_json(payload, str(args.out))
    if args.json:
        print(json.dumps(payload, indent=2, allow_nan=False))
    else:
        print(format_changepoint(report), end="")
    return 0


def _cmd_trend(args: argparse.Namespace) -> int:
    records, notes = load_history(None if args.file is None else str(args.file))
    for note in notes:
        print(f"warning: {note}", file=sys.stderr)
    suite = str(args.suite)
    points = suite_points(records, suite)
    if not points:
        raise ValueError(f"no history rows with a mean_score for suite {suite!r}")
    values = [score for _stamp, score in points]
    print(format_trend(suite, values), end="")
    write_trend_html(suite, points, str(args.out))
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    examples = load_calibration_jsonl(str(args.data))
    judge = _build_judge(args)
    report = calibrate_examples(
        examples,
        judge,
        str(args.rubric),
        threshold=float(args.threshold),
        conformal_alpha=float(args.conformal_alpha),
    )
    if args.out:
        _write_json(str(args.out), calibration_to_dict(report))
    print(format_calibration(report), end="")
    return 0


def _cmd_pareto(args: argparse.Namespace) -> int:
    report = pareto_from_files([str(path) for path in args.reports])
    payload = pareto_to_dict(report)
    if args.out:
        write_pareto_json(payload, str(args.out))
    if args.json:
        print(json.dumps(payload, indent=2, allow_nan=False))
    else:
        print(format_pareto(report), end="")
    return 0


def _cmd_risk(args: argparse.Namespace) -> int:
    comparisons = [load_comparison(str(path)) for path in args.comparisons]
    target = None if args.target is None else load_comparison(str(args.target))
    top = args.top
    if isinstance(top, bool) or not isinstance(top, int) or top < 1:
        raise ValueError("top must be an integer >= 1")
    try:
        model = train(comparisons)
    except RiskRefusal as exc:
        payload = refusal_to_dict(exc)
        if args.out:
            write_risk_json(payload, str(args.out))
        if args.json:
            print(json.dumps(payload, indent=2, allow_nan=False))
        else:
            print(format_refusal(exc), end="")
        return 1
    ranking = None
    observed = None
    if target is not None:
        ranking = rank_cases(model, target.baseline.results)[:top]
        observed = {row.case_id: row.label for row in labeled_cases(target)}
    payload = risk_to_dict(
        model,
        ranking,
        target=None if args.target is None else str(args.target),
        observed=observed,
    )
    if args.out:
        write_risk_json(payload, str(args.out))
    if args.json:
        print(json.dumps(payload, indent=2, allow_nan=False))
    else:
        print(
            format_risk(
                model,
                ranking,
                target=None if args.target is None else str(args.target),
                observed=observed,
            ),
            end="",
        )
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    summary = synth_to_path(
        args.suite,
        args.out,
        operators=None if args.operators is None else str(args.operators),
        per_case=int(args.per_case),
        seed=int(args.seed),
    )
    print(summary, end="")
    return 0


def _cmd_irt(args: argparse.Namespace) -> int:
    runs = [load_run_report(str(path)) for path in args.reports]
    result = fit_runs(
        runs,
        response=str(args.response),
        seed=int(args.seed),
        iterations=int(args.iterations),
    )
    if args.out:
        _write_json(str(args.out), irt_to_dict(result))
    print(format_irt(result), end="")
    return 0


def _cmd_power(args: argparse.Namespace) -> int:
    if args.effect is None and args.from_report is None:
        raise ValueError("power requires --effect or --from-report")
    if args.effect is not None and args.from_report is not None:
        raise ValueError("pass only one of --effect and --from-report")
    alpha = float(args.alpha)
    achieved = None
    observed = None
    if args.from_report is not None:
        diffs = load_report_diffs(str(args.from_report))
        achieved = achieved_power(diffs, alpha)
        observed = cohens_dz(diffs)
        effect = abs(observed)
        if effect == 0.0:
            text = format_power([], achieved=achieved, observed_dz=observed)
            if args.out:
                _write_json(
                    str(args.out),
                    power_to_dict([], achieved=achieved, observed_dz=observed),
                )
            print(text, end="")
            return 0
        if not math.isfinite(effect):
            effect = math.inf
    else:
        effect = float(args.effect)
    estimates = power_grid(effect, alpha, powers=POWER_GRID, seed=int(args.seed))
    if args.out:
        _write_json(
            str(args.out),
            power_to_dict(estimates, achieved=achieved, observed_dz=observed),
        )
    print(format_power(estimates, achieved=achieved, observed_dz=observed), end="")
    return 0


def _cmd_demo() -> int:
    comparison, verdict = run_demo()
    print(format_demo(comparison, verdict), end="")
    return 0 if verdict.passed else 1


def _cmd_doctor() -> int:
    report, ok = run_doctor()
    print(report, end="")
    return 0 if ok else 1


def _annotations_enabled(args: argparse.Namespace) -> bool:
    return github_annotations_enabled(bool(getattr(args, "github_annotations", False)))


def _optional_budget(args: argparse.Namespace, name: str) -> float | None:
    value = getattr(args, name, None)
    if value is None:
        return None
    return float(value)


def _samples_from_args(args: argparse.Namespace) -> int:
    samples = args.samples
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be an integer >= 1")
    return samples


def _pricing_from_args(args: argparse.Namespace) -> dict[str, ModelPrice] | None:
    raw = getattr(args, "pricing", None)
    if raw is None or raw == "":
        return None
    return load_pricing(str(raw))


def _attach_cost(
    result: RunResult,
    provider: Provider,
    prices: dict[str, ModelPrice] | None,
) -> RunResult:
    if prices is None:
        return result
    return replace(result, cost=estimate_cost(result, provider_model_name(provider), prices))


def _run_summary(result: RunResult) -> str:
    text = (
        f"suite={result.suite_name} config={result.config_name} "
        f"mean_score={result.mean_score:.6f} pass_rate={result.pass_rate:.6f} "
        f"cases={result.n_cases} "
        f"mean_latency_ms={result.mean_latency_ms:.6f} "
        f"p95_latency_ms={result.p95_latency_ms:.6f} "
        f"total_tokens={result.total_tokens}"
    )
    if result.cost is not None:
        text += f" estimated_cost_usd={_fmt_usd(result.cost)}"
        if result.cost.note:
            text += f" cost_note={result.cost.note}"
    if result.results and result.results[0].n_samples > 1:
        text += (
            f" samples={result.results[0].n_samples} "
            f"mean_within_case_sd={result.mean_within_case_sd:.6f}"
        )
    # Match format_text / audit: strip unpaired surrogates so a suite name
    # from a ``\\ud800`` JSON escape cannot crash a UTF-8 stdout.
    return strip_unpaired_surrogates(text)


def _fmt_usd(estimate: CostEstimate) -> str:
    if estimate.usd is None:
        return "unknown"
    return f"{estimate.usd:.8f}"


def _cache_from_args(args: argparse.Namespace) -> ResponseCache | None:
    if not args.cache:
        return None
    return ResponseCache(str(args.cache))


def _build_judge(args: argparse.Namespace) -> Provider:
    model = None if args.judge_model is None else str(args.judge_model)
    api_key = args.api_key if args.api_key is None else str(args.api_key)
    base_url = args.base_url if args.base_url is None else str(args.base_url)
    return _make_judge(
        kind=str(args.judge_provider),
        model=model,
        mock_rules=args.mock_rules,
        mock_default=str(args.mock_default),
        api_key=api_key,
        base_url=base_url,
        timeout=float(args.timeout),
    )


def _pairwise_from_args(
    args: argparse.Namespace,
    suite: Suite,
    comparison: Comparison,
) -> PairwiseSummary:
    if not args.judge_provider:
        raise ValueError("--pairwise-judge requires --judge-provider")
    if args.judge_rubric is None or str(args.judge_rubric).strip() == "":
        raise ValueError("--pairwise-judge requires --judge-rubric")
    model = None if args.judge_model is None else str(args.judge_model)
    api_key = args.judge_api_key if args.judge_api_key is not None else args.api_key
    base_url = args.judge_base_url if args.judge_base_url is not None else args.base_url
    if api_key is not None:
        api_key = str(api_key)
    if base_url is not None:
        base_url = str(base_url)
    judge = _make_judge(
        kind=str(args.judge_provider),
        model=model,
        mock_rules=args.judge_mock_rules,
        mock_default=str(args.judge_mock_default),
        api_key=api_key,
        base_url=base_url,
        timeout=float(args.timeout),
    )
    return pairwise_judge(
        comparison,
        suite,
        judge,
        str(args.judge_rubric),
        seed=int(args.seed),
        ci_method=str(args.ci_method),
    )


def _make_judge(
    *,
    kind: str,
    model: str | None,
    mock_rules: str | None,
    mock_default: str,
    api_key: str | None,
    base_url: str | None,
    timeout: float,
) -> Provider:
    if kind == "mock":
        return MockProvider(
            rules=_parse_mock_rules(mock_rules),
            default=mock_default,
            name="mock",
            model=model or "mock",
        )
    if kind == "echo":
        return EchoProvider()
    if kind == "openai":
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            name="openai",
        )
    raise ValueError(f"Unknown judge provider {kind}")


def _provider_for_matrix(config: MatrixConfig, args: argparse.Namespace) -> Provider:
    if config.provider == "mock":
        if config.mock_default is None:
            default = str(args.mock_default)
        else:
            default = config.mock_default
        model = config.model or "mock"
        if config.mock_rules is not None:
            return MockProvider(
                rules=list(config.mock_rules),
                default=default,
                name="mock",
                model=model,
            )
        return MockProvider(
            rules=_parse_mock_rules(args.mock_rules),
            default=default,
            name="mock",
            model=model,
        )
    if config.provider == "echo":
        return EchoProvider()
    if config.provider == "openai":
        api_key = config.api_key if config.api_key is not None else args.api_key
        base_url = config.base_url if config.base_url is not None else args.base_url
        timeout = float(args.timeout) if config.timeout is None else config.timeout
        if api_key is not None:
            api_key = str(api_key)
        if base_url is not None:
            base_url = str(base_url)
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=base_url,
            model=config.model,
            timeout=timeout,
        )
    raise ValueError(f"Unknown provider {config.provider}")


def _write_history(record: dict[str, Any]) -> None:
    """Append one history line. A failure warns and does not fail the command."""
    try:
        append_history(record)
    except Exception as exc:
        print(f"warning: could not write run history ({exc})", file=sys.stderr)


def _write_json(path: str, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_provider(args: argparse.Namespace) -> Provider:
    kind = str(args.provider)
    if kind == "mock":
        rules = _parse_mock_rules(args.mock_rules)
        return MockProvider(
            rules=rules,
            default=str(args.mock_default),
            vary=bool(getattr(args, "mock_vary", False)),
        )
    if kind == "echo":
        return EchoProvider()
    if kind == "openai":
        timeout = float(args.timeout)
        api_key = args.api_key if args.api_key is None else str(args.api_key)
        base_url = args.base_url if args.base_url is None else str(args.base_url)
        model = args.model if args.model is None else str(args.model)
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )
    if kind == "anthropic":
        timeout = float(args.timeout)
        api_key = args.api_key if args.api_key is None else str(args.api_key)
        model = args.model if args.model is None else str(args.model)
        return AnthropicProvider(
            api_key=api_key,
            model=model,
            timeout=timeout,
        )
    raise ValueError(f"Unknown provider {kind}")


def _parse_mock_rules(raw: str | None) -> list[tuple[str, str | tuple[str, ...]]]:
    if raw is None or raw == "":
        return []
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("--mock-rules must be a JSON list of [needle, response] pairs")
    rules: list[tuple[str, str | tuple[str, ...]]] = []
    for index, item in enumerate(data):
        if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
            raise ValueError(
                f"--mock-rules item {index} must be [needle, response] "
                "with a string needle and a string or list-of-strings response"
            )
        needle = item[0]
        response = item[1]
        if isinstance(response, str):
            rules.append((needle, response))
            continue
        if (
            isinstance(response, list)
            and response
            and all(isinstance(part, str) for part in response)
        ):
            rules.append((needle, tuple(response)))
            continue
        raise ValueError(
            f"--mock-rules item {index} response must be a string or a non-empty list of strings"
        )
    return rules


if __name__ == "__main__":
    raise SystemExit(main())
