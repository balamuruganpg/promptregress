"""Command-line interface for promptregress."""

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from promptregress import __version__
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
from promptregress.cluster import FailureCluster, cluster_failures, make_embedder
from promptregress.doctor import run_doctor
from promptregress.drift import measure_drift, with_drift
from promptregress.gate import GateVerdict, evaluate_gate
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
    EchoProvider,
    MockProvider,
    OpenAICompatibleProvider,
    Provider,
    ProviderError,
    provider_model_name,
)
from promptregress.report import format_text, write_html, write_json, write_markdown, write_run_json
from promptregress.runner import run_suite
from promptregress.scorers import ScoreError
from promptregress.stats import StatsError, cohens_dz, compare_runs

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
    )
    init_parser.add_argument("directory", metavar="dir")

    run_parser = subparsers.add_parser("run", help="Run one suite with one provider")
    run_parser.add_argument("suite")
    _add_provider_flags(run_parser)
    run_parser.add_argument("--config-name", required=True)
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--template", default="{input}")
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
        help="Compare a baseline prompt template with a candidate template",
    )
    compare_parser.add_argument("suite")
    _add_provider_flags(compare_parser)
    compare_parser.add_argument(
        "--baseline-template",
        default=None,
        help="Baseline prompt template. Required unless --baseline-name loads a stored run",
    )
    compare_parser.add_argument("--candidate-template", required=True)
    compare_parser.add_argument("--max-regression", type=float, default=0.0)
    compare_parser.add_argument("--min-pass-rate", type=float, default=0.0)
    compare_parser.add_argument(
        "--max-mean-drift",
        type=float,
        default=None,
        help="Fail when mean semantic drift exceeds this value (unset: no drift gate)",
    )
    compare_parser.add_argument("--require-significant", action="store_true")
    _add_samples_and_pricing(compare_parser)
    compare_parser.add_argument("--report-md", default=None)
    compare_parser.add_argument("--report-json", default=None)
    compare_parser.add_argument("--report-html", default=None)
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
    compare_parser.add_argument("--candidate-name", default="candidate")
    compare_parser.add_argument("--seed", type=int, default=13)
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
    )
    compare_parser.add_argument("--judge-model", default=None)
    compare_parser.add_argument("--judge-rubric", default=None)
    compare_parser.add_argument(
        "--judge-mock-rules",
        default=None,
        help="JSON list of [needle, response] pairs for the pairwise judge",
    )
    compare_parser.add_argument("--judge-mock-default", default="tie")
    compare_parser.add_argument("--judge-api-key", default=None)
    compare_parser.add_argument("--judge-base-url", default=None)
    _add_history_flag(compare_parser)

    matrix_parser = subparsers.add_parser(
        "matrix",
        help="Rank two or more named configs from a YAML matrix",
    )
    matrix_parser.add_argument("suite")
    matrix_parser.add_argument("matrix", help="YAML list or mapping of named configs")
    matrix_parser.add_argument(
        "--mock-rules",
        default=None,
        help="JSON list of [needle, response] pairs",
    )
    matrix_parser.add_argument("--mock-default", default="")
    matrix_parser.add_argument("--api-key", default=None)
    matrix_parser.add_argument("--base-url", default=None)
    matrix_parser.add_argument("--timeout", type=float, default=60.0)
    matrix_parser.add_argument("--workers", type=int, default=1)
    matrix_parser.add_argument("--cache", default=None)
    matrix_parser.add_argument("--report-md", default=None)
    matrix_parser.add_argument("--report-json", default=None)
    matrix_parser.add_argument("--seed", type=int, default=13)
    matrix_parser.add_argument("--n-boot", type=int, default=2000)
    matrix_parser.add_argument(
        "--ci-method",
        choices=("percentile", "bca"),
        default="percentile",
    )
    matrix_parser.add_argument("--alpha", type=float, default=0.05)

    history_parser = subparsers.add_parser("history", help="Print the local run history table")
    history_parser.add_argument(
        "--file",
        default=None,
        help="History JSONL path (default: .promptregress/history.jsonl)",
    )

    trend_parser = subparsers.add_parser(
        "trend",
        help="ASCII and HTML sparkline of mean score for one suite",
    )
    trend_parser.add_argument("suite")
    trend_parser.add_argument("--out", required=True, help="Write the standalone HTML page here")
    trend_parser.add_argument(
        "--file",
        default=None,
        help="History JSONL path (default: .promptregress/history.jsonl)",
    )

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="List or delete stored baseline runs",
    )
    baseline_commands = baseline_parser.add_subparsers(dest="baseline_command", required=True)
    baseline_commands.add_parser("list", help="List stored baselines (name, date, suite, mean)")
    delete_parser = baseline_commands.add_parser("delete", help="Delete a stored baseline")
    delete_parser.add_argument("name")

    subparsers.add_parser("schema", help="Print the suite JSON Schema (draft 2020-12)")
    validate_parser = subparsers.add_parser(
        "validate",
        help="Load each suite and print OK or the error",
    )
    validate_parser.add_argument("suites", nargs="+", metavar="FILE")

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Score a JSONL file with llm_judge and compare it to human scores",
    )
    calibrate_parser.add_argument("data", help="JSONL of {input, reference, output, human_score}")
    calibrate_parser.add_argument("--rubric", required=True)
    calibrate_parser.add_argument("--threshold", type=float, default=0.5)
    calibrate_parser.add_argument(
        "--judge-provider",
        required=True,
        choices=("mock", "echo", "openai"),
    )
    calibrate_parser.add_argument("--judge-model", default=None)
    calibrate_parser.add_argument(
        "--mock-rules",
        default=None,
        help="JSON list of [needle, response] pairs for --judge-provider mock",
    )
    calibrate_parser.add_argument("--mock-default", default="")
    calibrate_parser.add_argument("--api-key", default=None)
    calibrate_parser.add_argument("--base-url", default=None)
    calibrate_parser.add_argument("--timeout", type=float, default=60.0)
    calibrate_parser.add_argument("--out", default=None, help="Write the calibration JSON here")

    irt_parser = subparsers.add_parser(
        "irt",
        help="Fit a 2PL model to two or more saved run reports",
    )
    irt_parser.add_argument("reports", nargs="+", help="Run JSON files from promptregress run")
    irt_parser.add_argument("--out", default=None, help="Write the IRT JSON here")
    irt_parser.add_argument("--seed", type=int, default=13)
    irt_parser.add_argument("--iterations", type=int, default=800)
    irt_parser.add_argument("--response", choices=("score", "passed"), default="score")

    power_parser = subparsers.add_parser(
        "power",
        help="Required sample size and achieved power for a paired comparison",
    )
    power_parser.add_argument("--effect", type=float, default=None, help="Cohen's dz to plan for")
    power_parser.add_argument(
        "--from-report",
        default=None,
        help="Comparison JSON; uses per-case diffs for achieved power and |dz|",
    )
    power_parser.add_argument("--alpha", type=float, default=0.05)
    power_parser.add_argument("--out", default=None, help="Write the power JSON here")
    power_parser.add_argument("--seed", type=int, default=13)

    subparsers.add_parser(
        "doctor",
        help="Print diagnostics and a one-case mock smoke run",
        description="Print diagnostics and a one-case mock smoke run.",
    )
    subparsers.add_parser("demo", help="Run the bundled offline regression demo")
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
        if args.command == "power":
            return _cmd_power(args)
        if args.command == "matrix":
            return _cmd_matrix(args)
        if args.command == "history":
            return _cmd_history(args)
        if args.command == "trend":
            return _cmd_trend(args)
        if args.command == "baseline":
            return _cmd_baseline(args)
        if args.command == "schema":
            return _cmd_schema()
        if args.command == "validate":
            return _cmd_validate(args)
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


def _add_samples_and_pricing(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Score each case this many times and use the mean score (default: 1)",
    )
    parser.add_argument(
        "--pricing",
        default=None,
        help="YAML or JSON map of model to {prompt_per_1k, completion_per_1k} in USD",
    )


def _add_provider_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", required=True, choices=("mock", "echo", "openai"))
    parser.add_argument(
        "--mock-rules",
        default=None,
        help="JSON list of [needle, response] pairs; response may be a string or a list of strings",
    )
    parser.add_argument(
        "--mock-vary",
        action="store_true",
        help="Cycle mock-rule alternate outputs by sample index (opt-in)",
    )
    parser.add_argument("--mock-default", default="")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cache", default=None)


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
    print(_run_summary(result))
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
    return 0


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


def _cmd_history(args: argparse.Namespace) -> int:
    records, notes = load_history(None if args.file is None else str(args.file))
    for note in notes:
        print(f"warning: {note}", file=sys.stderr)
    print(format_history_table(records), end="")
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
    )
    if args.out:
        _write_json(str(args.out), calibration_to_dict(report))
    print(format_calibration(report), end="")
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
    return text


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
