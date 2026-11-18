"""Lint suite files, and audit comparison reports for weak cases.

``promptregress audit`` has two modes. Suite files are linted, as described
below. When every path is a comparison report from ``compare --report-json``,
:func:`audit_comparison_files` pools the configs and reports non-discriminating,
misfitting, redundant, and dead cases. That report is advisory. Flags are
advisory — the command exits 0 with a report and never changes the gate.

Suite lint loads each path with :func:`promptregress.loading.load_suite`.
A file that fails to load is one error finding. The command keeps going
and does not raise. The loader rejects a duplicate case id at the first
repeat, so that file contributes the duplicate finding and is not linted
further. Other files are still checked.

Checks, in case order after a successful load:

- ``no_assertions`` (error). The loader allows an empty assertion list.
  ``score_case`` then returns score 0 and ``passed`` false on every output,
  so the case never checks the model text.
- ``unknown_scorer`` (error). An assertion ``kind`` is missing from
  :data:`promptregress.scorers.SCORERS`. :func:`load_scorer_plugins` runs
  first, so entry-point scorers count as known. Built-ins are not replaced.
- ``duplicate_case_id`` (error). Reported from the loader's
  ``Duplicate case id`` error, with the id on the finding.
- ``identical_input`` (warning). Two cases in the same file have the same
  ``input`` string, including whitespace. The later case is the finding and
  names the earlier id. This is the usual copy-paste.
- ``empty_reference`` (warning). A scorer that needs a reference has no
  usable target text.

The registry does not say which kinds need a reference. ``SCORERS`` and
``EXACT_KINDS`` only record the callable and the exact-type pass rule.
:data:`REFERENCE_KINDS` is therefore hardcoded from ``scorers.py``: the
kinds whose target is ``assertion.value`` when it is set, and otherwise
``case.reference`` (``exact_match``, ``contains``, ``levenshtein``,
``tfidf_cosine``). Plugins are not on that list. A blank ``value`` is
empty even when the reference is not, because a set value replaces the
reference. ``None`` falls through to the reference. A missing reference
and a missing value are both empty. Unknown kinds are not given this
warning. Kinds that do not need a reference (``regex``, ``json_valid``,
``json_schema``, ``numeric_tolerance``, ``llm_judge``, ``trace_match``)
are not warned.

Tags that appear once, including suites that would slice on them, are not
checked.

Text output is grouped by file. Each finding line has the case id (``-``
when the finding is about the file), the severity, and the message.
``--json`` prints the same findings as JSON instead of that text. The
process exits 1 when any finding is an error. Warnings alone exit 0.
"""

import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from promptregress.irt import fit_runs
from promptregress.junit_xml import strip_unpaired_surrogates
from promptregress.loading import SuiteError, load_suite, strip_utf8_bom_text
from promptregress.models import AssertionSpec, Case, CaseResult, RunResult
from promptregress.risk import load_comparison
from promptregress.scorers import SCORERS, load_scorer_plugins

Severity = Literal["error", "warning"]

# Kinds that read assertion.value, else case.reference. See scorers._target.
# The registry has no flag for this. Plugins are graded-or-exact only.
REFERENCE_KINDS: frozenset[str] = frozenset(
    {"exact_match", "contains", "levenshtein", "tfidf_cosine"}
)

_DUPLICATE = re.compile(r"^Duplicate case id: (?P<case_id>.*)$")


@dataclass(frozen=True)
class Finding:
    """One lint result. ``case`` is empty when the finding is about the file."""

    file: str
    case: str
    severity: Severity
    check: str
    message: str


@dataclass(frozen=True)
class AuditReport:
    """Findings in file order, then case order."""

    findings: tuple[Finding, ...]

    @property
    def error_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "warning")


def audit_paths(paths: Sequence[str]) -> AuditReport:
    """Lint each path. Load errors become findings. Nothing here raises ``SuiteError``."""
    load_scorer_plugins()
    findings: list[Finding] = []
    for path in paths:
        findings.extend(_audit_path(path))
    return AuditReport(findings=tuple(findings))


def format_audit(report: AuditReport) -> str:
    """Grouped text report. Ends with a newline.

    The summary is ``errors=N warnings=M``. An empty report says
    ``no findings`` before that line. Messages are single-spaced so each
    finding stays one line.
    """
    lines: list[str] = []
    current: str | None = None
    if not report.findings:
        lines.append("no findings")
    for finding in report.findings:
        if finding.file != current:
            if current is not None:
                lines.append("")
            current = finding.file
            # Surrogates in paths / case ids (e.g. from a \\ud800 JSON escape)
            # must be stripped so printing to UTF-8 stdout cannot raise.
            lines.append(f"file: {strip_unpaired_surrogates(finding.file)}")
        case = strip_unpaired_surrogates(finding.case) if finding.case else "-"
        message = " ".join(strip_unpaired_surrogates(finding.message).split())
        lines.append(f"  case: {case}  severity: {finding.severity}  message: {message}")
    lines.append(f"errors={report.error_count} warnings={report.warning_count}")
    return "\n".join(lines) + "\n"


def audit_to_dict(report: AuditReport) -> dict[str, object]:
    """JSON-ready mapping. ``errors`` and ``warnings`` are counts."""
    return {
        "findings": [
            {
                "file": finding.file,
                "case": finding.case,
                "severity": finding.severity,
                "check": finding.check,
                "message": finding.message,
            }
            for finding in report.findings
        ],
        "errors": report.error_count,
        "warnings": report.warning_count,
    }


def _audit_path(path: str) -> list[Finding]:
    try:
        suite = load_suite(path)
    except SuiteError as exc:
        return [_load_finding(path, exc)]
    except OSError as exc:
        return [
            Finding(
                file=path,
                case="",
                severity="error",
                check="load",
                message=f"{exc.__class__.__name__}: {exc}",
            )
        ]
    return _audit_suite(path, suite.cases)


def _load_finding(path: str, exc: SuiteError) -> Finding:
    text = str(exc)
    match = _DUPLICATE.match(text)
    if match is not None:
        case_id = match.group("case_id")
        return Finding(
            file=path,
            case=case_id,
            severity="error",
            check="duplicate_case_id",
            message=f"duplicate case id {case_id!r}",
        )
    return Finding(file=path, case="", severity="error", check="load", message=text)


def _audit_suite(path: str, cases: Sequence[Case]) -> list[Finding]:
    findings: list[Finding] = []
    first_input: dict[str, str] = {}
    for case in cases:
        if not case.assertions:
            findings.append(
                Finding(
                    file=path,
                    case=case.id,
                    severity="error",
                    check="no_assertions",
                    message="case has no assertions; score is 0 and passed is false",
                )
            )
        else:
            findings.extend(_assertion_findings(path, case))
        previous = first_input.get(case.input)
        if previous is None:
            first_input[case.input] = case.id
        else:
            findings.append(
                Finding(
                    file=path,
                    case=case.id,
                    severity="warning",
                    check="identical_input",
                    message=f"identical input text to case {previous!r}",
                )
            )
    return findings


def _assertion_findings(path: str, case: Case) -> list[Finding]:
    findings: list[Finding] = []
    for assertion in case.assertions:
        if assertion.kind not in SCORERS:
            known = ", ".join(sorted(SCORERS))
            findings.append(
                Finding(
                    file=path,
                    case=case.id,
                    severity="error",
                    check="unknown_scorer",
                    message=f"unknown scorer kind {assertion.kind!r}. Known scorers: {known}.",
                )
            )
            continue
        if assertion.kind in REFERENCE_KINDS and _reference_blank(case, assertion):
            findings.append(
                Finding(
                    file=path,
                    case=case.id,
                    severity="warning",
                    check="empty_reference",
                    message=_empty_reference_message(assertion),
                )
            )
    return findings


def _empty_reference_message(assertion: AssertionSpec) -> str:
    if isinstance(assertion.value, str) and assertion.value.strip() == "":
        return f"{assertion.kind} value is empty"
    return f"{assertion.kind} has an empty reference and no value"


def _reference_blank(case: Case, assertion: AssertionSpec) -> bool:
    """True when the text ``_target`` would score is missing or whitespace.

    A ``value`` that is not ``None`` wins, including a blank string, so a
    blank value is empty even if the reference is filled in. ``None`` uses
    the reference.
    """
    if assertion.value is None:
        reference = case.reference
        return reference is None or reference.strip() == ""
    if isinstance(assertion.value, str):
        return assertion.value.strip() == ""
    return False


# Below this size the 2PL fit and the item correlations are not stable.
MIN_CONFIGS = 4
MIN_CASES = 5
# Fitted discrimination at or below this is "near zero". The 2PL fit clips
# ``a`` to [0.05, 5] and pulls it toward 1, so a non-separating item sits on
# the floor. This is not the IRT ``low_discrimination`` cut of 0.5.
NEAR_ZERO_DISCRIMINATION = 0.15
# Pearson point-biserial at or below this is "strongly negative".
STRONG_NEGATIVE_POINT_BISERIAL = -0.2
# Pass/fail vectors with at most this many disagreements are one redundant pair.
MAX_REDUNDANT_DISAGREEMENTS = 1
# The audit calls ``fit_runs`` with this seed. It is the project IRT default.
AUDIT_SEED = 13
AUDIT_RESPONSE = "passed"

SUITE_AUDIT_LIMIT = (
    "Flags are advisory — the command exits 0 with a report and never changes the gate."
)

_NOTE_SENTENCES: tuple[str, ...] = (
    "Suite audit reads comparison reports from compare --report-json and pools each config.",
    "A config is the baseline run or the candidate run, in file order, baseline first.",
    "The same config name with the same case scores and pass bits is one config.",
    "A later copy is ignored and counted.",
    "The same name with a different score or pass bit raises ValueError.",
    "Case ids must match across configs. Missing cases are not imputed.",
    "The pass bit is the stored passed field.",
    "A config's total score is math.fsum of its case scores.",
    "The point-biserial of a case is the Pearson correlation of its pass vector with the totals.",
    "The pass vector is 1 when the case passed and 0 when it failed.",
    "Pearson divides the centered sum of products by the square root of the two sums of squares.",
    "The correlation is undefined when either sum of squares is 0. Undefined is not a misfit.",
    (
        "A misfit is a defined point-biserial at or below "
        f"{STRONG_NEGATIVE_POINT_BISERIAL}, which is strongly negative."
    ),
    "The case score is inside the total, so that part-whole effect pulls the correlation up.",
    (
        "Discrimination is the project's 2PL fit on pass/fail "
        f"(fit_runs response {AUDIT_RESPONSE}, seed {AUDIT_SEED}), not on graded scores."
    ),
    (
        "A non-discriminating case has fitted discrimination at or below "
        f"{NEAR_ZERO_DISCRIMINATION}."
    ),
    "The fit clips discrimination to [0.05, 5] and pulls it toward 1.",
    "A non-separating item sits near 0.05.",
    "This cut is not the IRT low_discrimination flag, which uses 0.5.",
    "A redundant pair is two cases whose pass bits match, or differ on at most one config.",
    "They cost runtime and add no signal.",
    "A dead case passed on every config, or failed on every config, so it has no variance.",
    "That is distinct from non-discrimination.",
    "Non-discrimination is a fitted value and can occur when the case still varies.",
    (
        f"Fewer than {MIN_CONFIGS} configs or fewer than {MIN_CASES} cases raises ValueError, "
        "as do non-finite scores and unpairable inputs."
    ),
    SUITE_AUDIT_LIMIT,
    "A flagged case is a property of these configs.",
    "A case that is dead here can still separate a config that was not in the files.",
    "The fit is not a standard error, and a flag is not by itself a reason to delete a case.",
    "Text rounds fitted values to 6 decimals. JSON keeps the stored floats.",
)

SUITE_AUDIT_NOTE = " ".join(_NOTE_SENTENCES)

FlagName = Literal["non_discriminating", "misfit", "redundant", "dead"]


@dataclass(frozen=True)
class AuditedCase:
    """One case after the fit. ``passes`` is ``1``/``0`` in config order."""

    case_id: str
    discrimination: float
    difficulty: float
    point_biserial: float | None
    n_pass: int
    passes: str
    flags: tuple[FlagName, ...]


@dataclass(frozen=True)
class RedundantPair:
    """Two cases whose pass bits differ in at most one config.

    ``case_a`` is the earlier case in the first config's order.
    """

    case_a: str
    case_b: str
    disagreements: int


@dataclass(frozen=True)
class IgnoredConfig:
    """A repeated config name whose scores and pass bits matched the copy that was kept."""

    config: str
    path: str
    kept_path: str


@dataclass(frozen=True)
class ConfigFit:
    """One config's total score and its fitted ability. ``rank`` 1 is the highest theta."""

    config: str
    total_score: float
    theta: float
    rank: int


@dataclass(frozen=True)
class SuiteAuditReport:
    """Advisory suite-quality report. It does not carry a gate verdict."""

    comparison_paths: tuple[str, ...]
    configs: tuple[str, ...]
    cases: tuple[AuditedCase, ...]
    redundant_pairs: tuple[RedundantPair, ...]
    config_fits: tuple[ConfigFit, ...]
    duplicates_ignored: tuple[IgnoredConfig, ...]
    seed: int
    iterations: int
    note: str = SUITE_AUDIT_NOTE

    @property
    def n_comparisons(self) -> int:
        return len(self.comparison_paths)

    @property
    def n_configs(self) -> int:
        return len(self.configs)

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    def cases_with(self, flag: FlagName) -> tuple[AuditedCase, ...]:
        return tuple(case for case in self.cases if flag in case.flags)


def is_comparison_report(path: str | Path) -> bool:
    """True when ``path`` looks like ``compare --report-json``.

    A read error, invalid JSON, or any other shape is False. The suite
    linter then owns the path. This does not fully validate the runs.
    """
    payload = _read_json_value(path)
    if not isinstance(payload, dict):
        return False
    return _is_run_side(payload.get("baseline")) and _is_run_side(payload.get("candidate"))


def audit_comparison_files(paths: Sequence[str | Path]) -> SuiteAuditReport:
    """Load comparison reports and audit the pooled configs.

    The loader is :func:`promptregress.risk.load_comparison`. A file that
    is not a comparison, a non-finite score, or unpairable cases raises
    ``ValueError``. Fewer than :data:`MIN_CONFIGS` distinct configs or
    :data:`MIN_CASES` cases raises ``ValueError``.
    """
    if isinstance(paths, (str, bytes)):
        raise ValueError("paths must be a sequence of comparison files")
    listed = [str(path) for path in paths]
    if not listed:
        raise ValueError("suite audit needs at least one comparison report")
    entries: list[tuple[RunResult, str]] = []
    for path in listed:
        comparison = load_comparison(path)
        entries.append((comparison.baseline, path))
        entries.append((comparison.candidate, path))
    return _audit(entries, comparison_paths=tuple(listed))


def audit_runs(runs: Sequence[RunResult]) -> SuiteAuditReport:
    """Audit runs that are already loaded. One run is one config.

    The same rules as :func:`audit_comparison_files` apply. Source paths
    are empty because the caller did not pass files.
    """
    if isinstance(runs, (str, bytes)):
        raise ValueError("runs must be a sequence of run results")
    entries = [(run, "") for run in runs]
    return _audit(entries, comparison_paths=())


def format_suite_audit(report: SuiteAuditReport) -> str:
    """Text report. Ends with a newline.

    Fitted numbers use 6 decimals. The thresholds are printed in short
    form, the same values as the named constants.
    """
    lines = [
        "suite audit",
        f"n_comparisons={report.n_comparisons}",
        f"n_configs={report.n_configs}",
        f"n_cases={report.n_cases}",
        f"configs={','.join(_cell(name) for name in report.configs)}",
        f"seed={report.seed}",
        f"iterations={report.iterations}",
        f"response={AUDIT_RESPONSE}",
        f"near_zero_discrimination={_plain(NEAR_ZERO_DISCRIMINATION)}",
        f"strong_negative_point_biserial={_plain(STRONG_NEGATIVE_POINT_BISERIAL)}",
        f"min_configs={MIN_CONFIGS}",
        f"min_cases={MIN_CASES}",
        f"max_redundant_disagreements={MAX_REDUNDANT_DISAGREEMENTS}",
        f"non_discriminating={len(report.cases_with('non_discriminating'))}",
        f"misfit={len(report.cases_with('misfit'))}",
        f"redundant_pairs={len(report.redundant_pairs)}",
        f"dead={len(report.cases_with('dead'))}",
        f"duplicates_ignored={len(report.duplicates_ignored)}",
        f"note={_cell(report.note)}",
        "",
        "comparisons:",
    ]
    if report.comparison_paths:
        lines.extend(f"- {_cell(path)}" for path in report.comparison_paths)
    else:
        lines.append("- none")
    lines.extend(["", "non_discriminating:"])
    _extend_cases(
        lines,
        report.cases_with("non_discriminating"),
        lambda case: f"- case={_cell(case.case_id)} discrimination={case.discrimination:.6f}",
    )
    lines.extend(["", "misfit:"])
    _extend_cases(
        lines,
        report.cases_with("misfit"),
        lambda case: (
            f"- case={_cell(case.case_id)} point_biserial={_optional(case.point_biserial)}"
        ),
    )
    lines.extend(["", "redundant:"])
    if report.redundant_pairs:
        for pair in report.redundant_pairs:
            lines.append(
                f"- case_a={_cell(pair.case_a)} case_b={_cell(pair.case_b)} "
                f"disagreements={pair.disagreements}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "dead:"])
    _extend_cases(
        lines,
        report.cases_with("dead"),
        lambda case: f"- case={_cell(case.case_id)} passed={'true' if case.n_pass else 'false'}",
    )
    lines.extend(
        [
            "",
            "| case | passes | a | b | point_biserial | n_pass | flags |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for case in report.cases:
        flags = ",".join(case.flags) if case.flags else "-"
        lines.append(
            "| "
            f"{_table(case.case_id)} | {case.passes} | {case.discrimination:.6f} | "
            f"{case.difficulty:.6f} | {_optional(case.point_biserial)} | "
            f"{case.n_pass} | {flags} |"
        )
    lines.extend(
        [
            "",
            "| rank | config | theta | total_score |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    for fit in sorted(report.config_fits, key=lambda item: (item.rank, item.config)):
        lines.append(
            f"| {fit.rank} | {_table(fit.config)} | {fit.theta:.6f} | {fit.total_score:.6f} |"
        )
    lines.extend(["", "duplicates_ignored:"])
    if report.duplicates_ignored:
        for item in report.duplicates_ignored:
            lines.append(
                f"- config={_cell(item.config)} path={_cell(item.path)} "
                f"kept={_cell(item.kept_path)}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def suite_audit_to_dict(report: SuiteAuditReport) -> dict[str, object]:
    """JSON-ready suite audit. ``point_biserial`` is null when it is undefined."""
    return {
        "n_comparisons": report.n_comparisons,
        "n_configs": report.n_configs,
        "n_cases": report.n_cases,
        "configs": list(report.configs),
        "comparison_paths": list(report.comparison_paths),
        "seed": report.seed,
        "iterations": report.iterations,
        "response": AUDIT_RESPONSE,
        "near_zero_discrimination": NEAR_ZERO_DISCRIMINATION,
        "strong_negative_point_biserial": STRONG_NEGATIVE_POINT_BISERIAL,
        "min_configs": MIN_CONFIGS,
        "min_cases": MIN_CASES,
        "max_redundant_disagreements": MAX_REDUNDANT_DISAGREEMENTS,
        "duplicates_ignored": [
            {"config": item.config, "path": item.path, "kept_path": item.kept_path}
            for item in report.duplicates_ignored
        ],
        "non_discriminating": [
            {"case_id": case.case_id, "discrimination": case.discrimination}
            for case in report.cases_with("non_discriminating")
        ],
        "misfit": [
            {"case_id": case.case_id, "point_biserial": case.point_biserial}
            for case in report.cases_with("misfit")
        ],
        "redundant_pairs": [
            {
                "case_a": pair.case_a,
                "case_b": pair.case_b,
                "disagreements": pair.disagreements,
            }
            for pair in report.redundant_pairs
        ],
        "dead": [
            {"case_id": case.case_id, "passed": case.n_pass > 0}
            for case in report.cases_with("dead")
        ],
        "cases": [_case_to_dict(case) for case in report.cases],
        "abilities": [
            {
                "config": fit.config,
                "theta": fit.theta,
                "rank": fit.rank,
                "total_score": fit.total_score,
            }
            for fit in sorted(report.config_fits, key=lambda item: (item.rank, item.config))
        ],
        "note": report.note,
    }


def _case_to_dict(case: AuditedCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "discrimination": case.discrimination,
        "difficulty": case.difficulty,
        "point_biserial": case.point_biserial,
        "n_pass": case.n_pass,
        "passes": case.passes,
        "flags": list(case.flags),
    }


def _audit(
    entries: Sequence[tuple[RunResult, str]],
    *,
    comparison_paths: tuple[str, ...],
) -> SuiteAuditReport:
    if not entries:
        raise ValueError(
            "suite audit needs at least "
            f"{MIN_CONFIGS} configs and {MIN_CASES} cases for a stable 2PL fit; "
            "got 0 configs and 0 cases"
        )
    kept, ignored = _dedupe_configs(entries)
    _reject_unpairable(kept)
    n_configs = len(kept)
    n_cases = len(kept[0].results) if kept else 0
    if n_configs < MIN_CONFIGS or n_cases < MIN_CASES:
        raise ValueError(_size_message(n_configs, n_cases, len(ignored)))
    aligned = _align_runs(kept)
    totals = [_config_total(run) for run in aligned]
    outcome = fit_runs(aligned, response=AUDIT_RESPONSE, seed=AUDIT_SEED)
    fitted_names = {item.model for item in outcome.abilities}
    if fitted_names != {run.config_name for run in aligned}:
        raise ValueError("IRT fit renamed a config. Suite audit refuses that matrix.")
    by_case = {item.case_id: item for item in outcome.cases}
    by_model = {item.model: item for item in outcome.abilities}
    pass_columns = [tuple(1 if result.passed else 0 for result in run.results) for run in aligned]
    cases: list[AuditedCase] = []
    pairs: list[RedundantPair] = []
    redundant_ids: set[str] = set()
    case_ids = [result.case_id for result in aligned[0].results]
    for left in range(len(case_ids)):
        for right in range(left + 1, len(case_ids)):
            disagreements = sum(column[left] != column[right] for column in pass_columns)
            if disagreements <= MAX_REDUNDANT_DISAGREEMENTS:
                pairs.append(RedundantPair(case_ids[left], case_ids[right], disagreements))
                redundant_ids.add(case_ids[left])
                redundant_ids.add(case_ids[right])
    for index, case_id in enumerate(case_ids):
        item = by_case[case_id]
        if not math.isfinite(item.a) or not math.isfinite(item.b):
            raise ValueError(f"case {case_id!r} has a non-finite IRT parameter")
        passes = "".join(str(column[index]) for column in pass_columns)
        column = [float(bits[index]) for bits in pass_columns]
        correlation = _pearson(column, totals)
        if correlation is not None and not math.isfinite(correlation):
            raise ValueError(f"case {case_id!r} has a non-finite point-biserial")
        n_pass = sum(bits[index] for bits in pass_columns)
        flags: list[FlagName] = []
        if item.a <= NEAR_ZERO_DISCRIMINATION:
            flags.append("non_discriminating")
        if correlation is not None and correlation <= STRONG_NEGATIVE_POINT_BISERIAL:
            flags.append("misfit")
        if case_id in redundant_ids:
            flags.append("redundant")
        if n_pass == 0 or n_pass == n_configs:
            flags.append("dead")
        cases.append(
            AuditedCase(
                case_id=case_id,
                discrimination=item.a,
                difficulty=item.b,
                point_biserial=correlation,
                n_pass=n_pass,
                passes=passes,
                flags=tuple(flags),
            )
        )
    config_fits: list[ConfigFit] = []
    for run, total in zip(aligned, totals, strict=True):
        ability = by_model[run.config_name]
        if not math.isfinite(ability.theta):
            raise ValueError(f"config {run.config_name!r} has a non-finite ability")
        config_fits.append(
            ConfigFit(
                config=run.config_name,
                total_score=total,
                theta=ability.theta,
                rank=ability.rank,
            )
        )
    return SuiteAuditReport(
        comparison_paths=comparison_paths,
        configs=tuple(run.config_name for run in aligned),
        cases=tuple(cases),
        redundant_pairs=tuple(pairs),
        config_fits=tuple(config_fits),
        duplicates_ignored=tuple(ignored),
        seed=outcome.seed,
        iterations=outcome.iterations,
    )


def _dedupe_configs(
    entries: Sequence[tuple[RunResult, str]],
) -> tuple[list[RunResult], list[IgnoredConfig]]:
    kept: dict[str, tuple[RunResult, str]] = {}
    order: list[RunResult] = []
    ignored: list[IgnoredConfig] = []
    for index, (run, path) in enumerate(entries):
        if not isinstance(run, RunResult):
            raise ValueError("runs must be RunResult values")
        signature = _signature(run)
        label = path if path else f"run {index}"
        previous = kept.get(run.config_name)
        if previous is None:
            kept[run.config_name] = (run, label)
            order.append(run)
            continue
        previous_run, previous_label = previous
        if _signature(previous_run) != signature:
            raise ValueError(
                f"config {run.config_name!r} in {label} does not match {previous_label}: "
                "the same config name has different case scores or pass bits"
            )
        ignored.append(IgnoredConfig(config=run.config_name, path=label, kept_path=previous_label))
    return order, ignored


def _signature(run: RunResult) -> tuple[tuple[str, float, bool], ...]:
    indexed = _index_cases(run)
    return tuple(
        (case_id, indexed[case_id].score, indexed[case_id].passed) for case_id in sorted(indexed)
    )


def _index_cases(run: RunResult) -> dict[str, CaseResult]:
    if not isinstance(run.config_name, str):
        raise ValueError("config_name must be a string")
    found: dict[str, CaseResult] = {}
    for result in run.results:
        if not isinstance(result.case_id, str):
            raise ValueError(f"config {run.config_name!r} has a case id that is not a string")
        if result.case_id in found:
            raise ValueError(f"config {run.config_name!r} repeats case id {result.case_id!r}")
        if not isinstance(result.passed, bool):
            raise ValueError(
                f"config {run.config_name!r} case {result.case_id!r} passed must be a boolean"
            )
        score = result.score
        message = (
            f"config {run.config_name!r} case {result.case_id!r} score must be a finite number"
        )
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(message)
        try:
            number = float(score)
        except OverflowError as exc:
            raise ValueError(message) from exc
        if not math.isfinite(number):
            raise ValueError(message)
        found[result.case_id] = result
    return found


def _reject_unpairable(runs: Sequence[RunResult]) -> None:
    if not runs:
        return
    anchor = runs[0]
    expected = {result.case_id for result in anchor.results}
    for run in runs[1:]:
        found = {result.case_id for result in run.results}
        if found == expected:
            continue
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ValueError(
            f"config {run.config_name!r} case ids do not match {anchor.config_name!r}. "
            f"Missing: {missing}. Extra: {extra}. "
            "Suite audit does not impute missing cases."
        )


def _align_runs(runs: Sequence[RunResult]) -> list[RunResult]:
    case_ids = [result.case_id for result in runs[0].results]
    aligned: list[RunResult] = []
    for run in runs:
        by_id = _index_cases(run)
        aligned.append(
            RunResult(
                suite_name=run.suite_name,
                config_name=run.config_name,
                results=[by_id[case_id] for case_id in case_ids],
                cost=run.cost,
            )
        )
    return aligned


def _size_message(n_configs: int, n_cases: int, n_ignored: int) -> str:
    message = (
        "suite audit needs at least "
        f"{MIN_CONFIGS} configs and {MIN_CASES} cases for a stable 2PL fit; "
        f"got {n_configs} configs and {n_cases} cases"
    )
    if n_ignored == 1:
        return message + ". 1 duplicate config matched an earlier copy and was not counted"
    if n_ignored > 1:
        return (
            message
            + f". {n_ignored} duplicate configs matched an earlier copy and were not counted"
        )
    return message


def _config_total(run: RunResult) -> float:
    """Sum of case scores. The contract names ``math.fsum``.

    ``fsum`` raises ``OverflowError`` when a partial sum leaves the float
    range, including a series whose exact sum is finite
    (``1e308 + 1e308 + -1e308`` is ``1e308``). Equal scores use a product.
    Any other series scales by the largest magnitude and sums the unit
    terms. A sum that is not a finite float raises ``ValueError``.
    """
    scores = [result.score for result in run.results]
    try:
        total = math.fsum(scores)
    except OverflowError:
        total = _wide_sum(scores)
    if not math.isfinite(total):
        raise ValueError(f"config {run.config_name!r} total score is not a finite number")
    return total


def _wide_sum(values: Sequence[float]) -> float:
    """Sum when ``math.fsum`` raised ``OverflowError``. Non-finite returns inf."""
    if not values:
        return 0.0
    first = values[0]
    if all(value == first for value in values):
        try:
            return float(first) * len(values)
        except OverflowError:
            return math.inf
    try:
        peak = max(abs(float(value)) for value in values)
    except OverflowError:
        return math.inf
    if peak == 0.0 or not math.isfinite(peak):
        return math.inf
    try:
        scaled = math.fsum(float(value) / peak for value in values)
        return scaled * peak
    except OverflowError:
        return math.inf


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Pearson correlation. ``None`` when either side has no variance.

    The value is the centered sum of products divided by the square root of
    the two sums of squares. Squares near ``1e200`` overflow, a product of
    two finite sums of squares can overflow, and squares near ``1e-200``
    underflow to 0. Any of those collapses a correlation of -1 to 0 or to
    "undefined", so a misfit is missed. The retry centers and scales each
    side before squaring. Scaling does not change the correlation. A retry
    that still has no variance is undefined, not a misfit.
    """
    if len(left) != len(right) or len(left) < 2:
        return None
    try:
        sum_left, sum_right, product = _pearson_parts(left, right)
    except OverflowError:
        return _pearson_scaled(left, right)
    if not (math.isfinite(sum_left) and math.isfinite(sum_right) and math.isfinite(product)):
        return _pearson_scaled(left, right)
    if sum_left == 0.0 or sum_right == 0.0:
        return _pearson_scaled(left, right)
    denominator = math.sqrt(sum_left * sum_right)
    if not math.isfinite(denominator) or denominator == 0.0:
        return _pearson_scaled(left, right)
    value = product / denominator
    if not math.isfinite(value):
        return _pearson_scaled(left, right)
    return value


def _pearson_parts(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    mean_left = math.fsum(left) / len(left)
    mean_right = math.fsum(right) / len(right)
    delta_left = [value - mean_left for value in left]
    delta_right = [value - mean_right for value in right]
    sum_left = math.fsum(value * value for value in delta_left)
    sum_right = math.fsum(value * value for value in delta_right)
    product = math.fsum(a * b for a, b in zip(delta_left, delta_right, strict=True))
    return sum_left, sum_right, product


def _pearson_scaled(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_unit = _unit_deviations(left)
    right_unit = _unit_deviations(right)
    if left_unit is None or right_unit is None:
        return None
    sum_left = math.fsum(value * value for value in left_unit)
    sum_right = math.fsum(value * value for value in right_unit)
    if sum_left == 0.0 or sum_right == 0.0:
        return None
    product = math.fsum(a * b for a, b in zip(left_unit, right_unit, strict=True))
    return product / math.sqrt(sum_left * sum_right)


def _unit_deviations(values: Sequence[float]) -> list[float] | None:
    """Centered values scaled so the largest absolute deviation is 1."""
    deviations = _centered(values)
    if deviations is None:
        return None
    peak = max(abs(value) for value in deviations)
    if peak == 0.0 or not math.isfinite(peak):
        return None
    return [value / peak for value in deviations]


def _centered(values: Sequence[float]) -> list[float] | None:
    """Deviations from the mean. ``None`` when every value is equal.

    Shift by the first value so a tiny gap between large values stays
    exact. If that shift is not finite, scale by the largest magnitude
    first. The two steps are an affine transform with positive scale.
    """
    ref = float(values[0])
    shifted = [float(value) - ref for value in values]
    if all(math.isfinite(item) for item in shifted):
        center = _mean(shifted)
        if math.isfinite(center):
            deviations = [item - center for item in shifted]
            if all(math.isfinite(item) for item in deviations):
                if all(item == 0.0 for item in deviations):
                    return None
                return deviations
    try:
        peak = max(abs(float(value)) for value in values)
    except OverflowError:
        return None
    if peak == 0.0 or not math.isfinite(peak):
        return None
    scaled = [float(value) / peak for value in values]
    center = math.fsum(scaled) / len(scaled)
    deviations = [item - center for item in scaled]
    if not all(math.isfinite(item) for item in deviations):
        return None
    if all(item == 0.0 for item in deviations):
        return None
    return deviations


def _mean(values: Sequence[float]) -> float:
    """``math.fsum(values) / len(values)``, dividing first if ``fsum`` overflows."""
    count = len(values)
    try:
        return math.fsum(values) / count
    except OverflowError:
        first = values[0]
        if all(value == first for value in values):
            return float(first)
        try:
            return math.fsum(value / count for value in values)
        except OverflowError:
            return math.inf


def _read_json_value(path: str | Path) -> object | None:
    try:
        text = strip_utf8_bom_text(Path(path).read_text(encoding="utf-8-sig"))
        value: object = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value


def _is_run_side(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if not isinstance(value.get("config_name"), str):
        return False
    results = value.get("results")
    if not isinstance(results, list):
        return False
    if results and not isinstance(results[0], dict):
        return False
    return True


def _extend_cases(
    lines: list[str],
    cases: Sequence[AuditedCase],
    render: Callable[[AuditedCase], str],
) -> None:
    if not cases:
        lines.append("- none")
        return
    for case in cases:
        lines.append(render(case))


def _optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def _plain(value: float) -> str:
    return format(value, "g")


def _cell(value: str) -> str:
    """One output field. Newlines and unpaired surrogates cannot break the line."""
    return " ".join(strip_unpaired_surrogates(value).split())


def _table(value: str) -> str:
    return _cell(value).replace("|", "/")
