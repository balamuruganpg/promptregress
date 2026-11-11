"""Lint suite files before they silently inflate a pass rate.

``promptregress audit`` loads each path with :func:`promptregress.loading.load_suite`.
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

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from promptregress.loading import SuiteError, load_suite
from promptregress.models import AssertionSpec, Case
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
            lines.append(f"file: {finding.file}")
        case = finding.case if finding.case else "-"
        message = " ".join(finding.message.split())
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
