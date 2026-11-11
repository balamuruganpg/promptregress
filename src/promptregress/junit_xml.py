"""JUnit XML for one run or a paired comparison.

Mapping
-------

The document is a ``<testsuites>`` root. ``run`` writes one ``<testsuite>``.
``compare`` writes two, baseline first and candidate second. ``name`` is that
run's config label (``--config-name``, ``--baseline-config-name``, or
``--candidate-name``), not the words "baseline" and "candidate".

Each suite carries:

- ``tests``: number of cases
- ``failures``: cases whose ``passed`` bit is false
- ``errors``: always ``0``. A provider exception is a failed case, so it is
  a ``<failure>``, not a JUnit ``<error>``
- ``time``: ``total_latency_ms / 1000`` (suite latency in seconds)

Each case is a ``<testcase>``:

- ``classname``: the eval suite name (``RunResult.suite_name``)
- ``name``: the case id
- ``time``: that case's ``latency_ms / 1000``

JUnit failure means the case failed (``CaseResult.passed`` is false). The
gate is a separate decision and still owns the process exit code. A
candidate case that still passes, but whose score dropped versus the
baseline, is not a failure. "Dropped" is the paired outcome ``loss``: the
candidate score is lower by more than the tie epsilon (``1e-12``). That
case gets a ``<system-out>`` note and nothing else::

    regressed: baseline 1.000000 -> candidate 0.709297

Scores in the note are six decimal places. The note is only attached to
the candidate suite. A case that failed keeps the ``<failure>`` even when
its score also dropped; it does not get the regression note.

The failure ``message`` attribute, and the same string as the element
text, names the case score and the first failing score detail::

    score=0.500000; contains=0.000000

Score details are ``CaseResult.scores``: one subscore per scorer kind, in
the order kinds were first seen on the case. A kind fails when it misses
its pass rule. Exact kinds (``EXACT_KINDS``, including plugins registered
with ``exact=True``) fail below ``0.999``. Every other kind fails below
``0.5``. Subscores are the unweighted mean of that kind, so two assertions
of one kind share a detail. When no kind fails that rule, the message uses
``CaseResult.error`` if it is set (a provider exception or a judge note),
and otherwise ``no assertion detail``.

A failed case with a non-empty output also gets ``<system-out>`` set to
that output, so the model text is in the file. Passing cases that did not
regress have no ``<system-out>``. The regression note replaces the output
on a regressed-but-passing candidate case.

Attribute and text values are written with ``xml.etree.ElementTree``, which
escapes them. Characters XML 1.0 forbids (most C0 controls, U+FFFE, and
U+FFFF) are removed before that, so a completion cannot crash the export.
Removing them can glue a case id to a different id; those characters are
not meaningful in JUnit names.
"""

import math
import re
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

from promptregress.models import CaseResult, Comparison, RunResult
from promptregress.scorers import EXACT_KINDS, EXACT_PASS_FLOOR, GRADED_MEAN_FLOOR

# XML 1.0 Char production excludes these. ElementTree rejects them.
_ILLEGAL_XML = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\uFFFE\uFFFF]")


def write_run_junit(path: str | Path, result: RunResult) -> None:
    """Write one ``<testsuite>`` for a single run."""
    write_junit_xml(path, [result])


def write_compare_junit(path: str | Path, comparison: Comparison) -> None:
    """Write baseline and candidate suites, with regression notes on the candidate.

    A note is added only when the paired outcome is a loss and the candidate
    case still passed. See the module docstring for the exact text.
    """
    notes = {
        row.case_id: (
            f"regressed: baseline {_format_score(row.baseline_score)} -> "
            f"candidate {_format_score(row.candidate_score)}"
        )
        for row in comparison.per_case
        if row.outcome == "loss" and row.candidate_passed
    }
    write_junit_xml(
        path,
        [comparison.baseline, comparison.candidate],
        regression_notes=notes,
        candidate=comparison.candidate,
    )


def write_junit_xml(
    path: str | Path,
    runs: Sequence[RunResult],
    *,
    regression_notes: Mapping[str, str] | None = None,
    candidate: RunResult | None = None,
) -> None:
    """Write ``runs`` as JUnit XML, creating parent directories.

    ``regression_notes`` maps a case id to a ``<system-out>`` string. The
    notes are applied only to the ``candidate`` object, compared by identity,
    so a baseline case with the same id is left alone.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = render_junit_xml(runs, regression_notes=regression_notes, candidate=candidate)
    target.write_text(text, encoding="utf-8")


def render_junit_xml(
    runs: Sequence[RunResult],
    *,
    regression_notes: Mapping[str, str] | None = None,
    candidate: RunResult | None = None,
) -> str:
    """Return the JUnit document described in the module docstring."""
    notes = {} if regression_notes is None else regression_notes
    root = Element("testsuites")
    for run in runs:
        applied = notes if run is candidate else {}
        root.append(_suite_element(run, applied))
    return _serialize(root)


def failure_message(result: CaseResult) -> str:
    """Failure text for a case that did not pass.

    The caller decides that the case failed. This function only names the
    score and the first failing detail. See the module docstring.
    """
    detail = _first_failing_detail(result)
    score = _format_score(result.score)
    if detail is not None:
        return f"score={score}; {detail}"
    if result.error:
        return f"score={score}; {result.error}"
    return f"score={score}; no assertion detail"


def _first_failing_detail(result: CaseResult) -> str | None:
    for kind, value in result.scores.items():
        if _kind_failed(kind, value):
            return f"{kind}={_format_score(value)}"
    return None


def _kind_failed(kind: str, value: float) -> bool:
    if kind in EXACT_KINDS:
        return value < EXACT_PASS_FLOOR
    return value < GRADED_MEAN_FLOOR


def _format_score(value: float) -> str:
    if not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    return f"{value:.6f}"


def _seconds(latency_ms: float) -> str:
    seconds = latency_ms / 1000.0
    if not math.isfinite(seconds):
        return "0.000000"
    return f"{seconds:.6f}"


def _xml_text(value: object) -> str:
    return _ILLEGAL_XML.sub("", str(value))


def _suite_element(run: RunResult, notes: Mapping[str, str]) -> Element:
    failures = sum(1 for result in run.results if not result.passed)
    suite = Element(
        "testsuite",
        {
            "name": _xml_text(run.config_name),
            "tests": str(len(run.results)),
            "failures": str(failures),
            "errors": "0",
            "time": _seconds(run.total_latency_ms),
        },
    )
    for result in run.results:
        suite.append(_case_element(run, result, notes.get(result.case_id)))
    return suite


def _case_element(run: RunResult, result: CaseResult, note: str | None) -> Element:
    case = Element(
        "testcase",
        {
            "classname": _xml_text(run.suite_name),
            "name": _xml_text(result.case_id),
            "time": _seconds(result.latency_ms),
        },
    )
    if not result.passed:
        message = _xml_text(failure_message(result))
        failure = SubElement(case, "failure", {"message": message})
        failure.text = message
        output = _xml_text(result.output)
        if output:
            system_out = SubElement(case, "system-out")
            system_out.text = output
    elif note:
        system_out = SubElement(case, "system-out")
        system_out.text = _xml_text(note)
    return case


def _serialize(root: Element) -> str:
    indent(root, space="  ")
    buffer = BytesIO()
    ElementTree(root).write(buffer, encoding="utf-8", xml_declaration=True)
    text = buffer.getvalue().decode("utf-8")
    if not text.endswith("\n"):
        text += "\n"
    return text
