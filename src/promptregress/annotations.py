"""GitHub Actions workflow commands for failed cases and gate summaries.

``run`` and ``compare`` write these to stderr when ``--github-annotations``
is set, and also when ``GITHUB_ACTIONS=true`` (the variable Actions runners
set). One command per failed case::

    ::error file=<suite file>,line=<case line if known>::<id>: <assertion> (score <score>)

``line`` is omitted when the loader did not record a source line. Property
values escape ``%``, CR, LF, ``:``, and ``,``. The message escapes ``%``,
CR, and LF. ``:`` and ``,`` stay in the message.

A gate failure with no failed case (drift, or a cost or latency budget)
writes one summary line, ``::error::<reason>``. Case failures are not
repeated as that summary.
"""

import os
import sys
from collections.abc import Mapping, Sequence

from promptregress.gate import GateVerdict
from promptregress.junit_xml import failing_assertion, format_score, strip_unpaired_surrogates
from promptregress.models import CaseResult


def github_annotations_enabled(flag: bool, env: Mapping[str, str] | None = None) -> bool:
    """True when the flag is set or ``GITHUB_ACTIONS`` is the string ``true``."""
    if flag:
        return True
    values = os.environ if env is None else env
    return values.get("GITHUB_ACTIONS") == "true"


def escape_property(value: str) -> str:
    """Escape a workflow-command property value.

    Comma is ``%2C`` (the GitHub Actions runner / actions toolkit mapping).
    Percent is replaced first so the other escapes are not encoded a second
    time.
    """
    return (
        value.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
        .replace(":", "%3A")
        .replace(",", "%2C")
    )


def escape_message(value: str) -> str:
    """Escape workflow-command message data. ``:`` and ``,`` are left as written."""
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def case_annotation(result: CaseResult, suite_file: str) -> str:
    """One ``::error`` command for a case that did not pass."""
    case_id = strip_unpaired_surrogates(result.case_id)
    assertion = strip_unpaired_surrogates(failing_assertion(result))
    message = f"{case_id}: {assertion} (score {format_score(result.score)})"
    file_prop = escape_property(strip_unpaired_surrogates(suite_file))
    if result.source_line is None:
        props = f"file={file_prop}"
    else:
        props = f"file={file_prop},line={result.source_line}"
    return f"::error {props}::{escape_message(message)}"


def annotation_lines(
    results: Sequence[CaseResult],
    suite_file: str,
    verdict: GateVerdict | None = None,
) -> list[str]:
    """Commands for failed cases, or one gate summary when none of them failed.

    Passed cases are skipped. A failing gate with at least one failed case
    does not add the summary: the case lines are the annotation. A failing
    gate with no failed case joins every ``FAIL`` reason into one
    ``::error::<reason>`` line.
    """
    failed = [row for row in results if not row.passed]
    if failed:
        return [case_annotation(row, suite_file) for row in failed]
    if verdict is None or verdict.passed:
        return []
    reasons = [reason for reason in verdict.reasons if reason.startswith("FAIL ")]
    if not reasons:
        return []
    # Strip surrogates the same way case annotations do, so a cost note that
    # quotes a model id from a ``\\ud800`` JSON escape cannot crash stderr.
    summary = strip_unpaired_surrogates("; ".join(reasons))
    return ["::error::" + escape_message(summary)]


def emit_annotations(
    results: Sequence[CaseResult],
    suite_file: str,
    verdict: GateVerdict | None = None,
) -> None:
    """Print :func:`annotation_lines` to stderr. No lines prints nothing."""
    for line in annotation_lines(results, suite_file, verdict):
        print(line, file=sys.stderr)
