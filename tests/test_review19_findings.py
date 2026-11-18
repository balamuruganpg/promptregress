"""Regression tests for bugs found in the independent v0.8.0 review cycle #19.

Scope is quote-dialect BOM stripping after the 0.7.1/0.8.0 JSON/YAML
backslash fix: CSV must not treat ``\\`` as an escape, and YAML
single-quoted continuations must keep a line-leading ``U+FEFF``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from promptregress.loading import load_suite, strip_utf8_bom_text


def test_csv_trailing_backslash_does_not_keep_next_row_bom(tmp_path: Path) -> None:
    """A CSV field ending in ``\\`` must still close at the following quote.

    Review #18 taught ``strip_utf8_bom_text`` to treat ``\\`` as escaping the
    next character inside double quotes (JSON/YAML ``\\"``). In CSV a
    backslash is literal, so a Windows directory path such as
    ``"C:\\\\temp\\\\"`` ends at the quote after the trailing backslash.
    Treating that pair as an escape left quote-state open, so a mid-file
    chunk BOM on the next row was kept and the case id became
    ``\\ufeffb``. CSV loading disables backslash escapes so the BOM strips
    and the id stays ``b``.
    """
    path = tmp_path / "cases.csv"
    path.write_text(
        'id,input,reference\na,"C:\\temp\\",ref-a\n\ufeffb,plain,ref-b\n',
        encoding="utf-8",
    )
    assert "\ufeff" not in strip_utf8_bom_text(
        path.read_text(encoding="utf-8"),
        backslash_escapes=False,
    )
    suite = load_suite(path)
    assert [case.id for case in suite.cases] == ["a", "b"]
    assert suite.cases[0].input == "C:\\temp\\"
    assert suite.cases[1].input == "plain"


def test_yaml_single_quoted_continuation_keeps_line_leading_feff(tmp_path: Path) -> None:
    """YAML single-quoted continuations keep a line-leading ``U+FEFF``.

    Double-quoted YAML was fixed in review #16/#18, but single-quoted
    scalars only escape ``''`` and treat backslash as literal. Quote-blind
    (and double-quote-only) stripping deleted a continuation-line
    ``U+FEFF`` that ``yaml.safe_load`` on the raw text kept. Track
    single-quoted regions so in-scalar ZWNBSP survives while chunk BOMs
    outside quotes still clear.
    """
    path = tmp_path / "suite.yaml"
    raw = (
        "name: t\n"
        "cases:\n"
        "- id: a\n"
        "  input: 'hello\n"
        "\ufeffworld'\n"
        "  reference: x\n"
        "  assertions:\n"
        "  - kind: contains\n"
        "    value: hello\n"
    )
    path.write_text(raw, encoding="utf-8")
    assert yaml.safe_load(raw)["cases"][0]["input"] == "hello \ufeffworld"
    suite = load_suite(path)
    assert suite.cases[0].input == "hello \ufeffworld"


def test_yaml_plain_apostrophe_still_strips_midfile_bom(tmp_path: Path) -> None:
    """A plain-scalar apostrophe must not hide a later chunk BOM.

    Single-quote tracking has to ignore ``it's`` in an unquoted YAML
    value; otherwise quote-state stays open and a mid-file BOM on the
    next case is kept as part of the id. Openers are recognised only
    after a YAML token boundary.
    """
    path = tmp_path / "suite.yaml"
    path.write_text(
        "name: t\n"
        "cases:\n"
        "- id: a\n"
        "  input: it's fine\n"
        "  reference: x\n"
        "  assertions:\n"
        "  - kind: contains\n"
        "    value: fine\n"
        "\ufeff- id: b\n"
        "  input: other\n"
        "  reference: y\n"
        "  assertions:\n"
        "  - kind: contains\n"
        "    value: other\n",
        encoding="utf-8",
    )
    suite = load_suite(path)
    assert [case.id for case in suite.cases] == ["a", "b"]
    assert suite.cases[0].input == "it's fine"
