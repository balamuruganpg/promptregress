"""Defects found in review 23 against the synth contract.

Pareto dominance, axis omission, and the pareto CLI are not covered here.
Each test quotes the README, changelog, or module docstring it enforces, and
fails on the current code.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from promptregress.cli import main
from promptregress.loading import load_suite
from promptregress.models import AssertionSpec, Case, Suite
from promptregress.synth import number_format, synonym, write_suite


def _suite(text: str) -> Suite:
    return Suite(
        name="edge",
        cases=[
            Case(
                id="c1",
                input=text,
                reference="ref",
                assertions=[AssertionSpec(kind="contains", value="ref")],
            )
        ],
    )


def test_number_format_leaves_a_fraction_glued_to_a_letter() -> None:
    """A number next to a letter is not a thousands swap.

    ``number_format`` docstring: "Other numbers are left alone: fewer than
    four digits with no thousands commas, a leading zero, a number next to
    a letter, and European dot-grouping such as ``1.000``." README: the
    operator is "``1,000`` ↔ ``1000``, where that form is unambiguous."

    ``1000e2`` is left alone. The same letter after a decimal fraction is
    still one number, but the optional fraction is dropped so the match can
    end on the dot and the integer is rewritten.
    """
    assert number_format("1000e2") == "1000e2"
    assert number_format("1e1000") == "1e1000"
    assert number_format("1,000.50") == "1000.50"
    assert number_format("1000.50") == "1,000.50"
    assert number_format("1.000") == "1.000"
    assert number_format("1000.50e2") == "1000.50e2"
    assert number_format("1,000.50e2") == "1,000.50e2"
    assert number_format("1000.50e2", 7) == number_format("1000.50e2", 0)
    assert number_format("1000.50v") == "1000.50v"
    assert number_format("see 10,000.50e3 now") == "see 10,000.50e3 now"


def test_number_format_letter_and_digit_boundaries_are_not_ascii_only() -> None:
    """Homoglyph letters and non-ASCII digits are not the safe thousands form.

    Same docstring as the fraction case: leave a number next to a letter,
    and only swap an unambiguous ``1,000`` / ``1000``. ``A1000`` is the
    documented ASCII example and stays put. A fullwidth or Cyrillic letter
    is still a letter, and Devanagari or fullwidth digits are not that
    ASCII form. ``seed`` does not change the result.
    """
    assert number_format("A1000") == "A1000"
    assert number_format("1000A") == "1000A"
    assert number_format("1.000") == "1.000"
    assert number_format("1,000.50") == "1000.50"
    assert number_format("\uff211000") == "\uff211000"  # fullwidth A
    assert number_format("1000\uff21") == "1000\uff21"
    assert number_format("\u043e1000") == "\u043e1000"  # Cyrillic o
    assert number_format("1000\u043e") == "1000\u043e"
    assert number_format("1000.50\u043e") == "1000.50\u043e"
    assert number_format("\u0967\u0966\u0966\u0966") == "\u0967\u0966\u0966\u0966"
    assert number_format("\uff11\uff12\uff13\uff14") == "\uff11\uff12\uff13\uff14"
    assert number_format("\uff211000", 3) == number_format("\uff211000", 0)


def test_synonym_does_not_match_inside_a_longer_word_continued_by_a_letter() -> None:
    """A longer word that merely contains a key is not a match.

    ``synonym`` docstring: "A longer word that merely contains a key
    (``beginning``) is not a match." ``beginning`` stays. The same prefix
    continued by a non-ASCII letter (``é``, or Cyrillic ``а`` in
    ``beginаning``) is still a longer word, not the whole key.
    """
    assert synonym("beginning") == "beginning"
    assert synonym("BEGIN") == "START"
    assert synonym("Begin") == "Start"
    assert synonym("begin\u00e9") == "begin\u00e9"
    assert synonym("quickly\u0430") == "quickly\u0430"
    assert synonym("begin\u0430ning") == "begin\u0430ning"
    assert synonym("\u0430quickly") == "\u0430quickly"
    assert synonym("begin\u0430ning", 4) == synonym("begin\u0430ning", 0)


def test_jsonl_round_trip_keeps_unicode_line_separators(tmp_path: Path) -> None:
    """JSONL is one case object per line and must reload.

    ``write_suite`` docstring: "``.jsonl`` writes one case object per line
    (the loader names that suite from the file stem and does not store
    scoring weights)." ``load_suite`` splits that file on lines.
    U+2028, U+2029, and U+0085 are line separators. Written raw, they split
    one JSON value across lines and the loader rejects the file. YAML keeps
    the same input, so the text itself is a legal case.
    """
    text = "a\u2028b\u2029c\u0085d"
    suite = _suite(text)
    yaml_path = tmp_path / "edge.yaml"
    jsonl_path = tmp_path / "edge.jsonl"
    write_suite(suite, yaml_path)
    assert load_suite(yaml_path).cases[0].input == text
    write_suite(suite, jsonl_path)
    loaded = load_suite(jsonl_path)
    assert loaded.cases[0].input == text
    assert loaded.cases[0].reference == "ref"
    assert loaded.cases[0].assertions[0].kind == "contains"


def test_csv_round_trip_keeps_a_crlf_inside_the_field(tmp_path: Path) -> None:
    """CSV reload returns the input and reference columns.

    ``write_suite`` docstring: "``.csv`` writes columns ``id``, ``input``,
    and ``reference``; the loader then attaches ``exact_match`` and drops
    tags, metadata, and other assertions." A CR LF inside a quoted field is
    part of that field. The bytes on disk keep the CR. ``load_suite`` reads
    the file in universal-newline mode, so the reloaded input loses it.
    """
    text = "cr\r\nlf"
    path = tmp_path / "edge.csv"
    write_suite(_suite(text), path)
    assert b'"cr\r\nlf"' in path.read_bytes()
    loaded = load_suite(path)
    assert loaded.cases[0].input == text
    assert loaded.cases[0].reference == "ref"


def test_hard_link_out_does_not_modify_the_suite(tmp_path: Path) -> None:
    """``--out`` must not write through a hard link to the suite.

    README: "It does not modify the input." ``synth_to_path``: "The source
    file is only read. ``--out`` must be a different path." A hard link is
    another directory entry for that same file. Resolving the path strings
    does not notice the shared inode, and the write replaces the suite.
    """
    source = tmp_path / "suite.json"
    write_suite(_suite("Hello."), source)
    before = source.read_bytes()
    link = tmp_path / "linked.json"
    os.link(source, link)
    code = main(
        [
            "synth",
            "--suite",
            str(source),
            "--out",
            str(link),
            "--operators",
            "punctuation",
            "--seed",
            "13",
        ]
    )
    assert source.read_bytes() == before
    assert code == 2
    assert link.stat().st_ino == source.stat().st_ino


def test_symlink_loop_out_is_an_unwritable_path_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unwritable ``--out`` raises ``ValueError`` and the CLI exits 2.

    Changelog: "An unknown operator, an empty suite, or an unwritable
    output path raises ``ValueError`` (CLI exit 2)." The synth help text
    says the same for an unwritable ``--out``. A symlink loop cannot be
    written (the write itself is ``OSError``). ``Path.resolve`` on that
    path raises ``RuntimeError``, which ``_same_path`` does not catch, so
    the command crashes instead of exiting 2. The suite file stays put.
    """
    source = tmp_path / "suite.json"
    write_suite(_suite("Hello."), source)
    before = source.read_bytes()
    left = tmp_path / "loop-a"
    right = tmp_path / "loop-b"
    left.symlink_to(right)
    right.symlink_to(left)
    try:
        code = main(
            [
                "synth",
                "--suite",
                str(source),
                "--out",
                str(left / "out.json"),
                "--operators",
                "punctuation",
            ]
        )
    except Exception as exc:
        pytest.fail(f"unwritable --out crashed with {type(exc).__name__}: {exc}")
    captured = capsys.readouterr()
    assert code == 2
    assert "error:" in captured.err
    assert source.read_bytes() == before
