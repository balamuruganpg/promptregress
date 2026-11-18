"""Regression tests for bugs found in the independent v0.6.14 review cycle.

Scope is incomplete mid-file BOM handling after 0.6.14 fixed JSONL only:
CSV suites, matrix YAML, and pricing YAML still glued a line-leading
``U+FEFF`` onto case ids / config names / model keys when Windows-saved
chunks were concatenated; history JSONL with invalid UTF-8 still raised a
bare ``UnicodeDecodeError`` instead of the clear ``ValueError`` used by
calibration and run reports.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from promptregress.history import load_history
from promptregress.loading import load_suite
from promptregress.matrix import load_matrix
from promptregress.pricing import load_pricing


def test_csv_suite_strips_bom_on_each_line(tmp_path: Path) -> None:
    """Concatenated CSV chunks each carrying a UTF-8 BOM must keep real ids.

    0.6.14 stripped per-line BOMs for JSONL suites. CSV still only stripped
    a leading file BOM, so ``cat`` of two Windows-saved CSV files left
    ``U+FEFF`` on the first field of later rows. The case id became
    ``\\ufeffb``, annotations named the wrong id, and duplicate detection
    could not see a real ``b``. Strip line-leading BOMs before DictReader
    the same way JSONL does before ``json.loads``.
    """
    path = tmp_path / "cases.csv"
    path.write_bytes(b"id,input,reference\na,x,y\n\xef\xbb\xbfb,z,w\n")
    suite = load_suite(path)
    assert [(case.id, case.source_line) for case in suite.cases] == [("a", 2), ("b", 3)]


def test_matrix_and_pricing_yaml_strip_midfile_bom(tmp_path: Path) -> None:
    """Matrix and pricing YAML must not rename keys after a mid-file BOM.

    Mapping-style YAML tolerates one leading BOM, and 0.6.14 taught matrix
    to strip a double BOM at the file start. A BOM at the start of a later
    line (concatenated Windows-saved chunks) still stuck on the next key:
    config ``b`` loaded as ``\\ufeffb`` and pricing ``model-b`` was missing
    from the table, so cost lookup failed for the real model id. Per-line
    BOM stripping keeps the documented names.
    """
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_bytes(
        b"a:\n  provider: echo\n  template: '{input}'\n"
        b"\xef\xbb\xbfb:\n  provider: echo\n  template: '{input}'\n"
    )
    assert [config.name for config in load_matrix(matrix_path)] == ["a", "b"]

    pricing_path = tmp_path / "prices.yaml"
    pricing_path.write_bytes(
        b"model-a:\n  prompt_per_1k: 1\n  completion_per_1k: 1\n"
        b"\xef\xbb\xbfmodel-b:\n  prompt_per_1k: 2\n  completion_per_1k: 2\n"
    )
    prices = load_pricing(pricing_path)
    assert set(prices) == {"model-a", "model-b"}
    assert prices["model-b"].prompt_per_1k == 2.0


def test_history_invalid_utf8_raises_value_error(tmp_path: Path) -> None:
    """Invalid UTF-8 in history JSONL must raise ``ValueError``, not a codec error.

    0.6.14 wrapped decode failures for run reports, compare reports, and
    calibration JSONL as ``ValueError: … not valid UTF-8``. ``load_history``
    still let ``UnicodeDecodeError`` escape, so ``history`` / ``trend``
    printed a bare codec message instead of the same clear failure.
    """
    path = tmp_path / "history.jsonl"
    path.write_bytes(
        b'{"timestamp":"t","command":"run","suite":"s","config":{},'
        b'"metrics":{"mean_score":1.0}}\n\xff'
    )
    with pytest.raises(ValueError, match="not valid UTF-8") as exc:
        load_history(path)
    assert type(exc.value) is ValueError
