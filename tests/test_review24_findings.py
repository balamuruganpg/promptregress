"""Defects found in review 24 against the changepoint and select contracts.

Each test quotes the README or module docstring it enforces.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from promptregress.changepoint import changepoint_to_dict, detect_changepoint
from promptregress.cli import main
from promptregress.history import append_history
from promptregress.loading import load_suite
from promptregress.models import Comparison, PairedRow, RunResult, Suite
from promptregress.select import prioritize, selection_to_dict


def _rows(scores: list[float]) -> list[dict[str, object]]:
    return [{"metrics": {"mean_score": score}} for score in scores]


def _observation(
    case_id: str,
    baseline_score: float,
    candidate_score: float,
    *,
    baseline_passed: bool = True,
    candidate_passed: bool = True,
) -> PairedRow:
    return PairedRow(
        case_id=case_id,
        baseline_output="b",
        candidate_output="c",
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        diff=candidate_score - baseline_score,
        baseline_passed=baseline_passed,
        candidate_passed=candidate_passed,
        outcome="loss" if candidate_score < baseline_score else "tie",
    )


def _comparison(rows: list[PairedRow]) -> Comparison:
    return Comparison(
        baseline=RunResult("s", "baseline", []),
        candidate=RunResult("s", "candidate", []),
        mean_diff=0.0,
        ci_low=0.0,
        ci_high=0.0,
        mcnemar_p=1.0,
        wins=0,
        losses=0,
        ties=0,
        per_case=rows,
    )


def _suite(ids: list[str], name: str = "demo") -> Suite:
    from promptregress.models import AssertionSpec, Case

    return Suite(
        name=name,
        cases=[
            Case(
                id=case_id,
                input=case_id,
                reference=case_id,
                assertions=[AssertionSpec(kind="exact_match")],
            )
            for case_id in ids
        ],
    )


def test_large_finite_means_keep_the_cusum_and_json_finite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A finite ``mean_score`` series has a finite segment mean and a finite report.

    Changepoint module docstring: each segment mean is ``math.fsum`` of that
    segment divided by its length, and a non-finite statistic raises
    ``ValueError``. README: the command exits 2 when a present ``mean_score``
    is not finite, and JSON keeps the stored floats. Four ``1e308`` means
    followed by four zeros is not a constant series. The split at 4 has
    before mean ``1e308`` and statistic ``sqrt(2) * 1e308``. ``fsum`` of the
    large half raises ``OverflowError``, so the permutation test crashes
    instead of returning that number, and the JSON writer never runs.

    The opposite step, four ``1e308`` then four ``-1e308``, has a difference
    outside the float range. That must raise ``ValueError`` (exit 2), not
    ``OverflowError``, and must not write ``Infinity``.
    """
    big = 1e308
    scores = [big, big, big, big, 0.0, 0.0, 0.0, 0.0]
    report = detect_changepoint(_rows(scores), seed=13, permutations=20)
    assert report.change_index == 4
    assert report.before_n == 4
    assert report.after_n == 4
    assert report.before_mean == big
    assert report.after_mean == 0.0
    assert report.mean_shift == -big
    assert report.statistic == math.sqrt(2.0) * big
    assert math.isfinite(report.p_value)
    assert 0.0 < report.p_value <= 1.0
    payload = changepoint_to_dict(report)
    encoded = json.dumps(payload, allow_nan=False)
    parsed = json.loads(encoded)
    assert parsed["before_mean"] == big
    assert parsed["statistic"] == report.statistic
    assert math.isfinite(parsed["p_value"])
    assert "Infinity" not in encoded
    assert "NaN" not in encoded

    history = tmp_path / "history.jsonl"
    for score in scores:
        append_history({"metrics": {"mean_score": score}}, history)
    out = tmp_path / "report.json"
    code = main(
        [
            "changepoint",
            "--file",
            str(history),
            "--permutations",
            "20",
            "--json",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["change_index"] == 4
    assert written["before_mean"] == big
    assert "Infinity" not in out.read_text(encoding="utf-8")
    assert "NaN" not in out.read_text(encoding="utf-8")
    capsys.readouterr()

    opposite = tmp_path / "opposite.jsonl"
    for score in [big, big, big, big, -big, -big, -big, -big]:
        append_history({"metrics": {"mean_score": score}}, opposite)
    refused = tmp_path / "refused.json"
    try:
        code = main(
            ["changepoint", "--file", str(opposite), "--permutations", "5", "--out", str(refused)]
        )
    except Exception as exc:
        pytest.fail(f"non-finite statistic crashed with {type(exc).__name__}: {exc}")
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "non-finite" in captured.err
    assert not refused.exists()


def test_csv_selected_suite_reloads_a_cr_inside_the_field(tmp_path: Path) -> None:
    """A CSV-selected suite reloads the same case text the source loader saw.

    Select module docstring: JSONL and CSV copy the selected case rows, and
    the file is an ordinary suite that ``run`` and ``compare`` score with no
    extra flag. The CSV loader reads with ``newline=""`` so a CR LF inside a
    quoted field stays part of that field. Selecting the row must not drop
    the CR, or the reloaded case is a different input.
    """
    text = 'id,input,reference\nbilling,"cr\r\nlf",ref\ngreeting,hello,ref\n'
    source = tmp_path / "sheet.csv"
    source.write_bytes(text.encode("utf-8"))
    loaded = load_suite(source)
    assert loaded.cases[0].input == "cr\r\nlf"
    comparison = tmp_path / "cmp.json"
    _write_pair(
        comparison,
        [
            ("billing", 1.0, True, 0.0, False),
            ("greeting", 1.0, True, 1.0, True),
        ],
    )
    out = tmp_path / "picked.csv"
    from promptregress.select import select_suite

    select_suite(source, [comparison], budget=1, out_path=out, seed=13)
    reloaded = load_suite(out)
    assert [case.id for case in reloaded.cases] == ["billing"]
    assert reloaded.cases[0].input == "cr\r\nlf"
    assert reloaded.cases[0].reference == "ref"
    assert b"\r" in out.read_bytes()


def test_jsonl_selected_suite_reloads_unicode_line_separators(tmp_path: Path) -> None:
    """JSONL stays one case per line, so the selected suite reloads.

    ``load_suite`` docstring: JSONL is one case object per line, and the
    loader splits the file on lines. Select copies those rows and the result
    is an ordinary suite. U+2028, U+2029, and U+0085 are line separators.
    Written raw, they split one JSON value across lines and the loader
    rejects the file.
    """
    text = "a\u2028b\u2029c\u0085d"
    source = tmp_path / "rows.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "sep",
                        "input": text,
                        "reference": "ref",
                        "assertions": [{"kind": "exact_match"}],
                    },
                    ensure_ascii=True,
                ),
                json.dumps(
                    {
                        "id": "plain",
                        "input": "hello",
                        "reference": "ref",
                        "assertions": [{"kind": "exact_match"}],
                    },
                    ensure_ascii=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_suite(source).cases[0].input == text
    comparison = tmp_path / "cmp.json"
    _write_pair(
        comparison,
        [
            ("sep", 1.0, True, 0.0, False),
            ("plain", 1.0, True, 1.0, True),
        ],
    )
    out = tmp_path / "picked.jsonl"
    from promptregress.select import select_suite

    select_suite(source, [comparison], budget=1, out_path=out, seed=13)
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1
    reloaded = load_suite(out)
    assert [case.id for case in reloaded.cases] == ["sep"]
    assert reloaded.cases[0].input == text
    assert reloaded.cases[0].reference == "ref"


def test_hard_linked_out_does_not_modify_the_source_suite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--out`` must not write through a hard link to the suite.

    Select module docstring: "The source file is not modified. ``--out`` must
    be a different path." A hard link is another directory entry for that
    same file. Resolving the path strings does not notice the shared inode,
    and the write replaces the suite.
    """
    source = tmp_path / "suite.json"
    source.write_text(
        json.dumps(
            {
                "name": "demo",
                "cases": [
                    {
                        "id": "a",
                        "input": "a",
                        "reference": "a",
                        "assertions": [{"kind": "exact_match"}],
                    },
                    {
                        "id": "b",
                        "input": "b",
                        "reference": "b",
                        "assertions": [{"kind": "exact_match"}],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    before = source.read_bytes()
    link = tmp_path / "linked.json"
    os.link(source, link)
    comparison = tmp_path / "cmp.json"
    _write_pair(comparison, [("a", 1.0, True, 0.0, False), ("b", 1.0, True, 1.0, True)])
    code = main(
        [
            "select",
            "--suite",
            str(source),
            "--budget",
            "1",
            "--out",
            str(link),
            "--seed",
            "13",
            str(comparison),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "error:" in captured.err
    assert source.read_bytes() == before
    assert link.stat().st_ino == source.stat().st_ino


def test_symlink_loop_out_is_an_unwritable_path_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unwritable ``--out`` raises ``ValueError`` and the CLI exits 2.

    README: an unwritable output raises ``ValueError`` (exit 2). Select's
    path check resolves both paths. A symlink loop makes ``Path.resolve``
    raise ``RuntimeError``, which is not a ``ValueError``, so the command
    crashes instead of exiting 2. The suite file stays put.
    """
    source = tmp_path / "suite.json"
    source.write_text(
        json.dumps(
            {
                "name": "demo",
                "cases": [
                    {
                        "id": "a",
                        "input": "a",
                        "reference": "a",
                        "assertions": [{"kind": "exact_match"}],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    before = source.read_bytes()
    comparison = tmp_path / "cmp.json"
    _write_pair(comparison, [("a", 1.0, True, 0.0, False)])
    left = tmp_path / "loop-a"
    right = tmp_path / "loop-b"
    left.symlink_to(right)
    right.symlink_to(left)
    try:
        code = main(
            [
                "select",
                "--suite",
                str(source),
                "--budget",
                "1",
                "--out",
                str(left / "out.json"),
                str(comparison),
            ]
        )
    except Exception as exc:
        pytest.fail(f"unwritable --out crashed with {type(exc).__name__}: {exc}")
    captured = capsys.readouterr()
    assert code == 2
    assert "error:" in captured.err
    assert source.read_bytes() == before


def test_large_finite_scores_keep_finite_population_moments() -> None:
    """Population moments of finite scores stay finite, and JSON has no Infinity.

    Select module docstring: candidate variance is the population variance
    (one observation, and any equal series, has variance 0). Sums are
    ``math.fsum`` of the values sorted ascending. Scores outside ``[0, 1]``
    are not clamped. ``mean_drop`` is the mean of ``baseline_score -
    candidate_score`` on strict drops. A non-finite component raises
    ``ValueError``. Two candidate scores of ``1e308`` have variance 0. Two
    drops of ``1e308`` have mean drop ``1e308``. ``fsum`` raises
    ``OverflowError`` on those sums, and ``(value - mean) ** 2`` raises on a
    deviation whose square is not finite, so the formula crashes instead of
    returning the finite moment or ``ValueError``. The report JSON must not
    contain NaN or Infinity.
    """
    big = 1e308
    equal = prioritize(
        _suite(["wide"]),
        [
            _comparison([_observation("wide", 0.0, big)]),
            _comparison([_observation("wide", 0.0, big)]),
        ],
        budget=1,
        seed=13,
    )
    item = equal.selected[0]
    assert item.n_observed == 2
    assert item.candidate_variance == 0.0
    assert item.variance_component == 0.0
    assert item.mean_drop == 0.0
    assert item.n_regressed == 0
    assert item.priority == 0.0
    encoded = json.dumps(selection_to_dict(equal), allow_nan=False)
    assert "Infinity" not in encoded
    assert "NaN" not in encoded

    drops = prioritize(
        _suite(["drop"]),
        [
            _comparison([_observation("drop", big, 0.0)]),
            _comparison([_observation("drop", big, 0.0)]),
        ],
        budget=1,
        seed=4,
    )
    dropped = drops.selected[0]
    assert dropped.candidate_variance == 0.0
    assert dropped.n_regressed == 2
    assert dropped.mean_drop == big
    assert dropped.priority == big / 3.0
    assert drops.seed == 4
    encoded = json.dumps(selection_to_dict(drops), allow_nan=False)
    assert "Infinity" not in encoded
    assert "NaN" not in encoded
    assert json.loads(encoded)["selected"][0]["mean_drop"] == big

    with pytest.raises(ValueError, match="not finite"):
        prioritize(
            _suite(["wide"]),
            [
                _comparison([_observation("wide", 0.0, big)]),
                _comparison([_observation("wide", 0.0, -big)]),
            ],
            budget=1,
        )


def _write_pair(
    path: Path,
    rows: list[tuple[str, float, bool, float, bool]],
) -> None:
    from promptregress.models import CaseResult
    from promptregress.report import write_json
    from promptregress.stats import compare_runs

    baseline = RunResult(
        "s",
        "baseline",
        [
            CaseResult(case_id, "b", {"exact_match": b_score}, b_score, b_pass, 1.0)
            for case_id, b_score, b_pass, _c_score, _c_pass in rows
        ],
    )
    candidate = RunResult(
        "s",
        "candidate",
        [
            CaseResult(case_id, "c", {"exact_match": c_score}, c_score, c_pass, 1.0)
            for case_id, _b_score, _b_pass, c_score, c_pass in rows
        ],
    )
    write_json(compare_runs(baseline, candidate, n_boot=20, seed=1), path)
