"""Flakiness across stored baselines: flips, ordering, and the CI flag."""

import json
import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

from promptregress.baseline import save_baseline
from promptregress.cli import main
from promptregress.irt import load_run_report
from promptregress.models import AssertionSpec, Case, Suite
from promptregress.providers import Completion, MockProvider
from promptregress.report import write_run_json
from promptregress.runner import run_suite


class _VaryingMock:
    """Hide ``complete_at`` so repeated runs cycle ``MockProvider`` alternates.

    The runner pins sample index 0 when ``complete_at`` exists, which would
    keep every baseline on the first alternate.
    """

    def __init__(self, inner: MockProvider) -> None:
        self._inner = inner
        self.name = inner.name
        self.model = inner.model

    def complete(self, prompt: str) -> str:
        return self._inner.complete(prompt)

    def complete_detailed(self, prompt: str) -> Completion:
        return self._inner.complete_detailed(prompt)


def _suite() -> Suite:
    def case(case_id: str, text: str) -> Case:
        return Case(
            id=case_id,
            input=text,
            reference="good",
            assertions=[AssertionSpec("exact_match")],
        )

    return Suite(
        name="flicker",
        cases=[
            case("flipper", "FLIP"),
            case("stable_pass", "PASS"),
            case("stable_fail", "FAIL"),
            case("partial", "PARTIAL"),
        ],
    )


def _store_three(tmp_path: Path) -> None:
    """Three baselines via ``write_run_json`` (``run_to_dict``).

    Names are not in mtime order. ``flipper`` alternates good, bad, bad.
    ``partial`` is removed from the last run after it is written.
    """
    provider = _VaryingMock(
        MockProvider(
            rules=[
                ("FLIP", ("good", "bad", "bad")),
                ("PASS", "good"),
                ("FAIL", "bad"),
                ("PARTIAL", "good"),
            ],
            default="bad",
            vary=True,
        )
    )
    suite = _suite()
    names = ("z-first", "a-second", "m-third")
    for index, name in enumerate(names):
        result = run_suite(suite, provider, f"cfg-{index}")
        path, _overwritten = save_baseline(result, name, cwd=tmp_path)
        if index == 0:
            saved = path.read_text(encoding="utf-8")
            assert '"output"' in saved
            assert '"mean_score"' in saved
        stamp = (1_700_000_000 + index) * 1_000_000_000
        os.utime(path, ns=(stamp, stamp))
    last = tmp_path / ".promptregress" / "baselines" / "m-third.json"
    loaded_text = last.read_text(encoding="utf-8")
    assert "partial" in loaded_text
    stored = load_run_report(last)
    trimmed = replace(
        stored,
        results=[row for row in stored.results if row.case_id != "partial"],
    )
    write_run_json(trimmed, last)
    os.utime(last, ns=(1_700_000_002 * 1_000_000_000, 1_700_000_002 * 1_000_000_000))
    other = replace(run_suite(suite, provider, "ignored"), suite_name="other")
    save_baseline(other, "other-suite", cwd=tmp_path)


def test_flip_counts_ordering_and_missing_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _store_three(tmp_path)
    out = tmp_path / "flaky.json"
    assert main(["flaky", "flicker", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.index("flipper") < text.index("stable_fail") < text.index("stable_pass")
    assert text.index("stable_pass") < text.index("partial")
    assert "excluded from flip math" in text

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["enough"] is True
    assert payload["n_runs"] == 3
    assert [run["name"] for run in payload["runs"]] == ["z-first", "a-second", "m-third"]
    assert [case["case_id"] for case in payload["cases"]] == [
        "flipper",
        "stable_fail",
        "stable_pass",
        "partial",
    ]
    flipper = payload["cases"][0]
    assert flipper["flip_count"] == 1
    assert flipper["flip_rate"] == pytest.approx(0.5)
    assert flipper["scores"] == [1.0, 0.0, 0.0]
    assert flipper["passed"] == [True, False, False]
    assert flipper["mean"] == pytest.approx(1 / 3)
    assert flipper["sd"] == pytest.approx(math.sqrt(2 / 9))
    assert flipper["min"] == 0.0
    assert flipper["max"] == 1.0
    assert flipper["note"] is None
    assert payload["cases"][1]["flip_count"] == 0
    assert payload["cases"][1]["scores"] == [0.0, 0.0, 0.0]
    assert payload["cases"][1]["sd"] == 0.0
    assert payload["cases"][2]["flip_count"] == 0
    assert payload["cases"][2]["flip_rate"] == 0.0
    assert payload["cases"][2]["scores"] == [1.0, 1.0, 1.0]
    partial = payload["cases"][3]
    assert partial["n_runs"] == 2
    assert partial["n_suite_runs"] == 3
    assert partial["flip_count"] is None
    assert partial["flip_rate"] is None
    assert partial["scores"] == [1.0, 1.0]
    assert partial["mean"] == 1.0
    assert partial["sd"] == 0.0
    assert partial["note"] == "present in 2 of 3 runs; excluded from flip math"

    capsys.readouterr()
    flagged = tmp_path / "via-flag.json"
    assert main(["flaky", "--suite", "flicker", "--out", str(flagged)]) == 0
    again = json.loads(flagged.read_text(encoding="utf-8"))
    assert again["cases"] == payload["cases"]


def test_fail_on_flaky_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _store_three(tmp_path)

    assert main(["flaky", "flicker", "--fail-on-flaky", "0.5"]) == 1
    failed = capsys.readouterr().out
    assert "flaky: flipper flip_rate=0.500000 >= 0.500000" in failed
    assert "stable_pass" not in failed.split("flaky: ", 1)[-1]
    assert "partial" not in failed.split("flaky:")[-1]

    assert main(["flaky", "flicker", "--fail-on-flaky", "0.51"]) == 0
    assert "flaky:" not in capsys.readouterr().out

    assert main(["flaky", "flicker", "--fail-on-flaky", "0"]) == 1
    zero = capsys.readouterr().out
    lines = [line for line in zero.splitlines() if line.startswith("flaky:")]
    assert [line.split()[1] for line in lines] == ["flipper", "stable_fail", "stable_pass"]


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_runs_exits_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    count: int,
) -> None:
    monkeypatch.chdir(tmp_path)
    if count == 1:
        result = run_suite(_suite(), MockProvider(default="good"), "only")
        save_baseline(result, "only", cwd=tmp_path)
    out = tmp_path / "few.json"
    assert main(["flaky", "flicker", "--out", str(out), "--fail-on-flaky", "0"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("not enough runs:")
    assert "flicker" in text
    assert f"has {count} stored" in text

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["enough"] is False
    assert payload["n_runs"] == count
    assert payload["cases"] == []
    assert "not enough runs" in payload["message"]


def test_flaky_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["flaky"]) == 2
    assert "suite name" in capsys.readouterr().err
    assert main(["flaky", "left", "--suite", "right"]) == 2
    assert "does not match" in capsys.readouterr().err
    assert main(["flaky", "flicker", "--fail-on-flaky", "nan"]) == 2
    assert "[0, 1]" in capsys.readouterr().err
    assert main(["flaky", "flicker", "--fail-on-flaky", "1.1"]) == 2
