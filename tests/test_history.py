"""History JSONL, the history table, and the trend sparkline."""

import json
from pathlib import Path

import pytest

from promptregress.cli import main
from promptregress.history import (
    append_history,
    ascii_sparkline,
    format_trend,
    load_history,
    render_trend_html,
    suite_points,
)


def test_ascii_sparkline_uses_a_fixed_zero_to_one_scale() -> None:
    assert ascii_sparkline([0.0, 0.5, 1.0]) == "▁▅█"
    assert ascii_sparkline([1.0, 1.0]) == "██"
    assert ascii_sparkline([]) == ""


def test_run_appends_history_and_opt_out_skips_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [
                    {
                        "id": "a",
                        "input": "alpha",
                        "reference": "alpha",
                        "assertions": [{"kind": "exact_match"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "run.json"
    assert (
        main(
            [
                "run",
                str(suite),
                "--provider",
                "echo",
                "--config-name",
                "echo",
                "--template",
                "{input}",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    path = tmp_path / ".promptregress" / "history.jsonl"
    rows, notes = load_history(path)
    assert notes == []
    assert len(rows) == 1
    assert rows[0]["suite"] == "tiny"
    assert rows[0]["command"] == "run"
    assert rows[0]["metrics"]["mean_score"] == 1.0
    assert "config" in rows[0]
    quiet = tmp_path / "quiet.json"
    assert (
        main(
            [
                "run",
                str(suite),
                "--provider",
                "echo",
                "--config-name",
                "echo",
                "--template",
                "{input}",
                "--out",
                str(quiet),
                "--no-history",
            ]
        )
        == 0
    )
    again, _notes = load_history(path)
    assert len(again) == 1


def test_history_write_failure_warns_and_run_still_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".promptregress").write_text("not a directory", encoding="utf-8")
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [
                    {
                        "id": "a",
                        "input": "alpha",
                        "reference": "alpha",
                        "assertions": [{"kind": "exact_match"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "run",
            str(suite),
            "--provider",
            "echo",
            "--config-name",
            "echo",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "warning:" in captured.err
    assert "history" in captured.err
    assert "mean_score=1.000000" in captured.out


def test_history_table_and_trend_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "history.jsonl"
    append_history(
        {
            "timestamp": "2026-11-11T00:00:00Z",
            "command": "run",
            "suite": "tiny",
            "config": {"provider": "echo"},
            "metrics": {"mean_score": 0.0},
        },
        path,
    )
    append_history(
        {
            "timestamp": "2026-11-11T00:01:00Z",
            "command": "compare",
            "suite": "other",
            "config": {"provider": "echo"},
            "metrics": {"mean_score": 1.0, "mean_diff": 0.0, "gate_passed": True},
        },
        path,
    )
    append_history(
        {
            "timestamp": "2026-11-11T00:02:00Z",
            "command": "run",
            "suite": "tiny",
            "config": {"provider": "echo"},
            "metrics": {"mean_score": 1.0},
        },
        path,
    )
    path.write_text(path.read_text(encoding="utf-8") + "not-json\n", encoding="utf-8")
    code = main(["history", "--file", str(path)])
    assert code == 0
    points = suite_points(load_history(path)[0], "tiny")
    assert points == [("2026-11-11T00:00:00Z", 0.0), ("2026-11-11T00:02:00Z", 1.0)]
    out = tmp_path / "trend.html"
    trend = main(["trend", "tiny", "--out", str(out), "--file", str(path)])
    assert trend == 0
    text = out.read_text(encoding="utf-8")
    assert "<svg" in text
    assert "polyline" in text
    assert "tiny" in text
    assert "<script" not in text
    assert "http://" not in text
    assert "https://" not in text
    assert ascii_sparkline([0.0, 1.0]) in text
    assert format_trend("tiny", [0.0, 1.0]).strip().splitlines()[-1] == "▁█"
    missing = main(["trend", "missing", "--out", str(tmp_path / "no.html"), "--file", str(path)])
    assert missing == 2


def test_compare_history_records_the_candidate_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [
                    {
                        "id": "a",
                        "input": "alpha",
                        "reference": "alpha",
                        "assertions": [{"kind": "exact_match"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "compare",
            str(suite),
            "--provider",
            "echo",
            "--baseline-template",
            "{input}",
            "--candidate-template",
            "{input}",
            "--max-regression",
            "0",
        ]
    )
    assert code == 0
    capsys.readouterr()
    rows, _notes = load_history()
    assert len(rows) == 1
    assert rows[0]["command"] == "compare"
    assert rows[0]["suite"] == "tiny"
    assert rows[0]["metrics"]["mean_score"] == 1.0
    assert rows[0]["metrics"]["gate_passed"] is True
    table = main(["history"])
    assert table == 0
    shown = capsys.readouterr().out
    assert "tiny" in shown
    assert "compare" in shown
    assert "mean_score=1.000000" in shown


def test_render_handles_one_point() -> None:
    html = render_trend_html("s", [("t", 0.25)])
    assert "<circle" in html
    assert "polyline" not in html
