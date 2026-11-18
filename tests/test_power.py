"""Paired-test sample size and achieved power."""

import json
from pathlib import Path

import pytest

from helpers import run_with
from promptregress.cli import main
from promptregress.power import achieved_power, analytic_power, required_n
from promptregress.report import write_json
from promptregress.stats import compare_runs


def test_required_n_shrinks_as_the_effect_grows() -> None:
    small = required_n(0.2, n_sim=30, seed=1)
    mid = required_n(0.5, alpha=0.05, power=0.8, n_sim=4000, seed=13)
    large = required_n(0.8, n_sim=30, seed=1)
    assert small.n > mid.n > large.n
    assert mid.n == 32
    assert mid.n_continuous == pytest.approx(31.4, abs=0.2)
    assert 0.0 <= mid.analytic_power <= 1.0
    assert 0.0 <= mid.simulated_power <= 1.0
    assert abs(mid.simulated_power - mid.analytic_power) < 0.05
    assert mid.analytic_power == pytest.approx(analytic_power(0.5, mid.n, 0.05))
    repeated = required_n(0.5, n_sim=4000, seed=13)
    assert repeated.simulated_power == mid.simulated_power


def test_achieved_power_stays_inside_the_unit_interval() -> None:
    power = achieved_power([0.2, -0.1, 0.0, 0.4, 0.1], alpha=0.05)
    assert 0.0 <= power <= 1.0
    assert achieved_power([0.0, 0.0, 0.0, 0.0]) == pytest.approx(0.05)
    assert achieved_power([0.2]) == 0.0
    assert achieved_power([0.3, 0.3, 0.3]) == 1.0


def test_power_cli_effect_and_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["power", "--effect", "0.5", "--seed", "13"])
    assert code == 0
    text = capsys.readouterr().out
    assert "| 0.80 | 32 |" in text
    assert "| 0.50 |" in text
    assert "| 0.95 |" in text
    assert "simulated_power" in text

    comparison = compare_runs(
        run_with([("a", 1.0, True), ("b", 1.0, True), ("c", 1.0, True), ("d", 1.0, True)], "base"),
        run_with(
            [("a", 0.0, False), ("b", 1.0, True), ("c", 0.0, False), ("d", 1.0, True)],
            "cand",
        ),
        n_boot=20,
    )
    report = tmp_path / "comparison.json"
    out = tmp_path / "power.json"
    write_json(comparison, report)
    code = main(["power", "--from-report", str(report), "--out", str(out), "--seed", "3"])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert 0.0 <= payload["achieved_power"] <= 1.0
    assert payload["grid"]
    assert "achieved_power=" in capsys.readouterr().out
