"""Matrix ranking, including the demo's good and bad templates."""

import json
from pathlib import Path

import pytest
import yaml

from helpers import run_with
from promptregress.cli import (
    DEMO_BASELINE_TEMPLATE,
    DEMO_CANDIDATE_TEMPLATE,
    bundled_suite_path,
    demo_rules,
    main,
)
from promptregress.loading import load_suite
from promptregress.matrix import MatrixConfig, MatrixError, load_matrix, rank_runs


def test_rank_breaks_mean_ties_by_name() -> None:
    shared = [("a", 1.0, True), ("b", 0.0, False)]
    worse = [("a", 0.0, False), ("b", 0.0, False)]
    entries = [
        (_config("gamma"), run_with(shared, "gamma", suite="s"), "mock"),
        (_config("alpha"), run_with(shared, "alpha", suite="s"), "mock"),
        (_config("beta"), run_with(worse, "beta", suite="s"), "mock"),
    ]
    result = rank_runs(entries, suite_name="s", n_boot=40, seed=1)
    assert [row.name for row in result.rows] == ["alpha", "gamma", "beta"]
    assert result.best_name == "alpha"
    assert result.rows[0].diff_vs_best == 0.0
    assert result.rows[1].diff_vs_best == 0.0
    assert result.rows[0].ci_low == result.rows[0].ci_high == 0.0
    assert result.rows[2].diff_vs_best < 0.0
    assert result.rows[2].ci_low <= result.rows[2].diff_vs_best <= result.rows[2].ci_high


def test_load_matrix_rejects_a_single_config(tmp_path: Path) -> None:
    path = tmp_path / "one.yaml"
    path.write_text("only:\n  provider: echo\n  template: '{input}'\n", encoding="utf-8")
    with pytest.raises(MatrixError, match="at least two"):
        load_matrix(path)


def test_demo_templates_rank_good_then_bad_then_empty(tmp_path: Path) -> None:
    suite_path = bundled_suite_path()
    suite = load_suite(suite_path)
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "good",
                    "provider": "mock",
                    "model": "demo",
                    "template": DEMO_BASELINE_TEMPLATE,
                },
                {
                    "name": "bad",
                    "provider": "mock",
                    "model": "demo",
                    "template": DEMO_CANDIDATE_TEMPLATE,
                },
                {
                    "name": "empty",
                    "provider": "mock",
                    "model": "demo",
                    "template": "ZZZ_NO_MATCH",
                },
            ]
        ),
        encoding="utf-8",
    )
    report_json = tmp_path / "matrix.json"
    report_md = tmp_path / "matrix.md"
    code = main(
        [
            "matrix",
            str(suite_path),
            str(matrix),
            "--mock-rules",
            json.dumps(demo_rules(suite)),
            "--mock-default",
            "I do not know.",
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
            "--n-boot",
            "200",
            "--seed",
            "13",
        ]
    )
    assert code == 0
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    names = [row["name"] for row in payload["rows"]]
    assert names == ["good", "bad", "empty"]
    assert payload["best"] == "good"
    scores = [row["mean_score"] for row in payload["rows"]]
    assert scores[0] > scores[1] > scores[2]
    assert payload["rows"][0]["diff_vs_best"] == 0.0
    assert payload["rows"][1]["diff_vs_best"] < 0.0
    assert payload["rows"][2]["diff_vs_best"] < payload["rows"][1]["diff_vs_best"]
    for row in payload["rows"]:
        assert row["ci_low"] <= row["diff_vs_best"] <= row["ci_high"]
    markdown = report_md.read_text(encoding="utf-8")
    assert markdown.index("good") < markdown.index("bad") < markdown.index("empty")
    assert "Diff vs best" in markdown


def _config(name: str) -> MatrixConfig:
    return MatrixConfig(name=name, provider="mock", template="{input}", model="mock")
