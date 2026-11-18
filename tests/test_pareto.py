"""Pareto frontier: dominance, extraction from real reports, and the CLI."""

import json
import math
from pathlib import Path

import pytest

from promptregress.cli import main
from promptregress.matrix import MatrixResult, MatrixRow, write_matrix_json
from promptregress.models import CaseResult, CostEstimate, RunResult
from promptregress.pareto import (
    PARETO_NOTE,
    VariantPoint,
    dominates,
    format_pareto,
    load_variants,
    pareto_from_files,
    pareto_frontier,
    pareto_to_dict,
)
from promptregress.report import write_json, write_run_json
from promptregress.stats import compare_runs

ROOT = Path(__file__).resolve().parents[1]
AXES3 = ("score", "cost", "latency")


def _point(
    name: str,
    score: float,
    cost: float | None,
    latency: float | None,
    source: str = "",
) -> VariantPoint:
    return VariantPoint(name, score, cost, latency, source)


def _case(case_id: str, score: float, latency_ms: float) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        output="out",
        scores={"exact_match": score},
        score=score,
        passed=score >= 0.999,
        latency_ms=latency_ms,
    )


def _run(
    name: str,
    scores: list[float],
    latencies: list[float],
    cost: CostEstimate | None = None,
) -> RunResult:
    return RunResult(
        suite_name="pareto-demo",
        config_name=name,
        results=[
            _case(f"c{index}", score, latency)
            for index, (score, latency) in enumerate(zip(scores, latencies, strict=True))
        ],
        cost=cost,
    )


def test_clear_frontier_and_unique_dominators() -> None:
    high_score = _point("high_score", 0.9, 3.0, 30.0)
    low_cost = _point("low_cost", 0.5, 1.0, 30.0)
    low_latency = _point("low_latency", 0.5, 3.0, 10.0)
    worse_score = _point("worse_score", 0.8, 3.0, 30.0)
    worse_cost = _point("worse_cost", 0.4, 2.0, 40.0)
    points = [worse_cost, high_score, low_latency, worse_score, low_cost]

    assert dominates(high_score, worse_score, AXES3)
    assert not dominates(low_cost, worse_score, AXES3)
    assert not dominates(low_latency, worse_score, AXES3)
    assert dominates(low_cost, worse_cost, AXES3)
    assert not dominates(high_score, worse_cost, AXES3)
    assert not dominates(low_latency, worse_cost, AXES3)
    assert not dominates(high_score, low_cost, AXES3)
    assert not dominates(low_cost, high_score, AXES3)
    assert not dominates(high_score, low_latency, AXES3)
    assert not dominates(low_latency, high_score, AXES3)
    assert not dominates(low_cost, low_latency, AXES3)

    report = pareto_frontier(points)
    assert report.axes == AXES3
    assert report.omitted == ()
    assert report.note == PARETO_NOTE
    by_name = {row.name: row for row in report.rows}
    assert by_name["high_score"].on_frontier
    assert by_name["low_cost"].on_frontier
    assert by_name["low_latency"].on_frontier
    assert by_name["high_score"].dominated_by is None
    assert not by_name["worse_score"].on_frontier
    assert by_name["worse_score"].dominated_by == "high_score"
    assert not by_name["worse_cost"].on_frontier
    assert by_name["worse_cost"].dominated_by == "low_cost"
    assert [row.name for row in report.rows] == [
        "high_score",
        "low_cost",
        "low_latency",
        "worse_score",
        "worse_cost",
    ]
    reversed_names = [row.name for row in pareto_frontier(list(reversed(points))).rows]
    assert reversed_names == [row.name for row in report.rows]


def test_dominated_chain_picks_the_closer_dominator() -> None:
    """A dominates B dominates C. C's closest dominator is B, not A.

    Oriented so larger is better, then min-max scaled, the points are
    A=(1, 1, 1), B=(0.5, 0.5, 0.5), C=(0, 0, 0). Distance B–C is
    sqrt(0.75), which is less than distance A–C of sqrt(3).
    """
    first = _point("A", 1.0, 1.0, 1.0)
    middle = _point("B", 0.5, 2.0, 2.0)
    last = _point("C", 0.0, 3.0, 3.0)
    assert dominates(first, middle, AXES3)
    assert dominates(middle, last, AXES3)
    assert dominates(first, last, AXES3)
    assert not dominates(middle, first, AXES3)

    report = pareto_frontier([last, middle, first])
    by_name = {row.name: row for row in report.rows}
    assert by_name["A"].on_frontier
    assert not by_name["B"].on_frontier
    assert by_name["B"].dominated_by == "A"
    assert not by_name["C"].on_frontier
    assert by_name["C"].dominated_by == "B"
    assert [row.name for row in report.rows] == ["A", "B", "C"]


def test_exact_ties_stay_on_the_frontier_and_break_by_name() -> None:
    alpha = _point("alpha", 0.8, 1.0, 10.0)
    beta = _point("beta", 0.8, 1.0, 10.0)
    gamma = _point("gamma", 0.2, 4.0, 40.0)
    assert not dominates(alpha, beta, AXES3)
    assert not dominates(beta, alpha, AXES3)
    assert dominates(alpha, gamma, AXES3)
    assert dominates(beta, gamma, AXES3)

    report = pareto_frontier([gamma, beta, alpha])
    by_name = {row.name: row for row in report.rows}
    assert by_name["alpha"].on_frontier
    assert by_name["beta"].on_frontier
    assert not by_name["gamma"].on_frontier
    # Both dominators sit on the same coordinates, so the distances match.
    # "alpha" sorts before "beta".
    assert by_name["gamma"].dominated_by == "alpha"
    assert [row.name for row in report.rows] == ["alpha", "beta", "gamma"]


def test_any_strict_difference_counts() -> None:
    better = _point("better", 1.0, 1.0, 1.0)
    worse = _point("worse", 1.0 - 1e-12, 1.0, 1.0)
    report = pareto_frontier([worse, better])
    by_name = {row.name: row for row in report.rows}
    assert by_name["better"].on_frontier
    assert not by_name["worse"].on_frontier
    assert by_name["worse"].dominated_by == "better"


def test_single_variant_is_the_frontier() -> None:
    report = pareto_frontier([_point("only", 0.3, 1.5, 7.0)])
    assert report.axes == AXES3
    assert len(report.rows) == 1
    row = report.rows[0]
    assert row.name == "only"
    assert row.on_frontier
    assert row.dominated_by is None
    text = format_pareto(report)
    assert text.startswith("pareto\n")
    assert "axes=score,cost,latency\n" in text
    assert "omitted=none\n" in text
    assert "n=1\n" in text
    data_line = next(line for line in text.splitlines() if line.startswith("only"))
    assert data_line.split()[4:] == ["yes", "-"]


def test_two_axis_fallback_when_cost_is_missing() -> None:
    good = _point("good", 0.9, None, 5.0)
    fast = _point("fast", 0.5, None, 1.0)
    weak = _point("weak", 0.5, None, 5.0)
    assert not dominates(good, fast, ("score", "latency"))
    assert dominates(good, weak, ("score", "latency"))
    assert dominates(fast, weak, ("score", "latency"))

    report = pareto_frontier([weak, good, fast])
    assert report.axes == ("score", "latency")
    assert report.omitted == ("cost",)
    by_name = {row.name: row for row in report.rows}
    assert by_name["good"].on_frontier
    assert by_name["fast"].on_frontier
    assert not by_name["weak"].on_frontier
    # good and fast are the same distance from weak after scaling.
    # "fast" sorts before "good".
    assert by_name["weak"].dominated_by == "fast"
    assert [row.name for row in report.rows] == ["good", "fast", "weak"]
    text = format_pareto(report)
    assert "omitted=cost\n" in text
    assert "axes=score,latency\n" in text
    weak_line = next(line for line in text.splitlines() if line.startswith("weak"))
    assert weak_line.split()[2] == "-"


def test_two_axis_fallback_when_latency_is_missing() -> None:
    pricey = _point("pricey", 0.9, 5.0, None)
    thrifty = _point("thrifty", 0.4, 1.0, None)
    # Equal score to thrifty, higher cost. thrifty dominates it. pricey does
    # not: its cost is higher, so the cost axis is what keeps thrifty alive.
    weak = _point("weak", 0.4, 4.0, None)
    report = pareto_frontier([weak, pricey, thrifty])
    assert report.axes == ("score", "cost")
    assert report.omitted == ("latency",)
    by_name = {row.name: row for row in report.rows}
    assert by_name["pricey"].on_frontier
    assert by_name["thrifty"].on_frontier
    assert not by_name["weak"].on_frontier
    assert by_name["weak"].dominated_by == "thrifty"
    text = format_pareto(report)
    assert "omitted=latency\n" in text
    weak_line = next(line for line in text.splitlines() if line.startswith("weak"))
    assert weak_line.split()[3] == "-"


def test_validation_errors_name_the_axis() -> None:
    with pytest.raises(ValueError, match="mean score must be a finite number"):
        pareto_frontier([_point("nan-score", math.nan, 1.0, 1.0)])
    with pytest.raises(ValueError, match="variant 'inf-cost': cost must be a finite number"):
        pareto_frontier([_point("inf-cost", 1.0, math.inf, 1.0)])
    with pytest.raises(ValueError, match="mean latency must be a finite number"):
        pareto_frontier([_point("nan-latency", 1.0, 1.0, math.nan)])
    with pytest.raises(ValueError, match="mean score must be a finite number"):
        pareto_frontier([VariantPoint("flag", True, 1.0, 1.0)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pareto needs at least one variant"):
        pareto_frontier([])
    with pytest.raises(ValueError, match="variant name is empty"):
        pareto_frontier([_point("", 1.0, 1.0, 1.0, source="memo")])

    mixed_cost = [_point("priced", 1.0, 1.0, 1.0), _point("bare", 0.5, None, 1.0)]
    with pytest.raises(ValueError, match="missing cost on variant\\(s\\) 'bare'") as cost_error:
        pareto_frontier(mixed_cost)
    assert "other variants have a cost" in str(cost_error.value)

    mixed_latency = [_point("timed", 1.0, 1.0, 1.0), _point("untimed", 0.5, 1.0, None)]
    with pytest.raises(ValueError, match="missing latency on variant\\(s\\) 'untimed'"):
        pareto_frontier(mixed_latency)

    with pytest.raises(ValueError, match="missing cost and latency on every variant"):
        pareto_frontier([_point("empty", 1.0, None, None)])

    with pytest.raises(ValueError, match="duplicate variant name 'a'") as duplicate:
        pareto_frontier(
            [
                _point("a", 1.0, 1.0, 1.0, source="one.json"),
                _point("a", 0.0, 2.0, 2.0, source="two.json"),
            ]
        )
    message = str(duplicate.value)
    assert "one.json" in message
    assert "two.json" in message


def test_signed_zero_latency_prints_as_zero() -> None:
    report = pareto_frontier([_point("zero", 1.0, 0.0, -0.0)])
    assert report.rows[0].latency_ms == 0.0
    assert math.copysign(1.0, report.rows[0].latency_ms) == 1.0
    text = format_pareto(report)
    assert "0.000000" in text
    assert "-0.000000" not in text


def test_extraction_from_written_comparison_and_run(tmp_path: Path) -> None:
    cheap = _run(
        "cheap",
        [1.0, 0.0],
        [10.0, 30.0],
        CostEstimate(model="small", usd=0.01),
    )
    quality = _run(
        "quality",
        [1.0, 1.0],
        [40.0, 60.0],
        CostEstimate(model="large", usd=0.05),
    )
    slow = _run(
        "slow",
        [0.0, 0.0],
        [80.0, 80.0],
        CostEstimate(model="mid", usd=0.04),
    )
    comparison_path = tmp_path / "comparison.json"
    run_path = tmp_path / "slow.json"
    write_json(compare_runs(cheap, quality), comparison_path)
    write_run_json(slow, run_path)

    points = load_variants([comparison_path, run_path])
    by_source = {point.name: point for point in points}
    assert set(by_source) == {"cheap", "quality", "slow"}
    assert by_source["cheap"].score == cheap.mean_score == 0.5
    assert by_source["cheap"].latency_ms == cheap.mean_latency_ms == 20.0
    assert by_source["cheap"].cost == 0.01
    assert by_source["quality"].score == quality.mean_score == 1.0
    assert by_source["quality"].latency_ms == quality.mean_latency_ms == 50.0
    assert by_source["quality"].cost == 0.05
    assert by_source["slow"].score == 0.0
    assert by_source["slow"].latency_ms == 80.0
    assert by_source["slow"].cost == 0.04
    assert "(baseline)" in by_source["cheap"].source
    assert "(candidate)" in by_source["quality"].source

    report = pareto_frontier(points)
    by_name = {row.name: row for row in report.rows}
    assert by_name["quality"].on_frontier
    assert by_name["cheap"].on_frontier
    assert not by_name["slow"].on_frontier
    assert by_name["slow"].dominated_by == "cheap"
    assert [row.name for row in report.rows] == ["quality", "cheap", "slow"]

    payload = pareto_to_dict(report)
    assert payload["axes"] == ["score", "cost", "latency"]
    assert payload["omitted"] == []
    variants = payload["variants"]
    assert isinstance(variants, list)
    assert variants[2]["dominated_by"] == "cheap"
    assert variants[2]["on_frontier"] is False
    assert variants[0]["latency_ms"] == 50.0


def test_written_runs_without_cost_omit_that_axis(tmp_path: Path) -> None:
    fast = _run("fast", [1.0, 0.0], [5.0, 5.0])
    slow = _run("slow", [0.0, 0.0], [9.0, 9.0])
    fast_path = tmp_path / "fast.json"
    slow_path = tmp_path / "slow.json"
    write_run_json(fast, fast_path)
    write_run_json(slow, slow_path)
    report = pareto_from_files([slow_path, fast_path])
    assert report.axes == ("score", "latency")
    assert report.omitted == ("cost",)
    by_name = {row.name: row for row in report.rows}
    assert by_name["fast"].on_frontier
    assert by_name["fast"].cost is None
    assert not by_name["slow"].on_frontier
    assert by_name["slow"].dominated_by == "fast"
    assert by_name["fast"].score == fast.mean_score == 0.5
    assert by_name["fast"].latency_ms == fast.mean_latency_ms == 5.0


def test_null_cost_usd_is_missing_cost(tmp_path: Path) -> None:
    priced = _run("priced", [1.0], [2.0], CostEstimate(model="known", usd=0.2))
    unknown = _run(
        "unknown",
        [0.0],
        [2.0],
        CostEstimate(model="mystery", usd=None, note="unknown model 'mystery'"),
    )
    priced_path = tmp_path / "priced.json"
    unknown_path = tmp_path / "unknown.json"
    write_run_json(priced, priced_path)
    write_run_json(unknown, unknown_path)
    with pytest.raises(ValueError, match="missing cost on variant\\(s\\) 'unknown'"):
        pareto_from_files([priced_path, unknown_path])


def test_non_finite_score_in_a_run_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    write_run_json(_run("bad", [1.0], [1.0], CostEstimate(model="m", usd=0.1)), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"][0]["score"] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="score must be a finite number"):
        load_variants([path])


def test_duplicate_names_across_files(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_run_json(_run("same", [1.0], [1.0], CostEstimate(model="m", usd=0.1)), first)
    write_run_json(_run("same", [0.0], [2.0], CostEstimate(model="m", usd=0.2)), second)
    with pytest.raises(ValueError, match="duplicate variant name 'same'") as caught:
        pareto_from_files([first, second])
    message = str(caught.value)
    assert "first.json" in message
    assert "second.json" in message


def test_matrix_ranking_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "matrix.json"
    write_matrix_json(
        MatrixResult(
            suite_name="suite",
            best_name="a",
            rows=[
                MatrixRow(1, "a", "mock", "m", "{input}", 1.0, 1.0, 0.0, 0.0, 0.0),
                MatrixRow(2, "b", "mock", "m", "{input}", 0.5, 0.5, -0.5, -0.5, 0.0),
            ],
            n_boot=2000,
            seed=13,
            alpha=0.05,
            ci_method="percentile",
        ),
        path,
    )
    with pytest.raises(ValueError, match="matrix ranking JSON") as caught:
        load_variants([path])
    message = str(caught.value)
    assert "cost" in message
    assert "latency" in message


def test_loader_rejects_bad_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="Report not found"):
        load_variants([missing])

    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="file is empty"):
        load_variants([empty])

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_variants([invalid])

    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        load_variants([binary])

    bom = tmp_path / "bom.json"
    write_run_json(_run("bom", [1.0], [3.0], CostEstimate(model="m", usd=0.1)), bom)
    bom.write_bytes(b"\xef\xbb\xbf" + bom.read_bytes())
    points = load_variants([bom])
    assert points[0].name == "bom"
    assert points[0].score == 1.0

    history = tmp_path / "history.jsonl"
    history.write_text(
        json.dumps({"command": "run", "metrics": {"mean_score": 1.0}})
        + "\n"
        + json.dumps({"command": "compare", "metrics": {"mean_score": 0.5}})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="history JSONL"):
        load_variants([history])


def test_shipped_example_is_the_documented_frontier() -> None:
    """examples/pareto is the comparison and the run the README command reads."""
    report = pareto_from_files(
        [
            ROOT / "examples" / "pareto" / "comparison.json",
            ROOT / "examples" / "pareto" / "slow.json",
        ]
    )
    assert report.axes == AXES3
    assert report.omitted == ()
    assert [
        (row.name, row.on_frontier, row.dominated_by, row.score, row.cost, row.latency_ms)
        for row in report.rows
    ] == [
        ("quality", True, None, 1.0, 0.05, 50.0),
        ("cheap", True, None, 0.5, 0.01, 20.0),
        ("slow", False, "cheap", 0.0, 0.04, 80.0),
    ]


def test_cli_table_json_and_out(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cheap = _run("cheap", [1.0, 0.0], [10.0, 30.0], CostEstimate(model="small", usd=0.01))
    quality = _run("quality", [1.0, 1.0], [40.0, 60.0], CostEstimate(model="large", usd=0.05))
    slow = _run("slow", [0.0, 0.0], [80.0, 80.0], CostEstimate(model="mid", usd=0.04))
    comparison_path = tmp_path / "comparison.json"
    run_path = tmp_path / "slow.json"
    write_json(compare_runs(cheap, quality), comparison_path)
    write_run_json(slow, run_path)

    code = main(["pareto", str(comparison_path), str(run_path)])
    assert code == 0
    text = capsys.readouterr().out
    assert text.startswith("pareto\n")
    assert "does not change the gate" in text
    assert "strictly better on one" in text
    data = [line for line in text.splitlines() if line.startswith(("quality", "cheap", "slow"))]
    assert [line.split()[0] for line in data] == ["quality", "cheap", "slow"]
    assert data[0].split()[4:] == ["yes", "-"]
    assert data[1].split()[4:] == ["yes", "-"]
    assert data[2].split()[4:] == ["no", "cheap"]
    assert "variant" in text
    assert "frontier" in text
    assert "dominated_by" in text

    out = tmp_path / "nested" / "pareto.json"
    code = main(["pareto", str(comparison_path), str(run_path), "--json", "--out", str(out)])
    assert code == 0
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert payload["n"] == 3
    assert payload["variants"][2]["name"] == "slow"
    assert payload["variants"][2]["dominated_by"] == "cheap"
    assert payload["variants"][2]["on_frontier"] is False
    assert payload["note"] == PARETO_NOTE

    code = main(["pareto", str(run_path), "--out", str(tmp_path / "table-out.json")])
    assert code == 0
    table = capsys.readouterr().out
    assert table.startswith("pareto\n")
    written = json.loads((tmp_path / "table-out.json").read_text(encoding="utf-8"))
    assert written["n"] == 1
    assert written["variants"][0]["name"] == "slow"
    assert written["variants"][0]["on_frontier"] is True


def test_cli_errors_and_help(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["pareto", str(tmp_path / "missing.json")])
    assert code == 2
    missing = capsys.readouterr()
    assert "error:" in missing.err
    assert "Report not found" in missing.err

    matrix = tmp_path / "matrix.json"
    write_matrix_json(
        MatrixResult(
            suite_name="suite",
            best_name="a",
            rows=[
                MatrixRow(1, "a", "mock", "m", "{input}", 1.0, 1.0, 0.0, 0.0, 0.0),
                MatrixRow(2, "b", "mock", "m", "{input}", 0.0, 0.0, -1.0, -1.0, 0.0),
            ],
            n_boot=10,
            seed=13,
            alpha=0.05,
            ci_method="percentile",
        ),
        matrix,
    )
    code = main(["pareto", str(matrix)])
    assert code == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "cost" in err
    assert "latency" in err

    with pytest.raises(SystemExit) as help_exit:
        main(["pareto", "--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "--json" in help_text
    assert "--out" in help_text
    assert "frontier" in help_text
    assert "run --out" in help_text

    with pytest.raises(SystemExit) as missing_args:
        main(["pareto"])
    assert missing_args.value.code == 2
