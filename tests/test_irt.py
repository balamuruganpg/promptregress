"""2PL item-response fit on a planted easy/hard matrix."""

import json
from pathlib import Path

import pytest

from helpers import case_result
from promptregress.cli import main
from promptregress.irt import fit_2pl, load_run_report
from promptregress.models import RunResult
from promptregress.report import write_run_json


def _matrix() -> tuple[list[str], list[list[float]]]:
    # Three items both models pass, two only the strong model passes, one neither passes.
    case_ids = ["e1", "e2", "e3", "m1", "m2", "h1"]
    strong = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    weak = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    return case_ids, [strong, weak]


def test_irt_recovers_difficulty_order_and_ranks_the_stronger_model() -> None:
    case_ids, matrix = _matrix()
    result = fit_2pl(matrix, case_ids=case_ids, model_names=["strong", "weak"], seed=13)
    by_id = {item.case_id: item for item in result.cases}
    easy = [by_id[case_id].b for case_id in ("e1", "e2", "e3")]
    medium = [by_id[case_id].b for case_id in ("m1", "m2")]
    assert max(easy) < min(medium)
    assert max(medium) < by_id["h1"].b
    assert result.abilities[0].model == "strong"
    assert result.abilities[0].rank == 1
    assert result.abilities[0].theta > result.abilities[1].theta
    for item in result.cases:
        assert item.low_discrimination == (item.a < result.min_discrimination)
        assert item.very_hard == (item.b >= result.hard_threshold)
        assert item.very_easy == (item.b <= result.easy_threshold)
    again = fit_2pl(matrix, case_ids=case_ids, model_names=["strong", "weak"], seed=13)
    assert [item.b for item in again.cases] == [item.b for item in result.cases]
    assert [item.theta for item in again.abilities] == [item.theta for item in result.abilities]


def test_irt_cli_reads_run_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    case_ids, matrix = _matrix()

    def build(name: str, row: list[float]) -> RunResult:
        return RunResult(
            suite_name="planted",
            config_name=name,
            results=[
                case_result(case_id, score, score >= 0.5)
                for case_id, score in zip(case_ids, row, strict=True)
            ],
        )

    strong_path = tmp_path / "strong.json"
    weak_path = tmp_path / "weak.json"
    write_run_json(build("strong", matrix[0]), strong_path)
    write_run_json(build("weak", matrix[1]), weak_path)
    loaded = load_run_report(strong_path)
    assert loaded.config_name == "strong"
    assert loaded.results[0].case_id == "e1"
    out = tmp_path / "irt.json"
    code = main(["irt", str(strong_path), str(weak_path), "--out", str(out), "--seed", "13"])
    assert code == 0
    text = capsys.readouterr().out
    assert "| 1 | strong |" in text
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["abilities"][0]["model"] == "strong"
    assert payload["abilities"][0]["rank"] == 1
    difficulties = {item["case_id"]: item["b"] for item in payload["cases"]}
    assert max(difficulties[case_id] for case_id in ("e1", "e2", "e3")) < min(
        difficulties[case_id] for case_id in ("m1", "m2")
    )
