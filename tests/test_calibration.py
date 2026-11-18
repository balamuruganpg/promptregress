"""Judge calibration, Cohen's kappa, and bias audits."""

import json
import math
from pathlib import Path

import pytest

from promptregress.calibration import (
    CalibrationExample,
    calibrate,
    calibrate_examples,
    cohens_kappa,
    load_calibration_jsonl,
    pearson,
    position_bias_audit,
    spearman,
    verbosity_bias,
)
from promptregress.cli import main
from promptregress.providers import MockProvider


def test_cohens_kappa_matches_a_hand_computed_table() -> None:
    # Judge yes/no by human yes/no: 20, 5 / 10, 15.
    # p_o = 35/50 = 0.7, p_e = 0.5 * 0.6 + 0.5 * 0.4 = 0.5, kappa = 0.4.
    judge = [1.0] * 20 + [1.0] * 5 + [0.0] * 10 + [0.0] * 15
    human = [1.0] * 20 + [0.0] * 5 + [1.0] * 10 + [0.0] * 15
    report = calibrate(judge, human, threshold=0.5)
    assert report.both_pass == 20
    assert report.judge_only == 5
    assert report.human_only == 10
    assert report.both_fail == 15
    assert report.raw_agreement == pytest.approx(0.7)
    assert report.kappa == pytest.approx(0.4)
    assert report.mae == pytest.approx(15 / 50)
    kappa, agreement = cohens_kappa(
        [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15,
        [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15,
    )
    assert kappa == pytest.approx(0.4)
    assert agreement == pytest.approx(0.7)


def test_spearman_and_pearson_known_values() -> None:
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    # Ranks (1, 2, 3, 4) and (1, 3, 2, 4): 1 - 6*2/(4*15) = 0.8.
    assert spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [1, 2, 2]) == pytest.approx(math.sqrt(3) / 2)
    assert pearson([1, 1, 1], [1, 2, 3]) is None


def test_position_bias_flags_an_always_first_judge() -> None:
    pairs = [("alpha output", "beta output"), ("cat", "dog"), ("left side", "right side")]
    seen: list[tuple[str, str]] = []

    def always_first(first: str, second: str) -> str:
        seen.append((first, second))
        return "first"

    rigged = position_bias_audit(pairs, always_first)
    assert rigged.flip_rate == pytest.approx(1.0)
    assert rigged.first_position_win_rate == pytest.approx(1.0)
    assert rigged.flagged is True
    assert seen[0] == ("alpha output", "beta output")
    assert seen[1] == ("beta output", "alpha output")
    assert len(seen) == 6

    def consistent(first: str, second: str) -> str:
        if first < second:
            return "first"
        if second < first:
            return "second"
        return "tie"

    fair = position_bias_audit(pairs, consistent)
    assert fair.flip_rate == 0.0
    assert fair.n_flips == 0
    assert fair.flagged is False


def test_verbosity_correlation_tracks_length() -> None:
    pairs = [("aa", "bbbb"), ("ccccccc", "d")]

    def prefer_longer(first: str, second: str) -> str:
        if len(first) >= len(second):
            return "first"
        return "second"

    longer = verbosity_bias(pairs, prefer_longer)
    assert longer.length_diffs == [-2.0, 6.0]
    assert longer.preferences == [-1.0, 1.0]
    assert longer.correlation == pytest.approx(1.0)

    def prefer_shorter(first: str, second: str) -> str:
        if len(first) <= len(second):
            return "first"
        return "second"

    shorter = verbosity_bias(pairs, prefer_shorter)
    assert shorter.correlation == pytest.approx(-1.0)


def test_calibrate_cli_uses_mock_judge(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "good",
                        "input": "question one",
                        "reference": "reference one",
                        "output": "OUTCOME_GOOD",
                        "human_score": 1,
                    }
                ),
                json.dumps(
                    {
                        "input": "question two",
                        "reference": None,
                        "output": "OUTCOME_BAD",
                        "human_score": 0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "calibration.json"
    code = main(
        [
            "calibrate",
            str(path),
            "--rubric",
            "Score the answer.",
            "--judge-provider",
            "mock",
            "--judge-model",
            "scripted",
            "--mock-rules",
            json.dumps([["OUTCOME_GOOD", "1"], ["OUTCOME_BAD", "0"]]),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "pearson=1.000000" in text
    assert "spearman=1.000000" in text
    assert "kappa=1.000000" in text
    assert "| human_score | judge_score |" in text
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n"] == 2
    assert payload["mae"] == 0.0
    assert payload["rows"][0]["judge_score"] == 1.0
    examples = load_calibration_jsonl(path)
    direct = calibrate_examples(
        examples,
        MockProvider(rules=[("OUTCOME_GOOD", "0.25"), ("OUTCOME_BAD", "0.75")], default="0"),
        "Score the answer.",
    )
    assert direct.judge_scores == [0.25, 0.75]
    assert direct.mae == pytest.approx((0.75 + 0.75) / 2.0)


def test_calibration_example_type_is_exported_for_callers() -> None:
    example = CalibrationExample(input="q", output="a", human_score=0.5, reference="r")
    assert example.example_id == ""
