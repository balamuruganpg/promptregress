"""Split-conformal false-pass threshold: known answers, edges, CLI, simulation."""

import json
import math
import random
from pathlib import Path

import pytest

from promptregress.calibration import CalibrationExample, calibrate_examples
from promptregress.cli import main
from promptregress.conformal import (
    calibrate_threshold,
    conformal_to_dict,
    evaluate_threshold,
    predict_pass,
)
from promptregress.providers import MockProvider


def test_known_order_statistic_on_five_fail_scores() -> None:
    """Hand-computed split-conformal quantile.

    Pairs (score, label)::

        (0.10, 0), (0.20, 0), (0.30, 0), (0.40, 0), (0.80, 0),
        (0.90, 1), (1.00, 1)

    Gold-fail scores, sorted: 0.10, 0.20, 0.30, 0.40, 0.80. n = 5.
    alpha = 0.25. (n+1)(1-alpha) = 6 * 0.75 = 4.5, so
    k = ceil(4.5) = 5. The 5th order statistic is 0.80.
    guaranteed_bound = floor(0.25 * 6) / 6 = 1/6.
    Empirical false-pass: only 0.80 is >= 0.80, so 1/5.
    Pass-class rows do not enter the quantile.
    """
    scores = [0.10, 0.20, 0.30, 0.40, 0.80, 0.90, 1.00]
    labels = [0, 0, 0, 0, 0, 1, 1]
    result = calibrate_threshold(scores, labels, alpha=0.25)
    assert result.n == 5
    assert result.alpha == 0.25
    assert result.threshold == pytest.approx(0.80)
    assert result.guaranteed_bound == pytest.approx(1.0 / 6.0)
    assert result.empirical_error_rate == pytest.approx(0.2)
    assert predict_pass(0.80, result.threshold) is True
    assert predict_pass(0.79, result.threshold) is False
    assert predict_pass(0.90, result.threshold) is True


def test_n1_all_fail_and_all_pass() -> None:
    # n=1 fail, alpha=0.1: (1+1)(0.9)=1.8, k=ceil(1.8)=2 > 1 → +inf.
    # bound = floor(0.2)/2 = 0. Empirical error 0 (never pass).
    tiny = calibrate_threshold([0.4], [0], alpha=0.1)
    assert tiny.n == 1
    assert math.isinf(tiny.threshold) and tiny.threshold > 0
    assert tiny.guaranteed_bound == pytest.approx(0.0)
    assert tiny.empirical_error_rate == pytest.approx(0.0)
    assert predict_pass(1.0, tiny.threshold) is False

    # n=1 fail, alpha=0.5: k=ceil(2*0.5)=1, threshold is the one score.
    # The calibration fail itself passes (>=). bound = floor(1)/2 = 0.5.
    half = calibrate_threshold([0.4], [0], alpha=0.5)
    assert half.threshold == pytest.approx(0.4)
    assert half.guaranteed_bound == pytest.approx(0.5)
    assert half.empirical_error_rate == pytest.approx(1.0)

    # All-pass: n=0 fail scores, k=1 > 0 → +inf, empirical error 0.
    all_pass = calibrate_threshold([0.9, 1.0], [1, 1], alpha=0.1)
    assert all_pass.n == 0
    assert math.isinf(all_pass.threshold) and all_pass.threshold > 0
    assert all_pass.guaranteed_bound == pytest.approx(0.0)
    assert all_pass.empirical_error_rate == pytest.approx(0.0)

    # All-fail, n=3, alpha=0.5: k=ceil(4*0.5)=2, threshold=2nd=0.5.
    # bound = floor(2)/4 = 0.5. Scores >= 0.5: 0.5 and 0.9 → 2/3.
    all_fail = calibrate_threshold([0.1, 0.5, 0.9], [0, 0, 0], alpha=0.5)
    assert all_fail.n == 3
    assert all_fail.threshold == pytest.approx(0.5)
    assert all_fail.guaranteed_bound == pytest.approx(0.5)
    assert all_fail.empirical_error_rate == pytest.approx(2.0 / 3.0)


def test_ties_at_the_threshold_count_as_pass() -> None:
    """Three gold-fail scores sit on the k-th order statistic.

    Sorted fail scores: 0.20, 0.40, 0.40, 0.40, 0.90. n = 5, alpha = 0.5.
    (n+1)(1-alpha) = 6 * 0.5 = 3, k = 3, threshold = 0.40.
    Pass if score >= 0.40: four of the five fails pass (the three ties
    and 0.90). Empirical error 4/5. bound = floor(3)/6 = 0.5.
    """
    scores = [0.20, 0.40, 0.40, 0.40, 0.90]
    labels = [0, 0, 0, 0, 0]
    result = calibrate_threshold(scores, labels, alpha=0.5)
    assert result.threshold == pytest.approx(0.40)
    assert result.empirical_error_rate == pytest.approx(0.8)
    assert result.guaranteed_bound == pytest.approx(0.5)
    assert predict_pass(0.40, result.threshold) is True
    assert predict_pass(0.399, result.threshold) is False
    held = evaluate_threshold(scores, labels, result.threshold)
    assert held.n_false_pass == 4
    assert held.false_pass_rate == pytest.approx(0.8)
    assert held.coverage == pytest.approx(0.2)


def test_infinite_threshold_serializes_as_json_null() -> None:
    result = calibrate_threshold([0.4], [0], alpha=0.1)
    payload = conformal_to_dict(result)
    assert payload["threshold"] is None
    dumped = json.dumps(payload)
    assert "Infinity" not in dumped
    assert json.loads(dumped)["threshold"] is None


def test_evaluate_threshold_on_held_out_pairs() -> None:
    # Threshold 0.5: fail scores 0.2 (correct fail) and 0.8 (false pass);
    # pass scores 0.9 (correct pass) and 0.1 (false fail).
    evaluation = evaluate_threshold(
        [0.2, 0.8, 0.9, 0.1],
        [0, 0, 1, 1],
        0.5,
    )
    assert evaluation.n == 4
    assert evaluation.n_fail == 2
    assert evaluation.n_false_pass == 1
    assert evaluation.n_false_fail == 1
    assert evaluation.false_pass_rate == pytest.approx(0.5)
    assert evaluation.coverage == pytest.approx(0.5)


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="same length"):
        calibrate_threshold([0.1, 0.2], [0], 0.1)
    with pytest.raises(ValueError, match="at least one pair"):
        calibrate_threshold([], [], 0.1)
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        calibrate_threshold([0.1], [0], 0.0)
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        calibrate_threshold([0.1], [0], 1.0)
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        calibrate_threshold([0.1], [0], math.nan)
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        calibrate_threshold([0.1], [0], math.inf)
    with pytest.raises(ValueError, match="finite numbers in \\[0, 1\\]"):
        calibrate_threshold([math.nan], [0], 0.1)
    with pytest.raises(ValueError, match="finite numbers in \\[0, 1\\]"):
        calibrate_threshold([math.inf], [0], 0.1)
    with pytest.raises(ValueError, match="finite numbers in \\[0, 1\\]"):
        calibrate_threshold([1.2], [0], 0.1)
    with pytest.raises(ValueError, match="finite numbers in \\[0, 1\\]"):
        calibrate_threshold([-0.1], [0], 0.1)
    with pytest.raises(ValueError, match="labels must be 0 or 1"):
        calibrate_threshold([0.1], [2], 0.1)
    with pytest.raises(ValueError, match="labels must be 0 or 1"):
        calibrate_threshold([0.1], [-1], 0.1)
    with pytest.raises(ValueError, match="labels must be 0 or 1"):
        calibrate_threshold([0.1], [True], 0.1)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="labels must be 0 or 1"):
        calibrate_threshold([0.1], [0.0], 0.1)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="finite numbers in \\[0, 1\\]"):
        predict_pass(math.nan, 0.5)
    with pytest.raises(ValueError, match="threshold must be a finite number or \\+inf"):
        predict_pass(0.5, math.nan)
    with pytest.raises(ValueError, match="threshold must be a finite number or \\+inf"):
        predict_pass(0.5, -math.inf)
    with pytest.raises(ValueError, match="at least one pair"):
        evaluate_threshold([], [], 0.5)


def test_seeded_simulation_false_pass_rate_respects_alpha() -> None:
    """Uniform gold-fail scores, split conformal / test, several seeds.

    For each seed, 300 calibration fails and 300 test fails are drawn from
    U(0, 1) with ``random.Random(seed)``. The threshold is fit at alpha=0.1
    on the calibration split. Because the scores are continuous, ties have
    probability 0 and the finite-sample bound is P(s_new > t) <= 0.1.

    Each seed's test false-pass rate is asserted <= 0.1 + 0.07. The mean
    across seeds is asserted <= 0.1 + 0.03. Those tolerances cover binomial
    noise on 300 test points (se ≈ 0.017) without loosening the method.
    """
    alpha = 0.1
    n_cal = 300
    n_test = 300
    seeds = (1, 2, 3, 5, 8, 13, 21, 34)
    rates: list[float] = []
    for seed in seeds:
        rng = random.Random(seed)
        cal_scores = [rng.random() for _ in range(n_cal)]
        test_scores = [rng.random() for _ in range(n_test)]
        fitted = calibrate_threshold(cal_scores, [0] * n_cal, alpha)
        held = evaluate_threshold(test_scores, [0] * n_test, fitted.threshold)
        assert held.false_pass_rate <= alpha + 0.07
        assert fitted.guaranteed_bound <= alpha + 1e-12
        rates.append(held.false_pass_rate)
    assert sum(rates) / len(rates) <= alpha + 0.03


def test_calibrate_cli_prints_conformal_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "labels.jsonl"
    rows = [
        {"input": "q1", "output": "SCORE_A", "human_score": 0.0},
        {"input": "q2", "output": "SCORE_B", "human_score": 0.0},
        {"input": "q3", "output": "SCORE_C", "human_score": 0.0},
        {"input": "q4", "output": "SCORE_D", "human_score": 0.0},
        {"input": "q5", "output": "SCORE_E", "human_score": 1.0},
        {"input": "q6", "output": "SCORE_F", "human_score": 1.0},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    out = tmp_path / "calibration.json"
    code = main(
        [
            "calibrate",
            str(path),
            "--rubric",
            "Score the answer.",
            "--judge-provider",
            "mock",
            "--mock-rules",
            json.dumps(
                [
                    ["SCORE_A", "0.1"],
                    ["SCORE_B", "0.2"],
                    ["SCORE_C", "0.3"],
                    ["SCORE_D", "0.8"],
                    ["SCORE_E", "0.9"],
                    ["SCORE_F", "1"],
                ]
            ),
            "--conformal-alpha",
            "0.25",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "conformal" in text
    assert "alpha=0.250000" in text
    assert "threshold=0.800000" in text
    assert "guaranteed_bound=0.200000" in text
    assert "empirical_error=0.250000" in text
    payload = json.loads(out.read_text(encoding="utf-8"))
    conformal = payload["conformal"]
    assert conformal["n"] == 4
    assert conformal["alpha"] == pytest.approx(0.25)
    assert conformal["threshold"] == pytest.approx(0.8)
    assert conformal["guaranteed_bound"] == pytest.approx(0.2)
    assert conformal["empirical_error_rate"] == pytest.approx(0.25)


def test_calibrate_help_mentions_conformal_alpha(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as help_exit:
        main(["calibrate", "--help"])
    assert help_exit.value.code == 0
    text = capsys.readouterr().out
    assert "--conformal-alpha" in text
    assert "false-pass" in text


def test_calibrate_examples_forwards_conformal_alpha() -> None:
    examples = [
        CalibrationExample(input="q", output="LOW", human_score=0.0),
        CalibrationExample(input="q", output="HIGH", human_score=1.0),
    ]
    report = calibrate_examples(
        examples,
        MockProvider(rules=[("LOW", "0.2"), ("HIGH", "0.9")], default="0"),
        "Score the answer.",
        conformal_alpha=0.5,
    )
    assert report.conformal is not None
    assert report.conformal.n == 1
    assert report.conformal.threshold == pytest.approx(0.2)
