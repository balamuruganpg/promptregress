"""Learned regression-risk model: separable cases, refusal, and the CLI."""

import json
import math
from pathlib import Path

import pytest

from promptregress.cli import main
from promptregress.gate import evaluate_gate
from promptregress.models import CaseResult, Comparison, RunResult
from promptregress.report import comparison_to_dict
from promptregress.risk import (
    FEATURE_NAMES,
    MIN_LABELED_CASES,
    RiskRefusal,
    case_features,
    labeled_cases,
    load_comparison,
    predict_proba,
    rank_cases,
    regression_label,
    roc_auc,
    train,
)
from promptregress.stats import compare_runs

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "risk"


def make_case_pair(kind: str, index: int, wave: int) -> tuple[CaseResult, CaseResult]:
    """One baseline/candidate pair. ``risk`` cases drop; ``safe`` cases do not."""
    case_id = f"{kind}_{index}"
    if kind == "risk":
        score = 0.22 + 0.02 * index + 0.004 * wave
        latency = 480.0 + 25.0 * index + wave
        case_input = ("refund request " * (5 + index)) + f"w{wave}"
        output = ("partial reply " * (8 + index)) + f"w{wave}"
        tags = ["billing", "slow", "edge"] if index % 2 == 0 else ["billing", "slow"]
        scores = {"levenshtein": score, "contains": score}
        candidate_score = score - 0.2
        baseline_passed = False
        candidate_passed = False
        candidate_output = "shorter reply"
    elif kind == "safe":
        score = 0.90 + 0.005 * index
        latency = 12.0 + float(index)
        case_input = f"store hours {index}"
        output = "Open 9 to 5."
        tags = []
        scores = {"exact_match": score}
        candidate_score = min(1.0, score + 0.01)
        baseline_passed = True
        candidate_passed = True
        candidate_output = "Open 9 to 5 daily."
    else:
        raise AssertionError(kind)
    baseline = CaseResult(
        case_id=case_id,
        output=output,
        scores=scores,
        score=score,
        passed=baseline_passed,
        latency_ms=latency,
        tags=list(tags),
        case_input=case_input,
    )
    candidate = CaseResult(
        case_id=case_id,
        output=candidate_output,
        scores={"exact_match": candidate_score},
        score=candidate_score,
        passed=candidate_passed,
        latency_ms=5.0,
        tags=list(tags),
        case_input=case_input,
    )
    return baseline, candidate


def make_runs(wave: int, *, n_each: int = 6) -> tuple[RunResult, RunResult]:
    baseline_rows: list[CaseResult] = []
    candidate_rows: list[CaseResult] = []
    for index in range(n_each):
        for kind in ("risk", "safe"):
            baseline, candidate = make_case_pair(kind, index, wave)
            baseline_rows.append(baseline)
            candidate_rows.append(candidate)
    return (
        RunResult("risk-suite", f"baseline-{wave}", baseline_rows),
        RunResult("risk-suite", f"candidate-{wave}", candidate_rows),
    )


def make_comparison(wave: int, *, n_each: int = 6) -> Comparison:
    """In-memory comparison. Stats are unused by the risk model."""
    baseline, candidate = make_runs(wave, n_each=n_each)
    return Comparison(
        baseline=baseline,
        candidate=candidate,
        mean_diff=-0.1,
        ci_low=-0.2,
        ci_high=0.0,
        mcnemar_p=1.0,
        wins=0,
        losses=n_each,
        ties=n_each,
        per_case=[],
    )


def write_comparison(comparison: Comparison, path: Path) -> None:
    """Write a real comparison report, keeping ``case_input`` on each case."""
    payload = comparison_to_dict(comparison, evaluate_gate(comparison))
    for side in ("baseline", "candidate"):
        run = comparison.baseline if side == "baseline" else comparison.candidate
        by_id = {case.case_id: case for case in run.results}
        for row in payload[side]["results"]:
            row["case_input"] = by_id[row["case_id"]].case_input
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_example_files(directory: Path) -> None:
    """Four chronological comparisons. The fourth is the ranking target."""
    directory.mkdir(parents=True, exist_ok=True)
    for wave in range(4):
        baseline, candidate = make_runs(wave)
        write_comparison(
            compare_runs(baseline, candidate, seed=13),
            directory / f"cmp_{wave + 1:02d}.json",
        )


def _training() -> list[Comparison]:
    return [make_comparison(wave) for wave in range(3)]


def test_separable_holdout_ranks_planted_cases_first() -> None:
    comparisons = _training()
    model = train(comparisons)
    assert model.n_cases == 36
    assert model.n_train == 24
    assert model.n_holdout == 12
    assert model.holdout_split == "last_comparison"
    assert model.holdout_auc >= 0.9
    target = make_comparison(3)
    ranked = rank_cases(model, target.baseline.results)
    ids = [item.case_id for item in ranked]
    assert all(case_id.startswith("risk_") for case_id in ids[:6])
    assert all(case_id.startswith("safe_") for case_id in ids[6:])
    assert ranked[0].probability > ranked[-1].probability
    probs = predict_proba(model, target.baseline.results)
    by_id = {case.case_id: prob for case, prob in zip(target.baseline.results, probs, strict=True)}
    for item in ranked:
        assert item.probability == by_id[item.case_id]
        assert len(item.contributions) == 3
        for contrib in item.contributions:
            assert contrib.contribution == pytest.approx(
                contrib.coefficient * contrib.standardized_value
            )


def test_single_comparison_holdout_auc() -> None:
    model = train([make_comparison(0, n_each=12)])
    assert model.n_cases == 24
    assert model.holdout_split == "last_cases"
    assert model.holdout_auc >= 0.9
    assert 0 < model.n_holdout < model.n_cases


def test_train_is_deterministic() -> None:
    first = train(_training())
    second = train(_training())
    assert first.coefficients == second.coefficients
    assert first.intercept == second.intercept
    assert first.means == second.means
    assert first.scales == second.scales
    assert first.holdout_auc == second.holdout_auc
    assert all(math.isfinite(value) for value in first.coefficients)


def test_constant_features_do_not_invent_separation() -> None:
    """Identical baseline rows cannot rank one case above another."""

    def pair(case_id: str, candidate_score: float) -> tuple[CaseResult, CaseResult]:
        baseline = CaseResult(
            case_id=case_id,
            output="same",
            scores={"exact_match": 0.5},
            score=0.5,
            passed=True,
            latency_ms=10.0,
            tags=["a"],
            case_input="same input",
        )
        candidate = CaseResult(
            case_id=case_id,
            output="other",
            scores={"exact_match": candidate_score},
            score=candidate_score,
            passed=candidate_score >= 0.5,
            latency_ms=10.0,
            tags=["a"],
            case_input="same input",
        )
        return baseline, candidate

    def comparison(wave: int) -> Comparison:
        baseline_rows: list[CaseResult] = []
        candidate_rows: list[CaseResult] = []
        for index in range(12):
            # Alternate the label so both splits contain both classes.
            dropped = (index + wave) % 2 == 0
            baseline, candidate = pair(f"c{index}", 0.2 if dropped else 0.8)
            baseline_rows.append(baseline)
            candidate_rows.append(candidate)
        return Comparison(
            baseline=RunResult("flat", f"b{wave}", baseline_rows),
            candidate=RunResult("flat", f"c{wave}", candidate_rows),
            mean_diff=0.0,
            ci_low=0.0,
            ci_high=0.0,
            mcnemar_p=1.0,
            wins=0,
            losses=0,
            ties=12,
            per_case=[],
        )

    model = train([comparison(0), comparison(1)])
    assert model.holdout_auc == pytest.approx(0.5)
    ranked = rank_cases(model, comparison(2).baseline.results)
    assert len({item.probability for item in ranked}) == 1
    assert [item.case_id for item in ranked] == sorted(item.case_id for item in ranked)


def test_features_are_baseline_side_only() -> None:
    baseline, candidate = make_case_pair("risk", 1, 0)
    features = case_features(baseline)
    assert len(features) == len(FEATURE_NAMES)
    assert features[FEATURE_NAMES.index("baseline_score")] == pytest.approx(baseline.score)
    latency_index = FEATURE_NAMES.index("baseline_latency_ms")
    assert features[latency_index] == pytest.approx(baseline.latency_ms)
    assert features[FEATURE_NAMES.index("input_length")] == float(len(baseline.case_input))
    assert features[FEATURE_NAMES.index("baseline_output_length")] == float(len(baseline.output))
    assert features[FEATURE_NAMES.index("n_tags")] == float(len(baseline.tags))
    assert features[FEATURE_NAMES.index("scorer_levenshtein")] == 1.0
    assert features[FEATURE_NAMES.index("scorer_exact_match")] == 0.0
    changed = CaseResult(
        case_id=candidate.case_id,
        output="x" * 50,
        scores={"llm_judge": 0.0},
        score=0.0,
        passed=False,
        latency_ms=9999.0,
        tags=["unrelated"],
        case_input="totally different candidate input",
    )
    assert regression_label(baseline.score, candidate.score) == 1
    assert regression_label(baseline.score, baseline.score) == 0
    assert regression_label(baseline.score, baseline.score + 0.1) == 0
    again = case_features(baseline)
    assert again == features
    assert case_features(changed) != features


def test_contributions_match_the_largest_terms() -> None:
    model = train(_training())
    case = make_comparison(3).baseline.results[0]
    ranked = rank_cases(model, [case], top_features=3)[0]
    standardized = [
        (value - mean) / scale
        for value, mean, scale in zip(case_features(case), model.means, model.scales, strict=True)
    ]
    ordered = sorted(
        (
            (name, coefficient * value)
            for name, coefficient, value in zip(
                model.feature_names, model.coefficients, standardized, strict=True
            )
        ),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    assert [contrib.name for contrib in ranked.contributions] == [name for name, _ in ordered[:3]]


def test_refuses_too_few_cases() -> None:
    with pytest.raises(ValueError, match="at least 20") as caught:
        train([make_comparison(0)])
    assert isinstance(caught.value, RiskRefusal)
    assert caught.value.n_cases == 12
    assert caught.value.n_cases < MIN_LABELED_CASES


def test_refuses_single_class() -> None:
    def all_safe(wave: int) -> Comparison:
        baseline_rows: list[CaseResult] = []
        candidate_rows: list[CaseResult] = []
        for index in range(12):
            baseline, candidate = make_case_pair("safe", index, wave)
            baseline_rows.append(baseline)
            candidate_rows.append(candidate)
        return Comparison(
            baseline=RunResult("safe-only", f"b{wave}", baseline_rows),
            candidate=RunResult("safe-only", f"c{wave}", candidate_rows),
            mean_diff=0.0,
            ci_low=0.0,
            ci_high=0.0,
            mcnemar_p=1.0,
            wins=0,
            losses=0,
            ties=12,
            per_case=[],
        )

    with pytest.raises(ValueError, match="both classes") as caught:
        train([all_safe(0), all_safe(1)])
    assert isinstance(caught.value, RiskRefusal)
    assert caught.value.n_cases == 24


def test_refuses_one_class_holdout() -> None:
    """The last comparison is the holdout, so a one-class tail is refused."""
    mixed = make_comparison(0, n_each=10)
    baseline_rows: list[CaseResult] = []
    candidate_rows: list[CaseResult] = []
    for index in range(12):
        baseline, candidate = make_case_pair("safe", index, 1)
        baseline_rows.append(baseline)
        candidate_rows.append(candidate)
    holdout = Comparison(
        baseline=RunResult("risk-suite", "baseline-hold", baseline_rows),
        candidate=RunResult("risk-suite", "candidate-hold", candidate_rows),
        mean_diff=0.0,
        ci_low=0.0,
        ci_high=0.0,
        mcnemar_p=1.0,
        wins=0,
        losses=0,
        ties=12,
        per_case=[],
    )
    with pytest.raises(ValueError, match="holdout") as caught:
        train([mixed, holdout])
    assert isinstance(caught.value, RiskRefusal)


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="at least one comparison"):
        train([])
    empty = Comparison(
        baseline=RunResult("s", "b", []),
        candidate=RunResult("s", "c", []),
        mean_diff=0.0,
        ci_low=0.0,
        ci_high=0.0,
        mcnemar_p=1.0,
        wins=0,
        losses=0,
        ties=0,
        per_case=[],
    )
    with pytest.raises(ValueError, match="no cases"):
        labeled_cases(empty)

    good = make_comparison(0)
    baseline_rows = list(good.baseline.results)
    candidate_rows = list(good.candidate.results[:-1])
    mismatched = Comparison(
        baseline=good.baseline,
        candidate=RunResult("risk-suite", "candidate-0", candidate_rows),
        mean_diff=0.0,
        ci_low=0.0,
        ci_high=0.0,
        mcnemar_p=1.0,
        wins=0,
        losses=0,
        ties=0,
        per_case=[],
    )
    with pytest.raises(ValueError, match="do not match"):
        labeled_cases(mismatched)

    duplicated = Comparison(
        baseline=RunResult("s", "b", [baseline_rows[0], baseline_rows[0]]),
        candidate=RunResult("s", "c", [candidate_rows[0], candidate_rows[0]]),
        mean_diff=0.0,
        ci_low=0.0,
        ci_high=0.0,
        mcnemar_p=1.0,
        wins=0,
        losses=0,
        ties=0,
        per_case=[],
    )
    with pytest.raises(ValueError, match="repeats case id"):
        labeled_cases(duplicated)

    nan_case = CaseResult(
        case_id="nan",
        output="out",
        scores={"exact_match": 1.0},
        score=float("nan"),
        passed=True,
        latency_ms=1.0,
    )
    other = CaseResult(
        case_id="nan",
        output="out",
        scores={"exact_match": 1.0},
        score=1.0,
        passed=True,
        latency_ms=1.0,
    )
    poisoned = Comparison(
        baseline=RunResult("s", "b", [nan_case]),
        candidate=RunResult("s", "c", [other]),
        mean_diff=0.0,
        ci_low=0.0,
        ci_high=0.0,
        mcnemar_p=1.0,
        wins=0,
        losses=0,
        ties=0,
        per_case=[],
    )
    with pytest.raises(ValueError, match="finite"):
        labeled_cases(poisoned)

    inf_case = CaseResult(
        case_id="inf",
        output="out",
        scores={"exact_match": 1.0},
        score=1.0,
        passed=True,
        latency_ms=float("inf"),
    )
    inf_other = CaseResult(
        case_id="inf",
        output="out",
        scores={"exact_match": 1.0},
        score=1.0,
        passed=True,
        latency_ms=1.0,
    )
    with pytest.raises(ValueError, match="finite"):
        case_features(inf_case)
    with pytest.raises(ValueError, match="finite"):
        labeled_cases(
            Comparison(
                baseline=RunResult("s", "b", [inf_case]),
                candidate=RunResult("s", "c", [inf_other]),
                mean_diff=0.0,
                ci_low=0.0,
                ci_high=0.0,
                mcnemar_p=1.0,
                wins=0,
                losses=0,
                ties=0,
                per_case=[],
            )
        )

    model = train(_training())
    with pytest.raises(ValueError, match="at least one"):
        predict_proba(model, [])
    with pytest.raises(ValueError, match="at least one"):
        rank_cases(model, [])
    with pytest.raises(ValueError, match="top_features"):
        rank_cases(model, [good.baseline.results[0]], top_features=0)
    with pytest.raises(ValueError, match="learning_rate"):
        train(_training(), learning_rate=float("nan"))
    with pytest.raises(ValueError, match="iterations"):
        train(_training(), iterations=0)
    with pytest.raises(ValueError, match="l2"):
        train(_training(), l2=-0.1)
    with pytest.raises(ValueError, match="same length"):
        roc_auc([0, 1], [0.2])
    with pytest.raises(ValueError, match="both classes"):
        roc_auc([1, 1], [0.1, 0.9])
    with pytest.raises(ValueError, match="finite"):
        roc_auc([0, 1], [0.1, float("inf")])


def test_roc_auc_rank_statistic() -> None:
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert roc_auc([0, 1], [0.2, 0.2]) == pytest.approx(0.5)
    assert roc_auc([0, 1], [0.1, 0.9]) == pytest.approx(1.0)


def test_label_rows_follow_baseline_order() -> None:
    comparison = make_comparison(0)
    rows = labeled_cases(comparison)
    assert [row.case_id for row in rows] == [case.case_id for case in comparison.baseline.results]
    assert [row.label for row in rows if row.case_id.startswith("risk_")] == [1, 1, 1, 1, 1, 1]
    assert [row.label for row in rows if row.case_id.startswith("safe_")] == [0, 0, 0, 0, 0, 0]


def test_load_comparison_restores_case_input_and_rejects_history(tmp_path: Path) -> None:
    comparison = compare_runs(*make_runs(0), seed=13)
    path = tmp_path / "cmp.json"
    write_comparison(comparison, path)
    loaded = load_comparison(path)
    assert loaded.baseline.results[0].case_input == comparison.baseline.results[0].case_input
    assert loaded.baseline.results[0].case_input != ""
    bom = tmp_path / "bom.json"
    bom.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert load_comparison(bom).baseline.results[0].case_id == loaded.baseline.results[0].case_id
    history = tmp_path / "history.jsonl"
    history.write_text(
        '{"timestamp":"2026-11-12T00:00:00Z","command":"compare","suite":"s",'
        '"metrics":{"mean_score":1}}\n'
        '{"timestamp":"2026-11-12T00:00:01Z","command":"run","suite":"s",'
        '"metrics":{"mean_score":0.5}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="history"):
        load_comparison(history)
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        load_comparison(bad)


def test_cli_example_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    train_files = [EXAMPLES / f"cmp_{index:02d}.json" for index in (1, 2, 3)]
    target = EXAMPLES / "cmp_04.json"
    for path in (*train_files, target):
        assert path.is_file(), path
    out = tmp_path / "risk.json"
    code = main(
        [
            "risk",
            *(str(path) for path in train_files),
            "--target",
            str(target),
            "--top",
            "6",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "n_cases=36" in text
    assert "n_train=24" in text
    assert "n_holdout=12" in text
    assert "holdout_split=last_comparison" in text
    auc_line = next(line for line in text.splitlines() if line.startswith("holdout_auc="))
    assert float(auc_line.split("=", 1)[1]) >= 0.9
    ranked_ids = [line.split()[1] for line in text.splitlines() if line.startswith("- ")]
    assert len(ranked_ids) == 6
    assert all(case_id.startswith("risk_") for case_id in ranked_ids)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["holdout_auc"] >= 0.9
    assert [row["case_id"] for row in payload["ranking"]] == ranked_ids
    assert payload["ranking"][0]["observed_regressed"] is True
    assert "note" in payload
    assert "numpy" not in Path(train.__code__.co_filename).read_text(encoding="utf-8")
    assert "import random" not in Path(train.__code__.co_filename).read_text(encoding="utf-8")


def test_cli_json_and_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    small = tmp_path / "small.json"
    write_comparison(compare_runs(*make_runs(0), seed=13), small)
    code = main(["risk", str(small), "--json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_cases"] == 12
    assert payload["holdout_auc"] is None
    assert "at least 20" in payload["refused"]

    code = main(["risk", str(small)])
    assert code == 1
    text = capsys.readouterr().out
    assert "n_cases=12" in text
    assert "refused=" in text
    assert "at least 20" in text
