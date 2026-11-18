"""Suite scoring.weights: the weighted case score, and the unweighted mean."""

import json
from pathlib import Path

import pytest

from promptregress.loading import SuiteError, load_suite
from promptregress.models import AssertionSpec, Case, ScoringConfig, Suite
from promptregress.providers import EchoProvider
from promptregress.runner import run_suite
from promptregress.scorers import ScoreError, score_case

ROOT = Path(__file__).resolve().parents[1]


def _two_scorer_case(*, exact_weight: float = 1.0, levenshtein_weight: float = 1.0) -> Case:
    """exact_match('abcd') scores 1; levenshtein('ab') on 'abcd' scores 0.5."""
    return Case(
        id="c",
        input="abcd",
        reference=None,
        assertions=[
            AssertionSpec("exact_match", value="abcd", weight=exact_weight),
            AssertionSpec("levenshtein", value="ab", weight=levenshtein_weight),
        ],
    )


def test_weighted_and_unweighted_two_scorer_case() -> None:
    case = _two_scorer_case()
    output = "abcd"
    unweighted = score_case(case, output)
    equal = score_case(case, output, weights={"exact_match": 1, "levenshtein": 1})
    weighted = score_case(case, output, weights={"exact_match": 1, "levenshtein": 3})
    partial = score_case(case, output, weights={"levenshtein": 3})

    # (1.0 + 0.5) / 2 = 0.75. Equal suite weights cancel and match that mean.
    assert unweighted[0] == {"exact_match": 1.0, "levenshtein": 0.5}
    assert unweighted[1] == pytest.approx(0.75)
    assert unweighted[2] is True
    assert equal[0] == unweighted[0]
    assert equal[1] == pytest.approx(0.75)
    assert equal[2] is True

    # (1.0 * 1 + 0.5 * 3) / (1 + 3) = 2.5 / 4 = 0.625.
    # A kind omitted from the map still weighs 1, so naming only levenshtein matches.
    assert weighted[0] == {"exact_match": 1.0, "levenshtein": 0.5}
    assert weighted[1] == pytest.approx(0.625)
    assert weighted[2] is True
    assert partial[1] == pytest.approx(0.625)
    assert partial[0] == weighted[0]


def test_scorer_weights_multiply_assertion_weights() -> None:
    case = _two_scorer_case(exact_weight=2.0, levenshtein_weight=1.0)
    output = "abcd"
    plain = score_case(case, output)
    weighted = score_case(case, output, weights={"exact_match": 1, "levenshtein": 3})

    # Assertion weights only: (1*2 + 0.5*1) / 3 = 2.5/3.
    assert plain[1] == pytest.approx(2.5 / 3)
    # Effective weights 2*1 and 1*3: (1*2 + 0.5*3) / 5 = 3.5/5 = 0.7.
    assert weighted[1] == pytest.approx(0.7)
    assert weighted[0] == {"exact_match": 1.0, "levenshtein": 0.5}


def test_zero_weight_drops_that_scorer_from_the_mean() -> None:
    case = _two_scorer_case()
    scored = score_case(case, "abcd", weights={"exact_match": 0, "levenshtein": 1})
    # (1*0 + 0.5*1) / 1 = 0.5. The exact subscore is still reported.
    assert scored[1] == pytest.approx(0.5)
    assert scored[0]["exact_match"] == 1.0
    assert scored[0]["levenshtein"] == pytest.approx(0.5)
    # Graded floor is 0.5, so this still passes. The exact assertion itself passed.
    assert scored[2] is True


def test_negative_scorer_weight_is_a_score_error() -> None:
    case = _two_scorer_case()
    with pytest.raises(ScoreError, match="non-negative"):
        score_case(case, "abcd", weights={"exact_match": -1})


def _write_suite(path: Path, scoring: object) -> Path:
    path.write_text(
        json.dumps(
            {
                "name": "weighted",
                "scoring": scoring,
                "cases": [
                    {
                        "id": "a",
                        "input": "abcd",
                        "assertions": [
                            {"kind": "exact_match", "value": "abcd"},
                            {"kind": "levenshtein", "value": "ab"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_loader_rejects_unknown_negative_and_all_zero_weights(tmp_path: Path) -> None:
    unknown = _write_suite(tmp_path / "unknown.json", {"weights": {"nope": 1}})
    with pytest.raises(SuiteError, match="Unknown scorer 'nope'"):
        load_suite(unknown)

    negative = _write_suite(tmp_path / "negative.json", {"weights": {"exact_match": -0.5}})
    with pytest.raises(SuiteError, match="negative"):
        load_suite(negative)

    zeros = _write_suite(
        tmp_path / "zeros.json",
        {"weights": {"exact_match": 0, "levenshtein": 0}},
    )
    with pytest.raises(SuiteError, match="all zero"):
        load_suite(zeros)

    not_object = _write_suite(tmp_path / "bad.json", {"weights": ["exact_match", 1]})
    with pytest.raises(SuiteError, match="scoring.weights must be an object"):
        load_suite(not_object)


def test_loader_keeps_assertion_weights_and_applies_the_map(tmp_path: Path) -> None:
    path = _write_suite(tmp_path / "ok.json", {"weights": {"exact_match": 1, "levenshtein": 3}})
    suite = load_suite(path)
    assert suite.scoring.weights == {"exact_match": 1.0, "levenshtein": 3.0}
    assert suite.cases[0].assertions[0].weight == 1.0
    scores, score, _passed = score_case(suite.cases[0], "abcd", weights=suite.scoring.weights)
    assert score == pytest.approx(0.625)
    assert scores == {"exact_match": 1.0, "levenshtein": 0.5}


def test_yaml_weights_match_the_documented_shape(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    path.write_text(
        """
name: yaml-weights
scoring:
  weights:
    exact_match: 1
    levenshtein: 3
cases:
  - id: a
    input: abcd
    assertions:
      - kind: exact_match
        value: abcd
      - kind: levenshtein
        value: ab
""".lstrip(),
        encoding="utf-8",
    )
    suite = load_suite(path)
    assert suite.scoring == ScoringConfig(weights={"exact_match": 1.0, "levenshtein": 3.0})
    _scores, score, _passed = score_case(suite.cases[0], "abcd", weights=suite.scoring.weights)
    assert score == pytest.approx(0.625)


def test_demo_suite_equal_weights_match_the_unweighted_scores() -> None:
    """The bundled multi-scorer suite must not move when weights are all 1."""
    suite = load_suite(ROOT / "examples" / "support_qa.json")
    kinds = {assertion.kind for case in suite.cases for assertion in case.assertions}
    assert len(kinds) >= 2
    weights = {kind: 1.0 for kind in kinds}
    for case in suite.cases:
        for output in (case.reference or "", "Settings"):
            plain = score_case(case, output)
            weighted = score_case(case, output, weights=weights)
            assert weighted[0] == plain[0]
            assert weighted[1] == plain[1]
            assert weighted[2] is plain[2]

    weighted_suite = Suite(
        name=suite.name,
        description=suite.description,
        cases=suite.cases,
        scoring=ScoringConfig(weights=weights),
    )
    plain_run = run_suite(suite, EchoProvider(), "echo", prompt_template="{input}")
    weighted_run = run_suite(weighted_suite, EchoProvider(), "echo", prompt_template="{input}")
    assert [result.scores for result in weighted_run.results] == [
        result.scores for result in plain_run.results
    ]
    assert [result.score for result in weighted_run.results] == [
        result.score for result in plain_run.results
    ]
    assert [result.passed for result in weighted_run.results] == [
        result.passed for result in plain_run.results
    ]
