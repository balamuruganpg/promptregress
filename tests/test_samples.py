"""Multi-sample runs: mean score, population sd, cache, and mock variation."""

import json
import math
from pathlib import Path

import pytest

from promptregress.cache import ResponseCache
from promptregress.cli import (
    DEMO_BASELINE_TEMPLATE,
    DEMO_CANDIDATE_TEMPLATE,
    bundled_suite_path,
    demo_rules,
    main,
)
from promptregress.gate import evaluate_gate
from promptregress.loading import load_suite
from promptregress.models import AssertionSpec, Case, Suite
from promptregress.providers import EchoProvider, MockProvider
from promptregress.report import render_markdown
from promptregress.runner import run_suite
from promptregress.stats import compare_runs


def _suite(*cases: tuple[str, str]) -> Suite:
    return Suite(
        name="s",
        cases=[
            Case(
                id=case_id,
                input=text,
                reference="yes",
                assertions=[AssertionSpec("exact_match", value="yes")],
            )
            for case_id, text in cases
        ],
    )


def _sd(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def test_three_samples_two_good_one_bad_mean_is_two_thirds() -> None:
    provider = MockProvider(rules=[("question", ("yes", "yes", "no"))], vary=True)
    result = run_suite(_suite(("a", "question")), provider, "sampled", samples=3)
    row = result.results[0]
    assert provider.call_count == 3
    assert row.sample_outputs == ["yes", "yes", "no"]
    assert row.sample_scores == [1.0, 1.0, 0.0]
    assert row.score == pytest.approx(2 / 3)
    assert row.passed is True
    assert row.score_sd == pytest.approx(_sd([1.0, 1.0, 0.0]))
    assert row.n_samples == 3
    assert row.output == "yes"
    assert result.mean_score == pytest.approx(2 / 3)
    assert result.pass_rate == 1.0


def test_mean_at_one_half_passes() -> None:
    provider = MockProvider(rules=[("question", ("yes", "no"))], vary=True)
    row = run_suite(_suite(("a", "question")), provider, "sampled", samples=2).results[0]
    assert row.sample_scores == [1.0, 0.0]
    assert row.score == 0.5
    assert row.passed is True


def test_one_good_of_three_fails() -> None:
    provider = MockProvider(rules=[("question", ("yes", "no", "no"))], vary=True)
    row = run_suite(_suite(("a", "question")), provider, "sampled", samples=3).results[0]
    assert row.score == pytest.approx(1 / 3)
    assert row.passed is False


def test_samples_one_matches_the_demo_suite() -> None:
    suite = load_suite(bundled_suite_path())

    def _both(samples: int) -> tuple[list[str], list[float], list[bool]]:
        provider = MockProvider(
            rules=demo_rules(suite),
            default="I do not know.",
            name="mock",
            model="demo",
        )
        baseline = run_suite(
            suite,
            provider,
            "baseline",
            prompt_template=DEMO_BASELINE_TEMPLATE,
            samples=samples,
        )
        candidate = run_suite(
            suite,
            provider,
            "candidate",
            prompt_template=DEMO_CANDIDATE_TEMPLATE,
            samples=samples,
        )
        rows = baseline.results + candidate.results
        return (
            [row.output for row in rows],
            [row.score for row in rows],
            [row.passed for row in rows],
        )

    default_outputs, default_scores, default_passed = _both(1)
    again_outputs, again_scores, again_passed = _both(1)
    assert default_outputs == again_outputs
    assert default_scores == again_scores
    assert default_passed == again_passed
    explicit = run_suite(
        suite,
        MockProvider(rules=demo_rules(suite), default="I do not know.", model="demo"),
        "baseline",
        prompt_template=DEMO_BASELINE_TEMPLATE,
        samples=1,
    )
    plain = run_suite(
        suite,
        MockProvider(rules=demo_rules(suite), default="I do not know.", model="demo"),
        "baseline",
        prompt_template=DEMO_BASELINE_TEMPLATE,
    )
    assert [row.output for row in explicit.results] == [row.output for row in plain.results]
    assert [row.score for row in explicit.results] == [row.score for row in plain.results]
    assert [row.passed for row in explicit.results] == [row.passed for row in plain.results]
    assert [row.scores for row in explicit.results] == [row.scores for row in plain.results]
    assert [row.prompt_tokens for row in explicit.results] == [
        row.prompt_tokens for row in plain.results
    ]
    assert explicit.results[0].n_samples == 1
    assert explicit.results[0].sample_scores == [explicit.results[0].score]
    assert explicit.results[0].score_sd == 0.0
    assert explicit.results[0].output == plain.results[0].output


def test_one_sample_keeps_the_exact_pass_rule() -> None:
    """N=1 keeps the scorer pass bit. N>1 passes on the mean score."""
    suite = Suite(
        name="s",
        cases=[
            Case(
                id="a",
                input="hello",
                reference=None,
                assertions=[
                    AssertionSpec("contains", value="hello"),
                    AssertionSpec("contains", value="missing"),
                ],
            )
        ],
    )
    single = run_suite(suite, EchoProvider(), "echo", samples=1).results[0]
    assert single.score == 0.5
    assert single.passed is False
    repeated = run_suite(suite, EchoProvider(), "echo", samples=2).results[0]
    assert repeated.sample_scores == [0.5, 0.5]
    assert repeated.score == 0.5
    assert repeated.passed is True


def test_samples_must_be_positive() -> None:
    suite = _suite(("a", "question"))
    with pytest.raises(ValueError, match="samples"):
        run_suite(suite, MockProvider(default="yes"), "c", samples=0)
    with pytest.raises(ValueError, match="samples"):
        run_suite(suite, MockProvider(default="yes"), "c", samples=True)  # type: ignore[arg-type]


def test_cache_replays_the_first_sample(tmp_path: Path) -> None:
    provider = MockProvider(rules=[("question", ("yes", "yes", "no"))], vary=True)
    cache = ResponseCache(tmp_path)
    result = run_suite(
        _suite(("a", "question")),
        provider,
        "cached",
        samples=3,
        cache=cache,
    )
    row = result.results[0]
    assert provider.call_count == 1
    assert cache.hits == 2
    assert row.sample_outputs == ["yes", "yes", "yes"]
    assert row.cached is False
    assert row.score == 1.0


def test_vary_cycles_calls_and_string_rules_stay_fixed() -> None:
    cycling = MockProvider(rules=[("q", ("a", "b", "c"))], vary=True, default="nope")
    assert [cycling.complete("q") for _ in range(4)] == ["a", "b", "c", "a"]
    assert cycling.complete("other") == "nope"
    fixed = MockProvider(rules=[("q", "only")], vary=True)
    assert fixed.complete("q") == "only"
    assert fixed.complete("q") == "only"
    primary = MockProvider(rules=[("q", ("a", "b"))], vary=False)
    assert primary.complete("q") == "a"
    assert primary.complete("q") == "a"
    first_rule = MockProvider(
        rules=[("q", ("a", "b")), ("qq", ("z",))],
        vary=True,
    )
    assert first_rule.complete_at("qq", 1).text == "b"


def test_complete_at_uses_the_sample_index() -> None:
    provider = MockProvider(rules=[("q", ("a", "b", "c"))], vary=True)
    assert provider.complete_at("q", 2).text == "c"
    assert provider.complete_at("q", 0).text == "a"
    assert provider.complete_at("q", 3).text == "a"
    with pytest.raises(ValueError, match="sample_index"):
        provider.complete_at("q", -1)


def test_stability_section_lists_the_least_stable_cases() -> None:
    provider = MockProvider(
        rules=[
            ("stable-question", ("yes", "yes", "yes")),
            ("shaky-question", ("yes", "yes", "no")),
        ],
        vary=True,
    )
    suite = _suite(("stable", "stable-question"), ("shaky", "shaky-question"))
    baseline = run_suite(suite, provider, "baseline", samples=3)
    candidate = run_suite(suite, provider, "candidate", samples=3)
    shaky = next(row for row in baseline.results if row.case_id == "shaky")
    stable = next(row for row in baseline.results if row.case_id == "stable")
    assert shaky.score_sd > stable.score_sd
    assert stable.score_sd == 0.0
    assert baseline.mean_within_case_sd == pytest.approx(shaky.score_sd / 2)
    comparison = compare_runs(baseline, candidate, n_boot=20, seed=1)
    text = render_markdown(comparison, evaluate_gate(comparison, max_regression=1.0))
    assert "## Stability" in text
    assert "Mean within-case sd" in text
    assert text.index("| shaky |") < text.index("| stable |")


def test_workers_keep_per_case_sample_indexes(tmp_path: Path) -> None:
    del tmp_path
    provider = MockProvider(
        rules=[
            ("CASE-A", ("yes", "yes", "no")),
            ("CASE-B", ("no", "no", "no")),
        ],
        vary=True,
    )
    suite = _suite(("a", "CASE-A"), ("b", "CASE-B"))
    result = run_suite(suite, provider, "parallel", samples=3, max_workers=2)
    assert [row.case_id for row in result.results] == ["a", "b"]
    assert result.results[0].sample_scores == [1.0, 1.0, 0.0]
    assert result.results[1].sample_scores == [0.0, 0.0, 0.0]


def test_cli_samples_and_mock_vary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [
                    {
                        "id": "a",
                        "input": "question",
                        "reference": "yes",
                        "assertions": [{"kind": "exact_match", "value": "yes"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "run.json"
    code = main(
        [
            "run",
            str(suite),
            "--provider",
            "mock",
            "--mock-vary",
            "--mock-rules",
            json.dumps([["question", ["yes", "yes", "no"]]]),
            "--config-name",
            "sampled",
            "--samples",
            "3",
            "--out",
            str(out),
            "--no-history",
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["results"][0]["score"] == pytest.approx(2 / 3)
    assert payload["results"][0]["sample_scores"] == [1.0, 1.0, 0.0]
    assert payload["stability"]["least_stable"][0]["case_id"] == "a"
    assert "samples=3" in capsys.readouterr().out
    bad = main(
        [
            "run",
            str(suite),
            "--provider",
            "echo",
            "--config-name",
            "x",
            "--samples",
            "0",
            "--out",
            str(out),
            "--no-history",
        ]
    )
    assert bad == 2
    assert "samples" in capsys.readouterr().err
