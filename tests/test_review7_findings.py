"""Regression tests for bugs found in the independent v0.6.6 review cycle.

Scope is the 0.6.5–0.6.6 surface: GitHub Actions annotation line numbers from
YAML suites, offline latency budgets with EchoProvider, and cost/latency
budget comparisons when the observed value is non-finite.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from promptregress.annotations import case_annotation
from promptregress.gate import evaluate_run_budgets
from promptregress.loading import load_suite
from promptregress.models import (
    AssertionSpec,
    Case,
    CaseResult,
    CostEstimate,
    RunResult,
    Suite,
)
from promptregress.pricing import ModelPrice, estimate_cost
from promptregress.providers import Completion, EchoProvider, TokenUsage
from promptregress.runner import run_suite


def test_yaml_source_line_ignores_id_text_inside_block_scalar(
    tmp_path: Path,
) -> None:
    """Annotation ``line`` must be the case's ``id`` field, not input text.

    ``_attach_source_lines`` scanned the raw YAML with a loose ``id: value``
    regex. A later case id that also appeared inside an earlier case's
    literal/folded block scalar (for example ``input: |`` containing
    ``id: b``) was matched first, so ``--github-annotations`` pointed at the
    wrong line. Structural id marks from the YAML compose tree locate the
    real field.
    """
    path = tmp_path / "suite.yaml"
    path.write_text(
        "name: s\n"
        "cases:\n"
        "  - id: a\n"
        "    input: |\n"
        "      id: b\n"
        "      hello\n"
        "    assertions: []\n"
        "  - id: b\n"
        "    input: x\n"
        "    assertions: []\n",
        encoding="utf-8",
    )
    suite = load_suite(path)
    assert [(case.id, case.source_line) for case in suite.cases] == [("a", 3), ("b", 8)]

    failed = CaseResult(
        case_id="b",
        output="x",
        scores={"contains": 0.0},
        score=0.0,
        passed=False,
        latency_ms=1.0,
        source_line=suite.cases[1].source_line,
    )
    line = case_annotation(failed, str(path))
    assert ",line=8::" in line
    assert ",line=5::" not in line


def test_echo_provider_records_fixed_latency_for_deterministic_budgets() -> None:
    """Echo must record a fixed latency like Mock, not wall-clock noise.

    ``MockProvider`` was given a fixed 1 ms latency so offline
    ``--max-p95-latency-ms`` checks are deterministic. ``EchoProvider`` still
    left ``latency_ms`` unset, so the runner used ``perf_counter`` and the
    same suite could pass or fail a tight latency budget across runs.
    """
    suite = Suite(
        name="s",
        cases=[
            Case(
                id="a",
                input="hi",
                reference=None,
                assertions=[AssertionSpec(kind="contains", value="hi")],
            )
        ],
    )
    first = run_suite(suite, EchoProvider(), "echo")
    second = run_suite(suite, EchoProvider(), "echo")
    assert first.results[0].latency_ms == 1.0
    assert second.results[0].latency_ms == 1.0
    assert first.p95_latency_ms == second.p95_latency_ms == 1.0
    held = evaluate_run_budgets(first, max_p95_latency_ms=1.0)
    assert held.passed is True
    tight = evaluate_run_budgets(first, max_p95_latency_ms=0.5)
    assert tight.passed is False


def test_nonfinite_latency_fails_a_set_latency_budget() -> None:
    """A non-finite p95 must fail ``--max-p95-latency-ms``, not silently pass.

    Cost budgets already fail when the estimate is missing or ``usd`` is
    ``None``. Latency used a raw ``observed > budget`` compare, so ``NaN``
    (and ``-inf``) never exceeded the budget and produced a false PASS.
    Invalid latency is not evidence the budget held.
    """
    row = CaseResult(
        case_id="a",
        output="x",
        scores={"contains": 1.0},
        score=1.0,
        passed=True,
        latency_ms=float("nan"),
    )
    run = RunResult(suite_name="s", config_name="c", results=[row])
    verdict = evaluate_run_budgets(run, max_p95_latency_ms=100.0)
    assert verdict.passed is False
    assert any(reason.startswith("FAIL latency:") for reason in verdict.reasons)
    assert "nan" in "".join(verdict.reasons).lower() or "not finite" in "".join(verdict.reasons)

    neg = evaluate_run_budgets(
        RunResult(
            suite_name="s",
            config_name="c",
            results=[replace(row, latency_ms=float("-inf"))],
        ),
        max_p95_latency_ms=0.0,
    )
    assert neg.passed is False


def test_nonfinite_or_negative_cost_fails_a_set_cost_budget() -> None:
    """Non-finite or negative ``cost.usd`` must fail like a missing estimate.

    ``_cost_reason`` only tested ``usd > budget``. ``NaN`` and negative
    values never compared greater, so a stored run with ``usd: -5`` or
    ``usd: NaN`` passed every cost budget. Those values are not a valid
    USD estimate and must fail a set budget the same way ``usd: null`` does.
    """
    base = RunResult(
        suite_name="s",
        config_name="c",
        results=[
            CaseResult(
                case_id="a",
                output="x",
                scores={"contains": 1.0},
                score=1.0,
                passed=True,
                latency_ms=1.0,
            )
        ],
    )
    for usd in (float("nan"), float("-inf"), -1.0):
        verdict = evaluate_run_budgets(
            replace(base, cost=CostEstimate(model="m", usd=usd, note=None)),
            max_cost_usd=100.0,
        )
        assert verdict.passed is False, usd
        assert any(reason.startswith("FAIL cost:") for reason in verdict.reasons)


def test_estimate_cost_overflow_is_unknown_not_a_crash() -> None:
    """Absurd token totals must yield an unknown cost, not ``OverflowError``.

    ``estimate_cost`` multiplied token counts by prices with float arithmetic.
    A run whose prompt token total exceeds the float range (possible via a
    custom provider's ``TokenUsage``, or a hand-edited stored report) raised
    ``OverflowError`` out of ``_attach_cost``, so ``run``/``compare`` exited 2
    instead of recording unknown cost and failing a set ``--max-cost-usd``.
    """
    huge = 10**400
    run = RunResult(
        suite_name="s",
        config_name="c",
        results=[
            CaseResult(
                case_id="a",
                output="x",
                scores={"contains": 1.0},
                score=1.0,
                passed=True,
                latency_ms=1.0,
                prompt_tokens=huge,
                completion_tokens=0,
                total_tokens=huge,
            )
        ],
    )
    estimate = estimate_cost(run, "m", {"m": ModelPrice(1.0, 1.0)})
    assert estimate.usd is None
    assert estimate.note is not None
    verdict = evaluate_run_budgets(replace(run, cost=estimate), max_cost_usd=1.0)
    assert verdict.passed is False


def test_runner_ignores_nonfinite_provider_latency() -> None:
    """Provider ``latency_ms`` that is NaN/negative must not be stored as-is.

    ``_recorded_latency`` trusted any non-``None`` value. A ``complete_detailed``
    implementation that returned ``latency_ms=nan`` made p95 ``nan`` and
    (before the gate fix) falsely passed latency budgets. Invalid recorded
    latency falls back to a measured duration so the case still has a finite
    sample, and Mock's validation remains the rule for intentional fixtures.
    """

    class Nasty:
        name = "nasty"
        model = "nasty"

        def complete_detailed(self, prompt: str) -> Completion:
            return Completion(
                text="hi",
                usage=TokenUsage(2, 2, 4),
                latency_ms=float("nan"),
            )

    suite = Suite(
        name="s",
        cases=[
            Case(
                id="a",
                input="hi",
                reference=None,
                assertions=[AssertionSpec(kind="contains", value="hi")],
            )
        ],
    )
    result = run_suite(suite, Nasty(), "c")  # type: ignore[arg-type]
    latency = result.results[0].latency_ms
    assert math.isfinite(latency) and latency >= 0.0
    assert evaluate_run_budgets(result, max_p95_latency_ms=1_000_000.0).passed is True
