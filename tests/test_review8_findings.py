"""Regression tests for bugs found in the independent v0.6.7 review cycle.

Scope is the 0.6.7 surface and its interaction with 0.6.5–0.6.6 budgets and
YAML annotation lines: negative latency vs a set p95 budget, wall-clock
fallback that can go negative under clock skew, and compose-mark source lines
when a case mapping repeats the ``id`` key.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from promptregress.annotations import case_annotation
from promptregress.gate import evaluate_run_budgets
from promptregress.loading import load_suite
from promptregress.models import (
    AssertionSpec,
    Case,
    CaseResult,
    RunResult,
    Suite,
)
from promptregress.providers import Completion, TokenUsage
from promptregress.runner import run_suite


def test_negative_latency_fails_a_set_latency_budget() -> None:
    """A negative p95 must fail ``--max-p95-latency-ms``, like negative cost.

    0.6.7 taught the cost budget to reject non-finite and negative ``usd``
    (``nan > budget`` and ``-1 > budget`` are both false in IEEE compare).
    Latency only rejected non-finite values, so a stored run or clock-skewed
    measurement with ``latency_ms=-5`` still produced PASS for any positive
    budget. Negative latency is not evidence the budget held.
    """
    row = CaseResult(
        case_id="a",
        output="x",
        scores={"contains": 1.0},
        score=1.0,
        passed=True,
        latency_ms=-5.0,
    )
    run = RunResult(suite_name="s", config_name="c", results=[row])
    verdict = evaluate_run_budgets(run, max_p95_latency_ms=100.0)
    assert verdict.passed is False
    assert any(reason.startswith("FAIL latency:") for reason in verdict.reasons)
    assert "-5.000000" in "".join(verdict.reasons)


def test_runner_clamps_negative_wall_clock_latency() -> None:
    """Wall-clock fallback must not store a negative ``latency_ms``.

    When the provider leaves ``latency_ms`` unset, the runner subtracts
    ``perf_counter`` samples. A backwards clock step yields a negative
    duration; that value was stored and (before the gate fix) falsely passed
    a latency budget. Invalid measured latency is clamped to 0 so p95 stays
    a non-negative finite number.
    """

    class NoLatency:
        name = "n"
        model = "n"

        def complete_detailed(self, prompt: str) -> Completion:
            return Completion(
                text=prompt,
                usage=TokenUsage(0, 0, 0),
                latency_ms=None,
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
    # start=100s, end=99s → -1000 ms before the clamp
    with patch(
        "promptregress.runner.time.perf_counter",
        side_effect=[100.0, 99.0],
    ):
        result = run_suite(suite, NoLatency(), "c")  # type: ignore[arg-type]
    assert result.results[0].latency_ms == 0.0
    assert evaluate_run_budgets(result, max_p95_latency_ms=100.0).passed is True


def test_yaml_source_line_uses_last_duplicate_id_key(tmp_path: Path) -> None:
    """Compose marks must follow PyYAML's last-wins rule for duplicate ``id`` keys.

    ``safe_load`` keeps the last ``id`` value on a case mapping. Compose-mark
    attachment stopped at the first ``id`` key, so ``source_line`` pointed at
    the discarded key while ``case.id`` was the last value. Annotations then
    highlighted the wrong line for ``--github-annotations``.
    """
    path = tmp_path / "suite.yaml"
    path.write_text(
        "name: s\ncases:\n  - id: wrong_line\n    input: x\n    id: real_id\n    assertions: []\n",
        encoding="utf-8",
    )
    suite = load_suite(path)
    assert len(suite.cases) == 1
    case = suite.cases[0]
    assert case.id == "real_id"
    assert case.source_line == 5

    failed = CaseResult(
        case_id=case.id,
        output="x",
        scores={"contains": 0.0},
        score=0.0,
        passed=False,
        latency_ms=1.0,
        source_line=case.source_line,
    )
    line = case_annotation(failed, str(path))
    assert ",line=5::" in line
    assert ",line=3::" not in line
