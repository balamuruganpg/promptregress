"""Regression tests for bugs found in the independent v0.6.9 review cycle.

Scope is compose-mark ``source_line`` attachment interacting with deep JSON
documents, and the drift budget's IEEE comparisons for invalid mean drift
(the same class of false PASS as the 0.6.7–0.6.8 cost/latency guards).
"""

from __future__ import annotations

import json
from pathlib import Path

from promptregress.gate import evaluate_gate
from promptregress.loading import load_suite
from promptregress.models import (
    CaseResult,
    Comparison,
    DriftSummary,
    PairedRow,
    RunResult,
)


def test_deep_json_schema_does_not_crash_compose_source_lines(tmp_path: Path) -> None:
    """A deep ``json_schema`` must load; compose recursion must not abort.

    ``json.loads`` accepts nesting that pure-Python ``yaml.compose`` cannot
    (RecursionError around a few hundred levels). Attachment only caught
    ``YAMLError``, so a valid suite whose assertion holds a deep schema
    crashed ``load_suite`` instead of keeping ``source_line`` unset the way
    a compose failure is documented to behave. RecursionError is treated
    like ``YAMLError``: return the loaded suite without line marks.
    """
    schema: dict = {"type": "string"}
    for _ in range(400):
        schema = {"type": "object", "properties": {"x": schema}}
    path = tmp_path / "deep.json"
    path.write_text(
        json.dumps(
            {
                "name": "s",
                "cases": [
                    {
                        "id": "a",
                        "input": "{}",
                        "assertions": [{"kind": "json_schema", "value": schema}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    suite = load_suite(path)
    assert len(suite.cases) == 1
    assert suite.cases[0].id == "a"
    assert suite.cases[0].source_line is None


def test_nonfinite_or_negative_mean_drift_fails_a_set_drift_budget() -> None:
    """Non-finite or negative mean drift must fail ``--max-mean-drift``.

    Cost and latency budgets reject non-finite and negative observations
    because ``nan > budget`` and ``-1 > budget`` are false in IEEE compare
    and would PASS. Drift only tested ``mean_drift > budget``, so a crafted
    or corrupted summary with ``mean_drift=nan`` (or ``-inf`` / negative)
    falsely passed any finite ceiling. Drift is documented in ``[0, 2]``;
    an invalid mean is not evidence the budget held.
    """
    row = CaseResult(
        case_id="a",
        output="x",
        scores={"contains": 1.0},
        score=1.0,
        passed=True,
        latency_ms=1.0,
    )
    run = RunResult(suite_name="s", config_name="c", results=[row])
    paired = PairedRow(
        case_id="a",
        baseline_output="x",
        candidate_output="x",
        baseline_score=1.0,
        candidate_score=1.0,
        diff=0.0,
        baseline_passed=True,
        candidate_passed=True,
        outcome="tie",
    )
    for mean in (float("nan"), float("-inf"), -1.0):
        comparison = Comparison(
            baseline=run,
            candidate=run,
            mean_diff=0.0,
            ci_low=0.0,
            ci_high=0.0,
            mcnemar_p=1.0,
            wins=0,
            losses=0,
            ties=1,
            per_case=[paired],
            drift=DriftSummary(
                embedder="hash",
                mean_drift=mean,
                max_drift=1.0,
                per_case=[],
                top=[],
            ),
        )
        verdict = evaluate_gate(comparison, max_mean_drift=0.5)
        assert verdict.passed is False, mean
        assert any(reason.startswith("FAIL drift:") for reason in verdict.reasons)
