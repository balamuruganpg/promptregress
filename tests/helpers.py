"""Small builders shared by the statistical tests."""

from promptregress.models import CaseResult, Comparison, RunResult


def case_result(
    case_id: str,
    score: float,
    passed: bool,
    output: str = "out",
    tags: list[str] | None = None,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        output=output,
        scores={"exact_match": score},
        score=score,
        passed=passed,
        latency_ms=1.0,
        tags=list(tags or []),
    )


def run_with(
    rows: list[tuple[str, float, bool]],
    config: str,
    suite: str = "suite",
    tags: dict[str, list[str]] | None = None,
) -> RunResult:
    tag_map = tags or {}
    return RunResult(
        suite_name=suite,
        config_name=config,
        results=[
            case_result(case_id, score, passed, tags=tag_map.get(case_id, []))
            for case_id, score, passed in rows
        ],
    )


def comparison_with(
    *,
    ci_low: float,
    ci_high: float,
    mean_diff: float,
    candidate_rows: list[tuple[str, float, bool]],
    baseline_rows: list[tuple[str, float, bool]] | None = None,
) -> Comparison:
    if baseline_rows is None:
        baseline_rows = [(case_id, 1.0, True) for case_id, _, _ in candidate_rows]
    return Comparison(
        baseline=run_with(baseline_rows, "baseline"),
        candidate=run_with(candidate_rows, "candidate"),
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        mcnemar_p=1.0,
        wins=0,
        losses=0,
        ties=len(candidate_rows),
        per_case=[],
    )
