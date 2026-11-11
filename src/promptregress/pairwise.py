"""Two-order pairwise judge for a baseline and a candidate.

Each case is shown to the judge twice: baseline first, then candidate first.
Swapping the order cancels a judge that always prefers the first slot. The
case score is 1 when the candidate wins both orders, 0 when it loses both,
and 0.5 on a split or a tie. The reported win rate is the mean of those
scores. Its confidence interval is the paired bootstrap from
:func:`promptregress.stats.paired_bootstrap_ci`, applied to the scores.

An unreadable reply, or a judge exception, counts as a tie for that order
and is recorded on the case. It does not raise, and it cannot produce a
double win. The score gate does not read this result.
"""

from dataclasses import replace

from promptregress.calibration import parse_preference
from promptregress.models import (
    Comparison,
    Outcome,
    PairwiseCase,
    PairwiseSide,
    PairwiseSummary,
    Suite,
)
from promptregress.providers import Provider
from promptregress.stats import paired_bootstrap_ci


def pairwise_prompt(
    *,
    case_id: str,
    rubric: str,
    case_input: str,
    reference: str | None,
    first: str,
    second: str,
) -> str:
    """Prompt that asks for ``first``, ``second``, or ``tie``.

    Output A is the first slot. Output B is the second slot. The caller
    builds this twice per case, with the slots swapped.
    """
    shown_reference = "" if reference is None else reference
    return (
        "You are comparing two model outputs for the same input. "
        "Use the rubric. Reply with only one of these words: first, second, tie.\n"
        "first means Output A is better. second means Output B is better. "
        "tie means they are equally good.\n"
        f"Case: {case_id}\n"
        f"Rubric: {rubric}\n"
        f"Input: {case_input}\n"
        f"Reference: {shown_reference}\n"
        "Output A:\n"
        f"{first}\n"
        "Output B:\n"
        f"{second}\n"
    )


def combine_orders(baseline_first: str, candidate_first: str) -> tuple[Outcome, float]:
    """Map the two presentation winners onto a candidate outcome and score.

    Each argument is ``baseline``, ``candidate``, or ``tie``: who won when
    that config was shown in the first slot. Both candidate wins score 1.
    Both baseline wins score 0. Anything else, including a tie on either
    order, scores 0.5.
    """
    if baseline_first == "candidate" and candidate_first == "candidate":
        return "win", 1.0
    if baseline_first == "baseline" and candidate_first == "baseline":
        return "loss", 0.0
    return "tie", 0.5


def pairwise_judge(
    comparison: Comparison,
    suite: Suite,
    judge: Provider,
    rubric: str,
    *,
    n_boot: int = 2000,
    seed: int = 13,
    alpha: float = 0.05,
    ci_method: str = "percentile",
) -> PairwiseSummary:
    """Judge every paired case in both orders and summarize the candidate.

    Cases follow ``comparison.per_case``. The suite supplies the input and
    reference for the prompt. A case id missing from the suite raises
    ``ValueError``.
    """
    if not isinstance(rubric, str) or rubric.strip() == "":
        raise ValueError("rubric must be a non-empty string")
    text = rubric.strip()
    by_id = {case.id: case for case in suite.cases}
    judged: list[PairwiseCase] = []
    for row in comparison.per_case:
        case = by_id.get(row.case_id)
        if case is None:
            raise ValueError(f"suite has no case {row.case_id!r} for the pairwise judge")
        forward, forward_error = _one_order(
            judge,
            case_id=row.case_id,
            rubric=text,
            case_input=case.input,
            reference=case.reference,
            first_text=row.baseline_output,
            second_text=row.candidate_output,
            first_side="baseline",
            second_side="candidate",
        )
        backward, backward_error = _one_order(
            judge,
            case_id=row.case_id,
            rubric=text,
            case_input=case.input,
            reference=case.reference,
            first_text=row.candidate_output,
            second_text=row.baseline_output,
            first_side="candidate",
            second_side="baseline",
        )
        outcome, score = combine_orders(forward, backward)
        errors = [item for item in (forward_error, backward_error) if item]
        judged.append(
            PairwiseCase(
                case_id=row.case_id,
                outcome=outcome,
                score=score,
                order_baseline_first=forward,
                order_candidate_first=backward,
                error="; ".join(errors) if errors else None,
            )
        )
    scores = [item.score for item in judged]
    if scores:
        win_rate = sum(scores) / len(scores)
    else:
        win_rate = 0.0
    ci_low, ci_high = paired_bootstrap_ci(
        scores,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        method=ci_method,
    )
    wins = sum(1 for item in judged if item.outcome == "win")
    losses = sum(1 for item in judged if item.outcome == "loss")
    ties = sum(1 for item in judged if item.outcome == "tie")
    return PairwiseSummary(
        rubric=text,
        wins=wins,
        losses=losses,
        ties=ties,
        win_rate=win_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        cases=judged,
        n_boot=n_boot,
        seed=seed,
        alpha=float(alpha),
        ci_method=ci_method,
    )


def with_pairwise(comparison: Comparison, summary: PairwiseSummary) -> Comparison:
    """Return a copy of ``comparison`` carrying the pairwise summary."""
    return replace(comparison, pairwise=summary)


def _one_order(
    judge: Provider,
    *,
    case_id: str,
    rubric: str,
    case_input: str,
    reference: str | None,
    first_text: str,
    second_text: str,
    first_side: PairwiseSide,
    second_side: PairwiseSide,
) -> tuple[PairwiseSide, str | None]:
    prompt = pairwise_prompt(
        case_id=case_id,
        rubric=rubric,
        case_input=case_input,
        reference=reference,
        first=first_text,
        second=second_text,
    )
    try:
        raw = judge.complete(prompt)
    except Exception as exc:
        return "tie", f"judge provider failed: {type(exc).__name__}: {exc}"
    if not isinstance(raw, str):
        return "tie", "judge provider returned a non-string"
    try:
        side = parse_preference(first_text, second_text, raw)
    except ValueError as exc:
        return "tie", str(exc)
    if side == "first":
        return first_side, None
    if side == "second":
        return second_side, None
    return "tie", None
