"""Judge calibration and two bias audits. All of this is stdlib.

``calibrate`` compares judge scores with human scores: Pearson, Spearman,
MAE, and Cohen's kappa on a pass/fail cut. ``position_bias_audit`` presents
each pair in both orders. ``verbosity_bias`` correlates the length gap with
the judge's preference on the given order only, so position is not swapped
in that audit.

Simplifications, stated here so the numbers are not read as more than they are:

- Pearson and Spearman are undefined (``None``) when a side has no variance
  or there are fewer than two rows. They are not imputed as 0.
- Spearman uses midranks and then Pearson on those ranks. The
  ``1 - 6 sum(d^2) / (n (n^2 - 1))`` shortcut is not used, because it is
  wrong when ranks tie.
- A score at or above ``threshold`` is a pass. Kappa's chance term is the
  product of the marginal pass rates. When chance agreement is already 1,
  kappa is 1 if the raters fully agree and ``None`` otherwise.
- The position-bias flag is ``flip_rate >= 0.5``. It is a descriptive cut,
  not a hypothesis test. A flip is a change in which output wins when the
  order is swapped, including a side versus a tie.
- Verbosity preference is ``+1`` for the first output, ``-1`` for the
  second, and ``0`` for a tie. The reported correlation is Pearson.
- Ensemble agreement (see :func:`judge_agreement_for_run`) reports the mean
  pairwise absolute score difference and Fleiss' kappa on pass/fail. The
  pass cut is the graded floor (0.5). It is Fleiss' kappa, not Cohen's.
"""

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from promptregress.models import AssertionSpec, Case, JudgeAgreement, RunResult
from promptregress.providers import Provider
from promptregress.scorers import GRADED_MEAN_FLOOR, ScoringContext, score_case

# A judge score at or above this is a pass for Fleiss' kappa. It is the same
# cut the graded pass rule uses for a case mean. It is not a second threshold.
JUDGE_PASS_THRESHOLD = GRADED_MEAN_FLOOR

JUDGE_AGREEMENT_DEFINITION = (
    "mean_pairwise_abs is the mean, across cases, of the mean absolute "
    "difference between every pair of judge scores on that case. "
    "fleiss_kappa is Fleiss' kappa on pass/fail: a judge passes when its "
    "score is >= pass_threshold (the graded pass cut, 0.5). It is undefined "
    "when chance agreement is 1 and the judges are not in full agreement. "
    "per_judge_mean is each judge's mean score on the included cases. "
    "The case score is the unweighted mean of the judge scores. "
    "This does not change the gate."
)

JudgeFn = Callable[[str, str], str]


@dataclass(frozen=True)
class CalibrationReport:
    """Agreement between judge scores and human scores on the same rows."""

    n: int
    pearson: float | None
    spearman: float | None
    mae: float
    threshold: float
    kappa: float | None
    raw_agreement: float
    both_pass: int
    judge_only: int
    human_only: int
    both_fail: int
    judge_scores: list[float]
    human_scores: list[float]
    notes: list[str]


@dataclass(frozen=True)
class PositionBiasAudit:
    """Order-swap audit for a pairwise judge.

    ``flip_rate`` is ``n_flips / n_pairs``. ``first_position_win_rate`` counts
    presentations where the first slot wins, over both orders (ties are not
    first-position wins). ``flagged`` is true when ``flip_rate`` is at least
    ``flag_flip_rate`` (default 0.5).
    """

    n_pairs: int
    n_flips: int
    flip_rate: float
    first_position_wins: int
    first_position_win_rate: float
    flagged: bool
    flag_flip_rate: float


@dataclass(frozen=True)
class VerbosityBias:
    """Pearson correlation of length gap and judge preference.

    ``length_diffs`` is ``len(output_a) - len(output_b)``. ``preferences`` is
    ``+1`` when the judge picks A, ``-1`` when it picks B, and ``0`` for a
    tie. A positive correlation means longer outputs are preferred.
    ``correlation`` is ``None`` when either side has no variance.
    """

    n_pairs: int
    correlation: float | None
    length_diffs: list[float]
    preferences: list[float]


@dataclass(frozen=True)
class CalibrationExample:
    """One JSONL row for ``promptregress calibrate``."""

    input: str
    output: str
    human_score: float
    reference: str | None = None
    example_id: str = ""


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Pearson correlation, or ``None`` if it is undefined."""
    xs = _pair(left, right, "scores")
    n = len(xs[0])
    if n < 2:
        return None
    mean_x = sum(xs[0]) / n
    mean_y = sum(xs[1]) / n
    diff_x = [value - mean_x for value in xs[0]]
    diff_y = [value - mean_y for value in xs[1]]
    numerator = sum(a * b for a, b in zip(diff_x, diff_y, strict=True))
    sum_x = math.sqrt(sum(value * value for value in diff_x))
    sum_y = math.sqrt(sum(value * value for value in diff_y))
    if sum_x == 0.0 or sum_y == 0.0:
        return None
    return numerator / (sum_x * sum_y)


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Spearman rank correlation with midranks, or ``None`` if undefined."""
    xs, ys = _pair(left, right, "scores")
    return pearson(_average_ranks(xs), _average_ranks(ys))


def cohens_kappa(
    judge_labels: Sequence[int], human_labels: Sequence[int]
) -> tuple[float | None, float]:
    """Cohen's kappa and raw agreement for binary labels in ``{0, 1}``.

    Returns ``(kappa, raw_agreement)``. ``kappa`` is ``None`` when the
    chance agreement is 1 and the raters are not in full agreement.
    """
    judge_bin, human_bin = _binary_pair(judge_labels, human_labels)
    both_pass, judge_only, human_only, both_fail = _confusion(judge_bin, human_bin)
    return _kappa_from_counts(both_pass, judge_only, human_only, both_fail)


def calibrate(
    judge_scores: Sequence[float],
    human_scores: Sequence[float],
    *,
    threshold: float = 0.5,
    notes: Sequence[str] = (),
) -> CalibrationReport:
    """Compare judge scores with human scores.

    ``threshold`` binarises both series: a score ``>= threshold`` is a pass.
    MAE is the mean absolute error on the raw scores.
    """
    judge, human = _pair(judge_scores, human_scores, "scores")
    if not judge:
        raise ValueError("calibrate requires at least one pair of scores")
    cut = _threshold(threshold)
    judge_bin = [1 if score >= cut else 0 for score in judge]
    human_bin = [1 if score >= cut else 0 for score in human]
    both_pass, judge_only, human_only, both_fail = _confusion(judge_bin, human_bin)
    kappa, agreement = _kappa_from_counts(both_pass, judge_only, human_only, both_fail)
    return CalibrationReport(
        n=len(judge),
        pearson=pearson(judge, human),
        spearman=spearman(judge, human),
        mae=sum(abs(a - b) for a, b in zip(judge, human, strict=True)) / len(judge),
        threshold=cut,
        kappa=kappa,
        raw_agreement=agreement,
        both_pass=both_pass,
        judge_only=judge_only,
        human_only=human_only,
        both_fail=both_fail,
        judge_scores=judge,
        human_scores=human,
        notes=list(notes),
    )


def position_bias_audit(
    pairs: Sequence[tuple[str, str]],
    judge: JudgeFn,
    *,
    flag_flip_rate: float = 0.5,
) -> PositionBiasAudit:
    """Run each pair in both orders and measure order flips.

    ``judge(first, second)`` returns ``"first"``, ``"second"``, ``"tie"``,
    or the exact text of the preferred output. An always-first judge has
    flip rate 1 on distinct outputs. A judge that picks the same output in
    both orders has flip rate 0 and is not flagged.
    """
    if not math.isfinite(flag_flip_rate) or not 0.0 <= flag_flip_rate <= 1.0:
        raise ValueError("flag_flip_rate must be in [0, 1]")
    n_flips = 0
    first_wins = 0
    n_pairs = 0
    for left, right in pairs:
        if not isinstance(left, str) or not isinstance(right, str):
            raise TypeError("each pair must be two strings")
        forward = judge(left, right)
        backward = judge(right, left)
        if _winner(left, right, forward) != _winner(right, left, backward):
            n_flips += 1
        if _side(left, right, forward) == "first":
            first_wins += 1
        if _side(right, left, backward) == "first":
            first_wins += 1
        n_pairs += 1
    if n_pairs == 0:
        return PositionBiasAudit(
            n_pairs=0,
            n_flips=0,
            flip_rate=0.0,
            first_position_wins=0,
            first_position_win_rate=0.0,
            flagged=False,
            flag_flip_rate=flag_flip_rate,
        )
    flip_rate = n_flips / n_pairs
    return PositionBiasAudit(
        n_pairs=n_pairs,
        n_flips=n_flips,
        flip_rate=flip_rate,
        first_position_wins=first_wins,
        first_position_win_rate=first_wins / (2 * n_pairs),
        flagged=flip_rate >= flag_flip_rate,
        flag_flip_rate=flag_flip_rate,
    )


def verbosity_bias(pairs: Sequence[tuple[str, str]], judge: JudgeFn) -> VerbosityBias:
    """Correlate ``len(a) - len(b)`` with the judge's preference for ``a``.

    The judge is called once per pair, in the given order. Preference is
    ``+1`` / ``-1`` / ``0``. The correlation is Pearson.
    """
    length_diffs: list[float] = []
    preferences: list[float] = []
    for left, right in pairs:
        if not isinstance(left, str) or not isinstance(right, str):
            raise TypeError("each pair must be two strings")
        side = _side(left, right, judge(left, right))
        length_diffs.append(float(len(left) - len(right)))
        if side == "first":
            preferences.append(1.0)
        elif side == "second":
            preferences.append(-1.0)
        else:
            preferences.append(0.0)
    return VerbosityBias(
        n_pairs=len(length_diffs),
        correlation=pearson(length_diffs, preferences) if length_diffs else None,
        length_diffs=length_diffs,
        preferences=preferences,
    )


def parse_preference(first: str, second: str, raw: str) -> str:
    """Return ``first``, ``second``, or ``tie`` for one pairwise reply.

    Same rules as :func:`position_bias_audit`: the reply may be those words
    (or ``equal`` / ``draw`` / ``same``) or the exact text of one of the two
    outputs (including when an output has leading/trailing whitespace). A
    reply that matches after stripping both sides also counts. When both
    outputs are the same text (exact or after strip) and the reply matches
    that text, the result is ``tie``. Anything else raises ``ValueError``.
    """
    return _side(first, second, raw)


def load_calibration_jsonl(path: str | Path) -> list[CalibrationExample]:
    """Load ``{input, reference, output, human_score}`` rows.

    Blank lines are skipped. ``reference`` may be omitted or null. ``id`` is
    optional. A malformed line raises :class:`ValueError` with the line number.
    """
    examples: list[CalibrationExample] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if raw.strip() == "":
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        examples.append(_example_from_payload(payload, path, line_number))
    if not examples:
        raise ValueError(f"{path}: no calibration rows")
    return examples


def score_with_judge(
    examples: Sequence[CalibrationExample],
    judge: Provider,
    rubric: str,
) -> tuple[list[float], list[str]]:
    """Score each example with the ``llm_judge`` scorer.

    Judge failures stay inside that scorer: the score is 0 and the note is
    returned alongside the scores. This function does not reimplement parsing.
    """
    if not isinstance(rubric, str) or rubric.strip() == "":
        raise ValueError("rubric must be a non-empty string")
    scores: list[float] = []
    notes: list[str] = []
    for index, example in enumerate(examples):
        case = Case(
            id=example.example_id or f"row-{index}",
            input=example.input,
            reference=example.reference,
            assertions=[AssertionSpec(kind="llm_judge", params={"rubric": rubric.strip()})],
        )
        outcome = score_case(case, example.output, context=ScoringContext(judge=judge))
        scores.append(float(outcome[1]))
        notes.extend(outcome.notes)
    return scores, notes


def calibrate_examples(
    examples: Sequence[CalibrationExample],
    judge: Provider,
    rubric: str,
    *,
    threshold: float = 0.5,
) -> CalibrationReport:
    """Run ``llm_judge`` on each example and calibrate against ``human_score``."""
    judge_scores, notes = score_with_judge(examples, judge, rubric)
    human_scores = [example.human_score for example in examples]
    return calibrate(judge_scores, human_scores, threshold=threshold, notes=notes)


def calibration_to_dict(report: CalibrationReport) -> dict[str, object]:
    """JSON-ready calibration report. Undefined correlations are ``None``."""
    return {
        "n": report.n,
        "pearson": _json_float(report.pearson),
        "spearman": _json_float(report.spearman),
        "mae": report.mae,
        "threshold": report.threshold,
        "kappa": _json_float(report.kappa),
        "raw_agreement": report.raw_agreement,
        "both_pass": report.both_pass,
        "judge_only": report.judge_only,
        "human_only": report.human_only,
        "both_fail": report.both_fail,
        "notes": list(report.notes),
        "rows": [
            {"human_score": human, "judge_score": judge}
            for human, judge in zip(report.human_scores, report.judge_scores, strict=True)
        ],
    }


def format_calibration(report: CalibrationReport) -> str:
    """Human table plus a JSON document. Ends with a newline."""
    lines = [
        "calibration",
        f"n={report.n}",
        f"pearson={_fmt_optional(report.pearson)}",
        f"spearman={_fmt_optional(report.spearman)}",
        f"mae={report.mae:.6f}",
        f"threshold={report.threshold:.6f}",
        f"kappa={_fmt_optional(report.kappa)}",
        f"raw_agreement={report.raw_agreement:.6f}",
        f"both_pass={report.both_pass}",
        f"judge_only={report.judge_only}",
        f"human_only={report.human_only}",
        f"both_fail={report.both_fail}",
        "",
        "| human_score | judge_score |",
        "| ---: | ---: |",
    ]
    for human, judge in zip(report.human_scores, report.judge_scores, strict=True):
        lines.append(f"| {human:.6f} | {judge:.6f} |")
    if report.notes:
        lines.extend(["", "notes:"])
        lines.extend(f"- {note}" for note in report.notes)
    lines.extend(["", json.dumps(calibration_to_dict(report), indent=2), ""])
    return "\n".join(lines)


def fleiss_kappa(ratings: Sequence[Sequence[int]]) -> float | None:
    """Fleiss' kappa for equal-length categorical ratings.

    Each inner sequence is one subject and holds one label per rater. Labels
    are compared with ``==``. Returns ``None`` when there are no subjects,
    fewer than two raters, or chance agreement is 1 and the raters are not
    in full agreement. Full agreement when chance is already 1 returns 1.
    """
    if not ratings:
        return None
    n_raters = len(ratings[0])
    if n_raters < 2:
        return None
    if any(len(row) != n_raters for row in ratings):
        raise ValueError("Fleiss' kappa requires the same number of raters on every case")
    categories: list[int] = []
    seen: set[int] = set()
    for row in ratings:
        for label in row:
            if label not in seen:
                seen.add(label)
                categories.append(label)
    n_cases = len(ratings)
    observed_sum = 0.0
    totals = dict.fromkeys(categories, 0)
    pair_scale = float(n_raters * (n_raters - 1))
    for row in ratings:
        counts = dict.fromkeys(categories, 0)
        for label in row:
            counts[label] += 1
            totals[label] += 1
        squared = sum(count * count for count in counts.values())
        observed_sum += (squared - n_raters) / pair_scale
    observed = observed_sum / n_cases
    total_ratings = float(n_cases * n_raters)
    chance = 0.0
    for category in categories:
        proportion = totals[category] / total_ratings
        chance += proportion * proportion
    if math.isclose(chance, 1.0, rel_tol=0.0, abs_tol=1e-12):
        if math.isclose(observed, 1.0, rel_tol=0.0, abs_tol=1e-12):
            return 1.0
        return None
    return (observed - chance) / (1.0 - chance)


def mean_pairwise_absolute(panels: Sequence[Sequence[float]]) -> float:
    """Mean, across panels, of the mean absolute pairwise difference.

    Panel ``i`` is one case. Within a panel every unordered pair contributes
    ``|score_a - score_b|``, those values are averaged, and the per-case
    means are then averaged. Every panel must have at least two scores.
    """
    if not panels:
        raise ValueError("mean pairwise absolute disagreement requires at least one case")
    total = 0.0
    for scores in panels:
        width = len(scores)
        if width < 2:
            raise ValueError("mean pairwise absolute disagreement requires at least two judges")
        pair_total = 0.0
        pair_count = 0
        for left in range(width):
            for right in range(left + 1, width):
                pair_total += abs(scores[left] - scores[right])
                pair_count += 1
        total += pair_total / pair_count
    return total / len(panels)


def judge_agreement_for_run(result: RunResult, side: str) -> JudgeAgreement | None:
    """Agreement of an ensemble panel on ``result``, or ``None`` if there is none.

    A case counts only when it stored at least two judge scores. The panel
    is the label tuple with the most such cases. Ties keep the first label
    tuple in result order. Other panels are excluded and named in ``notes``.
    Pass/fail for Fleiss' kappa uses :data:`JUDGE_PASS_THRESHOLD`.
    """
    groups: dict[tuple[str, ...], list[tuple[float, ...]]] = {}
    order: list[tuple[str, ...]] = []
    for row in result.results:
        if len(row.judge_scores) < 2:
            continue
        labels = tuple(label for label, _score in row.judge_scores)
        scores = tuple(score for _label, score in row.judge_scores)
        if labels not in groups:
            order.append(labels)
            groups[labels] = []
        groups[labels].append(scores)
    if not order:
        return None
    signature = max(order, key=lambda key: (len(groups[key]), -order.index(key)))
    panels = groups[signature]
    notes: list[str] = []
    excluded = sum(len(rows) for key, rows in groups.items() if key != signature)
    if excluded:
        rendered = ", ".join(signature)
        notes.append(f"excluded {excluded} cases whose judge labels were not ({rendered})")
    binary = [
        tuple(1 if score >= JUDGE_PASS_THRESHOLD else 0 for score in panel) for panel in panels
    ]
    width = len(signature)
    per_judge = tuple(sum(panel[index] for panel in panels) / len(panels) for index in range(width))
    return JudgeAgreement(
        side=side,
        config_name=result.config_name,
        n_cases=len(panels),
        n_judges=width,
        mean_pairwise_abs=mean_pairwise_absolute(panels),
        fleiss_kappa=fleiss_kappa(binary),
        pass_threshold=JUDGE_PASS_THRESHOLD,
        labels=signature,
        per_judge_mean=per_judge,
        notes=tuple(notes),
    )


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = ((start + 1) + (end + 1)) / 2.0
        for position in range(start, end + 1):
            ranks[order[position]] = average
        start = end + 1
    return ranks


def _pair(
    left: Sequence[float],
    right: Sequence[float],
    label: str,
) -> tuple[list[float], list[float]]:
    if len(left) != len(right):
        raise ValueError(f"{label} must have the same length")
    xs = [_finite(value, label) for value in left]
    ys = [_finite(value, label) for value in right]
    return xs, ys


def _binary_pair(left: Sequence[int], right: Sequence[int]) -> tuple[list[int], list[int]]:
    if len(left) != len(right):
        raise ValueError("labels must have the same length")
    return [_as_bit(value, "judge label") for value in left], [
        _as_bit(value, "human label") for value in right
    ]


def _as_bit(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError(f"{label} must be 0 or 1")
    return value


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("threshold must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("threshold must be a finite number")
    return number


def _confusion(judge_bin: Sequence[int], human_bin: Sequence[int]) -> tuple[int, int, int, int]:
    both_pass = judge_only = human_only = both_fail = 0
    for judge, human in zip(judge_bin, human_bin, strict=True):
        if judge == 1 and human == 1:
            both_pass += 1
        elif judge == 1 and human == 0:
            judge_only += 1
        elif judge == 0 and human == 1:
            human_only += 1
        else:
            both_fail += 1
    return both_pass, judge_only, human_only, both_fail


def _kappa_from_counts(
    both_pass: int,
    judge_only: int,
    human_only: int,
    both_fail: int,
) -> tuple[float | None, float]:
    n = both_pass + judge_only + human_only + both_fail
    if n == 0:
        return None, 1.0
    agreement = (both_pass + both_fail) / n
    judge_pass = (both_pass + judge_only) / n
    human_pass = (both_pass + human_only) / n
    chance = judge_pass * human_pass + (1.0 - judge_pass) * (1.0 - human_pass)
    if abs(1.0 - chance) <= 1e-15:
        if abs(agreement - 1.0) <= 1e-15:
            return 1.0, agreement
        return None, agreement
    return (agreement - chance) / (1.0 - chance), agreement


def _side(first: str, second: str, raw: str) -> str:
    if not isinstance(raw, str):
        raise TypeError("judge preference must be a string")
    token = raw.strip().casefold()
    if token in {"first", "second", "tie"}:
        return token
    # Exact text of one output (docstring contract). Check the raw reply
    # before any strip so outputs with leading/trailing whitespace still match
    # when the judge returns that exact string. Matching both sides is a tie.
    if raw == first and raw == second:
        return "tie"
    if raw == first:
        return "first"
    if raw == second:
        return "second"
    stripped = raw.strip()
    first_stripped = first.strip()
    second_stripped = second.strip()
    if stripped == first_stripped and stripped == second_stripped:
        return "tie"
    if stripped == first_stripped:
        return "first"
    if stripped == second_stripped:
        return "second"
    if token in {"equal", "draw", "same"}:
        return "tie"
    raise ValueError(
        f"judge preference {raw!r} is not 'first', 'second', 'tie', or one of the two outputs"
    )


def _winner(first: str, second: str, raw: str) -> str | None:
    side = _side(first, second, raw)
    if side == "first":
        return first
    if side == "second":
        return second
    return None


def _example_from_payload(
    payload: object, path: str | Path, line_number: int
) -> CalibrationExample:
    location = f"{path}:{line_number}"
    if not isinstance(payload, dict):
        raise ValueError(f"{location}: row must be a JSON object")
    if "input" not in payload or "output" not in payload or "human_score" not in payload:
        raise ValueError(f"{location}: row needs input, output, and human_score")
    raw_input = payload["input"]
    raw_output = payload["output"]
    if not isinstance(raw_input, str) or not isinstance(raw_output, str):
        raise ValueError(f"{location}: input and output must be strings")
    reference: str | None
    if "reference" not in payload or payload["reference"] is None:
        reference = None
    elif isinstance(payload["reference"], str):
        reference = payload["reference"]
    else:
        raise ValueError(f"{location}: reference must be a string or null")
    example_id = ""
    if "id" in payload and payload["id"] is not None:
        if not isinstance(payload["id"], str):
            raise ValueError(f"{location}: id must be a string")
        example_id = payload["id"]
    try:
        human_score = _finite(payload["human_score"], "human_score")
    except ValueError as exc:
        raise ValueError(f"{location}: {exc}") from exc
    return CalibrationExample(
        input=raw_input,
        output=raw_output,
        human_score=human_score,
        reference=reference,
        example_id=example_id,
    )


def _json_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def _fmt_optional(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.6f}"
