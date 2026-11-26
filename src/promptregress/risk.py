"""Learned regression-risk ranking from past paired comparisons.

The model estimates which cases are likely to regress on the next prompt
change. A case's label in a stored comparison is 1 when the candidate score
is strictly below the baseline score, and 0 otherwise (a tie or a gain).

Features come from the baseline side and from case metadata only:

- baseline score
- baseline latency in milliseconds (0 when a report omitted it)
- case input length in characters (0 for older reports without a length
  or input text)
- baseline output length in characters
- number of tags
- one indicator per common scorer kind, 1 when that kind is a key of the
  baseline ``scores`` map

They are baseline-side on purpose. The label is a function of the candidate
score, so a feature that used the candidate score, the paired diff, or the
candidate output would leak the answer and could not be computed before the
new run is trusted. The score is a ranking aid. It is not a probability
guarantee, and the compare/run gate does not read it.

Training is full-batch gradient descent on mean binary cross-entropy plus
``(l2 / 2) * ||w||^2``. Weights and the intercept start at 0. Features are
standardized with the training split's mean and population standard
deviation (divide by N). A constant feature uses scale 1, so it stays 0
after centering and its weight stays 0. The intercept is not penalized.
Finite feature values near the float limits are supported. Standardization
rescales an overflowing subtraction, and holdout probabilities use an
exact logit if the standardized values or the float logit overflow.
The learning rate and the iteration count are fixed. Nothing in training
draws a random number.

With two or more comparisons, in the order given (treat that as oldest
first), the last comparison is the holdout and does not affect the
coefficients. With one comparison, the last ``holdout_fraction`` of cases
in baseline order is the holdout. Holdout AUC is the Wilcoxon rank
statistic. Fewer than 20 labeled cases, a single class, or a split that
puts only one class on either side raises :class:`RiskRefusal` (a
``ValueError``) instead of inventing a ranking.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from statistics import mean

from promptregress.irt import run_from_dict
from promptregress.junit_xml import strip_unpaired_surrogates
from promptregress.loading import strip_utf8_bom_text
from promptregress.models import CaseResult, Comparison, RunResult
from promptregress.stats import compare_runs

MIN_LABELED_CASES = 20
DEFAULT_LEARNING_RATE = 0.5
DEFAULT_ITERATIONS = 800
DEFAULT_L2 = 0.01
DEFAULT_HOLDOUT_FRACTION = 0.2
DEFAULT_TOP_FEATURES = 3

COMMON_SCORER_KINDS: tuple[str, ...] = (
    "exact_match",
    "contains",
    "regex",
    "json_valid",
    "json_schema",
    "numeric_tolerance",
    "levenshtein",
    "tfidf_cosine",
    "llm_judge",
    "trace_match",
)

FEATURE_NAMES: tuple[str, ...] = (
    "baseline_score",
    "baseline_latency_ms",
    "input_length",
    "baseline_output_length",
    "n_tags",
    *(f"scorer_{kind}" for kind in COMMON_SCORER_KINDS),
)

RISK_NOTE = (
    "Features use the baseline side and case metadata only, so the ranking "
    "does not look at the candidate run. The score estimates the chance the "
    "candidate score drops below the baseline. It is a ranking aid, not a "
    "guarantee, and it does not change the gate."
)


class RiskRefusal(ValueError):
    """Too little data, or a one-class split, so no ranking is produced.

    This is a ``ValueError``. Callers that catch ``ValueError`` see it.
    ``n_cases`` is the number of labeled cases that were available.
    """

    n_cases: int
    n_comparisons: int

    def __init__(self, message: str, *, n_cases: int, n_comparisons: int) -> None:
        super().__init__(message)
        self.n_cases = n_cases
        self.n_comparisons = n_comparisons


@dataclass(frozen=True)
class LabeledCase:
    """One case in one comparison, with baseline features and a 0/1 label."""

    case_id: str
    comparison_index: int
    features: tuple[float, ...]
    label: int


@dataclass(frozen=True)
class FeatureContribution:
    """One term of the linear score: coefficient times the standardized value."""

    name: str
    coefficient: float
    standardized_value: float
    contribution: float


@dataclass(frozen=True)
class CaseRisk:
    """A case ranked by predicted regression risk, highest first when listed."""

    case_id: str
    probability: float
    contributions: tuple[FeatureContribution, ...]


@dataclass(frozen=True)
class RiskModel:
    """Logistic regression on standardized baseline features.

    ``coefficients`` line up with :data:`FEATURE_NAMES` and with ``means``
    and ``scales``. They were fit on the training split only. ``holdout_auc``
    is the rank-statistic AUC of this same model on the holdout split.
    ``holdout_split`` is ``last_comparison`` or ``last_cases``.
    """

    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    means: tuple[float, ...]
    scales: tuple[float, ...]
    learning_rate: float
    iterations: int
    l2: float
    n_comparisons: int
    n_cases: int
    n_train: int
    n_holdout: int
    holdout_auc: float
    holdout_split: str


def case_features(case: CaseResult) -> tuple[float, ...]:
    """Baseline-side feature row. Order is :data:`FEATURE_NAMES`.

    The candidate result is not an argument. Input length is the character
    count of ``case_input`` when available, then the stored length, then 0.
    Output length is the character count of the baseline output. Scorer
    indicators are presence in ``scores``, not the subscore value.
    """
    case_id = _require_case_id(case, "baseline")
    score = _finite(case.score, f"case {case_id!r} baseline score")
    latency = _finite(case.latency_ms, f"case {case_id!r} baseline latency_ms")
    if not isinstance(case.case_input, str):
        raise ValueError(f"case {case_id!r} input must be a string")
    length = case.input_length
    if length is not None and (
        isinstance(length, bool) or not isinstance(length, int) or length < 0
    ):
        raise ValueError(f"case {case_id!r} input length must be a non-negative integer")
    input_length = len(case.case_input) if case.case_input or length is None else length
    if not isinstance(case.output, str):
        raise ValueError(f"case {case_id!r} baseline output must be a string")
    if not isinstance(case.tags, list) or not all(isinstance(tag, str) for tag in case.tags):
        raise ValueError(f"case {case_id!r} tags must be a list of strings")
    if not isinstance(case.scores, dict) or not all(isinstance(key, str) for key in case.scores):
        raise ValueError(f"case {case_id!r} scores must map scorer kinds to numbers")
    features = (
        score,
        latency,
        _finite(input_length, f"case {case_id!r} input length"),
        float(len(case.output)),
        float(len(case.tags)),
        *(1.0 if kind in case.scores else 0.0 for kind in COMMON_SCORER_KINDS),
    )
    if len(features) != len(FEATURE_NAMES):
        raise ValueError("feature row length does not match FEATURE_NAMES")
    for name, value in zip(FEATURE_NAMES, features, strict=True):
        if not math.isfinite(value):
            raise ValueError(f"case {case_id!r} feature {name} must be finite")
    return features


def regression_label(baseline_score: float, candidate_score: float) -> int:
    """1 when ``candidate_score < baseline_score``, else 0.

    The comparison report's tie band (``1e-12``) is not applied. A tiny
    drop is a regression here and may be a tie in ``compare``.
    """
    baseline = _finite(baseline_score, "baseline score")
    candidate = _finite(candidate_score, "candidate score")
    return 1 if candidate < baseline else 0


def labeled_cases(comparison: Comparison, *, comparison_index: int = 0) -> list[LabeledCase]:
    """One labeled row per case, in baseline order.

    Baseline and candidate case ids must be the same set, with no duplicates
    on either side. An empty comparison raises ``ValueError``.
    """
    if isinstance(comparison_index, bool) or not isinstance(comparison_index, int):
        raise ValueError("comparison_index must be an integer")
    if comparison_index < 0:
        raise ValueError("comparison_index must be >= 0")
    rows: list[LabeledCase] = []
    for baseline, candidate in _pairs(comparison):
        rows.append(
            LabeledCase(
                case_id=baseline.case_id,
                comparison_index=comparison_index,
                features=case_features(baseline),
                label=regression_label(baseline.score, candidate.score),
            )
        )
    return rows


def train(
    comparisons: Sequence[Comparison],
    *,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    iterations: int = DEFAULT_ITERATIONS,
    l2: float = DEFAULT_L2,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> RiskModel:
    """Fit on the training split and score the holdout.

    ``comparisons`` are oldest first. See the module docstring for the
    split, the features, and the cases that raise :class:`RiskRefusal`.
    Malformed comparisons, non-finite inputs, and illegal hyperparameters
    raise ``ValueError``.
    """
    if isinstance(comparisons, (str, bytes)):
        raise ValueError("comparisons must be a sequence of comparisons")
    step = _positive("learning_rate", learning_rate)
    n_iter = _iterations(iterations)
    penalty = _non_negative("l2", l2)
    fraction = _holdout_fraction(holdout_fraction)
    if len(comparisons) == 0:
        raise ValueError("risk training needs at least one comparison")
    examples: list[LabeledCase] = []
    for index, comparison in enumerate(comparisons):
        examples.extend(labeled_cases(comparison, comparison_index=index))
    n_cases = len(examples)
    n_comparisons = len(comparisons)
    if n_cases < MIN_LABELED_CASES:
        raise RiskRefusal(
            f"need at least {MIN_LABELED_CASES} labeled cases to train a risk model; got {n_cases}",
            n_cases=n_cases,
            n_comparisons=n_comparisons,
        )
    counts = _label_counts(examples)
    if counts[0] == 0 or counts[1] == 0:
        only = 0 if counts[1] == 0 else 1
        raise RiskRefusal(
            "risk model needs both classes "
            "(1 = candidate score < baseline score, 0 = otherwise); "
            f"every labeled case has label {only}",
            n_cases=n_cases,
            n_comparisons=n_comparisons,
        )
    train_rows, hold_rows, split_name = _split(examples, n_comparisons, fraction)
    means, scales = _fit_scale([row.features for row in train_rows])
    train_x = [_standardize(row.features, means, scales) for row in train_rows]
    weights, intercept = _fit_logistic(
        train_x,
        [row.label for row in train_rows],
        learning_rate=step,
        iterations=n_iter,
        l2=penalty,
    )
    _require_finite_params(weights, intercept)
    hold_scores = [
        _predict_probability(row.features, means, scales, weights, intercept) for row in hold_rows
    ]
    auc = roc_auc([row.label for row in hold_rows], hold_scores)
    return RiskModel(
        feature_names=FEATURE_NAMES,
        coefficients=tuple(weights),
        intercept=intercept,
        means=means,
        scales=scales,
        learning_rate=step,
        iterations=n_iter,
        l2=penalty,
        n_comparisons=n_comparisons,
        n_cases=n_cases,
        n_train=len(train_rows),
        n_holdout=len(hold_rows),
        holdout_auc=auc,
        holdout_split=split_name,
    )


def predict_proba(model: RiskModel, cases: Sequence[CaseResult]) -> list[float]:
    """Predicted P(regressed) for each baseline case, in input order.

    An empty case list raises ``ValueError``. Duplicate case ids raise
    ``ValueError``. The candidate run is not read.
    """
    rows = _feature_matrix(model, cases, what="predict_proba")
    return [
        _predict_probability(row, model.means, model.scales, model.coefficients, model.intercept)
        for row in rows
    ]


def rank_cases(
    model: RiskModel,
    cases: Sequence[CaseResult],
    *,
    top_features: int = DEFAULT_TOP_FEATURES,
) -> list[CaseRisk]:
    """Cases sorted by descending risk, then by case id.

    Each row carries the ``top_features`` contributions with the largest
    absolute value (coefficient times standardized feature). Ties in that
    absolute value break by feature name. Pass baseline cases: the features
    are the baseline side of the case you want to rank.
    """
    if isinstance(top_features, bool) or not isinstance(top_features, int) or top_features < 1:
        raise ValueError("top_features must be an integer >= 1")
    kept = min(top_features, len(model.feature_names))
    ranked: list[CaseRisk] = []
    for case, probability in zip(cases, predict_proba(model, cases), strict=True):
        standardized = _standardize(case_features(case), model.means, model.scales)
        contributions = _top_contributions(model, standardized, kept)
        ranked.append(
            CaseRisk(
                case_id=case.case_id,
                probability=probability,
                contributions=tuple(contributions),
            )
        )
    ranked.sort(key=lambda item: (-item.probability, item.case_id))
    return ranked


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """AUC by the Mann-Whitney rank statistic.

    Ties in ``scores`` take the average rank. Equal scores contribute 0.5
    to the pairwise count, so a constant score has AUC 0.5 when both
    classes are present. One class, a length mismatch, an empty input, a
    label other than 0 or 1, or a non-finite score raises ``ValueError``.
    """
    if isinstance(labels, (str, bytes)) or isinstance(scores, (str, bytes)):
        raise ValueError("labels and scores must be sequences")
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if len(labels) == 0:
        raise ValueError("AUC needs at least one labeled case")
    parsed_labels: list[int] = []
    parsed_scores: list[float] = []
    for index, (label, score) in enumerate(zip(labels, scores, strict=True)):
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(f"labels[{index}] must be 0 or 1")
        parsed_labels.append(label)
        parsed_scores.append(_finite(score, f"scores[{index}]"))
    n_pos = sum(parsed_labels)
    n_neg = len(parsed_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes")
    order = sorted(range(len(parsed_scores)), key=lambda index: (parsed_scores[index], index))
    ranks = [0.0] * len(parsed_scores)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and parsed_scores[order[end + 1]] == parsed_scores[order[start]]:
            end += 1
        average = (start + 1 + end + 1) / 2.0
        for cursor in range(start, end + 1):
            ranks[order[cursor]] = average
        start = end + 1
    sum_pos = sum(ranks[index] for index, label in enumerate(parsed_labels) if label == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_comparison(path: str | Path) -> Comparison:
    """Load a ``compare --report-json`` file.

    Standard reports store ``input_length`` in characters. Older or custom
    files may provide ``case_input`` instead. When neither is present, the
    input length feature is 0.
    A history JSONL file is rejected: those rows are headline metrics, not
    per-case pairs. Invalid UTF-8, invalid JSON, and a report that is not a
    comparison raise ``ValueError``.

    The runs keep their original scores and pass bits. ``compare_runs``
    rescales differences for the statistical calculations when finite scores
    produce an oversized difference.
    """
    target = Path(path)
    payload = _read_json_object(target)
    baseline = _run_side(payload, "baseline", target)
    candidate = _run_side(payload, "candidate", target)
    # Validate before compare_runs so a non-finite score fails here, with
    # a risk message, instead of inside the bootstrap.
    labeled_cases(
        Comparison(
            baseline=baseline,
            candidate=candidate,
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
    return compare_runs(baseline, candidate)


def risk_to_dict(
    model: RiskModel,
    ranking: Sequence[CaseRisk] | None = None,
    *,
    target: str | None = None,
    observed: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """JSON-ready training report, plus an optional target ranking."""
    payload: dict[str, object] = {
        "n_comparisons": model.n_comparisons,
        "n_cases": model.n_cases,
        "n_train": model.n_train,
        "n_holdout": model.n_holdout,
        "holdout_auc": model.holdout_auc,
        "holdout_split": model.holdout_split,
        "iterations": model.iterations,
        "learning_rate": model.learning_rate,
        "l2": model.l2,
        "intercept": model.intercept,
        "feature_names": list(model.feature_names),
        "coefficients": list(model.coefficients),
        "means": list(model.means),
        "scales": list(model.scales),
        "note": RISK_NOTE,
        "target": target,
        "ranking": None
        if ranking is None
        else [_case_risk_to_dict(item, observed) for item in ranking],
    }
    return payload


def refusal_to_dict(exc: RiskRefusal) -> dict[str, object]:
    """JSON-ready refusal. ``holdout_auc`` and ``ranking`` are null."""
    return {
        "refused": str(exc),
        "n_comparisons": exc.n_comparisons,
        "n_cases": exc.n_cases,
        "n_train": None,
        "n_holdout": None,
        "holdout_auc": None,
        "ranking": None,
        "note": RISK_NOTE,
    }


def format_risk(
    model: RiskModel,
    ranking: Sequence[CaseRisk] | None = None,
    *,
    target: str | None = None,
    observed: Mapping[str, int] | None = None,
) -> str:
    """Text report. Ends with a newline."""
    lines = [
        "risk",
        f"n_comparisons={model.n_comparisons}",
        f"n_cases={model.n_cases}",
        f"n_train={model.n_train}",
        f"n_holdout={model.n_holdout}",
        f"holdout_split={model.holdout_split}",
        f"holdout_auc={model.holdout_auc:.6f}",
        f"iterations={model.iterations}",
        f"learning_rate={model.learning_rate:.6f}",
        f"l2={model.l2:.6f}",
        f"note={RISK_NOTE}",
    ]
    if ranking is not None:
        label = target if target is not None else ""
        lines.append(f"target={label}")
        for item in ranking:
            observed_bit = _observed_bit(item.case_id, observed)
            suffix = "" if observed_bit is None else f" observed_regressed={observed_bit}"
            lines.append(f"- {item.case_id} probability={item.probability:.6f}{suffix}")
            for contrib in item.contributions:
                lines.append(
                    f"  {contrib.name} contribution={contrib.contribution:.6f} "
                    f"coefficient={contrib.coefficient:.6f} "
                    f"standardized_value={contrib.standardized_value:.6f}"
                )
    return strip_unpaired_surrogates("\n".join(lines) + "\n")


def format_refusal(exc: RiskRefusal) -> str:
    """Text report when the model refuses to rank. Ends with a newline."""
    lines = [
        "risk",
        f"n_comparisons={exc.n_comparisons}",
        f"n_cases={exc.n_cases}",
        f"refused={exc}",
        f"note={RISK_NOTE}",
    ]
    return strip_unpaired_surrogates("\n".join(lines) + "\n")


def write_risk_json(payload: Mapping[str, object], path: str | Path) -> None:
    """Write ``payload`` as indented JSON. Rejects NaN and Infinity."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _case_risk_to_dict(item: CaseRisk, observed: Mapping[str, int] | None) -> dict[str, object]:
    body: dict[str, object] = {
        "case_id": strip_unpaired_surrogates(item.case_id),
        "probability": item.probability,
        "contributions": [
            {
                "name": contrib.name,
                "coefficient": contrib.coefficient,
                "standardized_value": contrib.standardized_value,
                "contribution": contrib.contribution,
            }
            for contrib in item.contributions
        ],
    }
    bit = _observed_bit(item.case_id, observed)
    if bit is not None:
        body["observed_regressed"] = bit == 1
    return body


def _observed_bit(case_id: str, observed: Mapping[str, int] | None) -> int | None:
    if observed is None or case_id not in observed:
        return None
    value = observed[case_id]
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError(f"observed label for {case_id!r} must be 0 or 1")
    return value


def _top_contributions(
    model: RiskModel,
    standardized: Sequence[float],
    kept: int,
) -> list[FeatureContribution]:
    items: list[FeatureContribution] = []
    for name, coefficient, value in zip(
        model.feature_names, model.coefficients, standardized, strict=True
    ):
        contribution = coefficient * value
        if not math.isfinite(contribution):
            raise ValueError(f"feature {name} contribution must be finite")
        items.append(
            FeatureContribution(
                name=name,
                coefficient=coefficient,
                standardized_value=value,
                contribution=contribution,
            )
        )
    items.sort(key=lambda item: (-abs(item.contribution), item.name))
    return items[:kept]


def _feature_matrix(
    model: RiskModel,
    cases: Sequence[CaseResult],
    *,
    what: str,
) -> list[tuple[float, ...]]:
    if isinstance(cases, (str, bytes)):
        raise ValueError(f"{what} cases must be a sequence of case results")
    if len(cases) == 0:
        raise ValueError(f"{what} needs at least one case")
    ids = [_require_case_id(case, "baseline") for case in cases]
    _reject_duplicate_ids(ids, "baseline")
    rows: list[tuple[float, ...]] = []
    width = len(model.feature_names)
    if width != len(FEATURE_NAMES) or len(model.coefficients) != width:
        raise ValueError("model feature width does not match FEATURE_NAMES")
    if len(model.means) != width or len(model.scales) != width:
        raise ValueError("model standardization width does not match FEATURE_NAMES")
    for case in cases:
        row = case_features(case)
        if len(row) != width:
            raise ValueError(
                f"case {case.case_id!r} has {len(row)} features; the model expects {width}"
            )
        rows.append(row)
    return rows


def _pairs(comparison: Comparison) -> list[tuple[CaseResult, CaseResult]]:
    baseline_results = comparison.baseline.results
    candidate_results = comparison.candidate.results
    if not isinstance(baseline_results, list) or not isinstance(candidate_results, list):
        raise ValueError("comparison results must be lists")
    if len(baseline_results) == 0 or len(candidate_results) == 0:
        raise ValueError("comparison has no cases")
    baseline_ids = [_require_case_id(case, "baseline") for case in baseline_results]
    candidate_ids = [_require_case_id(case, "candidate") for case in candidate_results]
    _reject_duplicate_ids(baseline_ids, "baseline")
    _reject_duplicate_ids(candidate_ids, "candidate")
    if set(baseline_ids) != set(candidate_ids):
        missing_from_candidate = sorted(set(baseline_ids) - set(candidate_ids))
        missing_from_baseline = sorted(set(candidate_ids) - set(baseline_ids))
        raise ValueError(
            "baseline and candidate case ids do not match: "
            f"missing from candidate {missing_from_candidate}, "
            f"missing from baseline {missing_from_baseline}"
        )
    by_id = {case.case_id: case for case in candidate_results}
    pairs: list[tuple[CaseResult, CaseResult]] = []
    for case in baseline_results:
        other = by_id[case.case_id]
        _finite(other.score, f"case {case.case_id!r} candidate score")
        pairs.append((case, other))
    return pairs


def _reject_duplicate_ids(ids: Sequence[str], side: str) -> None:
    seen: set[str] = set()
    for case_id in ids:
        if case_id in seen:
            raise ValueError(f"{side} run repeats case id {case_id!r}")
        seen.add(case_id)


def _split(
    examples: Sequence[LabeledCase],
    n_comparisons: int,
    holdout_fraction: float,
) -> tuple[list[LabeledCase], list[LabeledCase], str]:
    if n_comparisons >= 2:
        holdout_index = examples[-1].comparison_index
        train_rows = [row for row in examples if row.comparison_index != holdout_index]
        hold_rows = [row for row in examples if row.comparison_index == holdout_index]
        split_name = "last_comparison"
    else:
        n = len(examples)
        holdout_n = int(round(holdout_fraction * n))
        holdout_n = max(1, min(n - 1, holdout_n))
        train_rows = list(examples[:-holdout_n])
        hold_rows = list(examples[-holdout_n:])
        split_name = "last_cases"
    if not train_rows or not hold_rows:
        raise RiskRefusal(
            "holdout split produced an empty train or holdout set",
            n_cases=len(examples),
            n_comparisons=n_comparisons,
        )
    train_counts = _label_counts(train_rows)
    hold_counts = _label_counts(hold_rows)
    if train_counts[0] == 0 or train_counts[1] == 0 or hold_counts[0] == 0 or hold_counts[1] == 0:
        raise RiskRefusal(
            "holdout AUC needs both classes in the train split and the holdout split; "
            f"train labels={train_counts[0]} class-0 / {train_counts[1]} class-1, "
            f"holdout labels={hold_counts[0]} class-0 / {hold_counts[1]} class-1",
            n_cases=len(examples),
            n_comparisons=n_comparisons,
        )
    return train_rows, hold_rows, split_name


def _label_counts(rows: Sequence[LabeledCase]) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for row in rows:
        if row.label not in counts:
            raise ValueError(f"label for {row.case_id!r} must be 0 or 1")
        counts[row.label] += 1
    return counts


def _population_std(values: Sequence[float]) -> float:
    """Population standard deviation that stays finite for huge finite values.

    ``statistics.pstdev`` on Python 3.10 sums squared deviations as exact
    fractions and converts the result with ``float(Fraction)``; for values
    like ``1e200`` the squared deviation (~``1e400``) overflows that
    conversion with ``OverflowError``. (Python 3.12's ``statistics`` avoids
    the exact path, so the same call passes there and CI went red only on
    the 3.10 job.) Shift by the first value before centering: the shift
    keeps the tiny differences between near-equal values exact
    (``0.5 +/- 1e-13`` must keep its ``1e-13`` deviation to 1e-12 relative,
    which a rounded float mean destroys), and scaling the deviations by the
    largest one keeps every square in ``[-1, 1]``. When that shift is not
    finite, or ``math.fsum`` overflows on the shifted residuals, scale by
    the largest magnitude first and center in those units. Identical values
    give exactly 0.0, ``(1e200, 2e200)`` gives exactly ``5e199``, and finite
    inputs produce a finite result.
    """
    if not values:
        raise ValueError("population std needs at least one value")
    deviations = _centered_deviations(values)
    if deviations is not None:
        return _scaled_std(deviations)
    peak = max(abs(value) for value in values)
    if peak == 0.0 or not math.isfinite(peak):
        return 0.0
    scaled = _centered_deviations([value / peak for value in values])
    if scaled is None:
        return 0.0
    return peak * _scaled_std(scaled)


def _centered_deviations(values: Sequence[float]) -> list[float] | None:
    """Deviations from the mean after shifting by the first value.

    ``None`` when a residual or the shifted mean is not finite, including
    when ``math.fsum`` raises ``OverflowError`` on the shifted values.
    """
    ref = values[0]
    shifted: list[float] = []
    for value in values:
        gap = value - ref
        if not math.isfinite(gap):
            return None
        shifted.append(gap)
    try:
        center = math.fsum(shifted) / len(shifted)
    except OverflowError:
        return None
    if not math.isfinite(center):
        return None
    deviations: list[float] = []
    for value in shifted:
        gap = value - center
        if not math.isfinite(gap):
            return None
        deviations.append(gap)
    return deviations


def _scaled_std(deviations: Sequence[float]) -> float:
    """Population standard deviation. Squares are scaled into ``[0, 1]``."""
    top = max((abs(value) for value in deviations), default=0.0)
    if top == 0.0:
        return 0.0
    scaled = [value / top for value in deviations]
    variance = math.fsum(value * value for value in scaled) / len(scaled)
    return top * math.sqrt(variance)


def _fit_scale(
    rows: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    n = len(rows)
    if n == 0:
        raise ValueError("standardization needs at least one row")
    width = len(rows[0])
    means: list[float] = []
    scales: list[float] = []
    if any(len(row) != width for row in rows):
        raise ValueError("feature rows must have the same length")
    for column in range(width):
        values = [row[column] for row in rows]
        # Shift-then-scale, falling back to a scale-first population std when
        # the shift overflows. Identical values give exactly 0.0.
        center = mean(values)
        std = _population_std(values)
        means.append(center)
        scales.append(1.0 if std == 0.0 else std)
    return tuple(means), tuple(scales)


def _standardize(
    row: Sequence[float],
    means: Sequence[float],
    scales: Sequence[float],
) -> list[float]:
    if not (len(row) == len(means) == len(scales)):
        raise ValueError("feature row length does not match the training standardization")
    standardized = [
        _standardized_value(value, center, scale)
        for value, center, scale in zip(row, means, scales, strict=True)
    ]
    if any(not math.isfinite(value) for value in standardized):
        raise ValueError("standardized features must be finite")
    return standardized


def _standardized_value(value: float, center: float, scale: float) -> float:
    gap = value - center
    if math.isinf(gap):
        # Rescale before subtracting so a representable quotient survives.
        return ((value * 0.5 - center * 0.5) / scale) * 2.0
    return gap / scale


def _fit_logistic(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    learning_rate: float,
    iterations: int,
    l2: float,
) -> tuple[list[float], float]:
    """Full-batch gradient descent from a zero start. No RNG."""
    if len(rows) != len(labels) or len(rows) == 0:
        raise ValueError("logistic regression needs one label per row")
    width = len(rows[0])
    weights = [0.0] * width
    intercept = 0.0
    n = float(len(rows))
    for _ in range(iterations):
        grad_w = [0.0] * width
        grad_b = 0.0
        for row, label in zip(rows, labels, strict=True):
            probability = _sigmoid(_logit(weights, intercept, row))
            error = probability - float(label)
            for index, value in enumerate(row):
                grad_w[index] += error * value
            grad_b += error
        for index in range(width):
            weights[index] -= learning_rate * (grad_w[index] / n + l2 * weights[index])
        intercept -= learning_rate * (grad_b / n)
    return weights, intercept


def _logit(weights: Sequence[float], intercept: float, row: Sequence[float]) -> float:
    score = intercept
    for weight, value in zip(weights, row, strict=True):
        score += weight * value
    return score


def _predict_probability(
    row: Sequence[float],
    means: Sequence[float],
    scales: Sequence[float],
    weights: Sequence[float],
    intercept: float,
) -> float:
    standardized = [
        _standardized_value(value, center, scale)
        for value, center, scale in zip(row, means, scales, strict=True)
    ]
    score = _logit(weights, intercept, standardized)
    if not math.isfinite(score):
        # A holdout value can sit far outside the training distribution.
        # Compute its logit exactly so large contributions can still cancel.
        exact = Fraction(intercept)
        for weight, value, center, scale in zip(weights, row, means, scales, strict=True):
            if weight:
                exact += Fraction(weight) * (Fraction(value) - Fraction(center)) / Fraction(scale)
        try:
            score = float(exact)
        except OverflowError:
            return float(exact > 0)
    return _sigmoid(score)


def _require_finite_params(weights: Sequence[float], intercept: float) -> None:
    if not math.isfinite(intercept):
        raise ValueError("logistic regression produced a non-finite intercept")
    for weight in weights:
        if not math.isfinite(weight):
            raise ValueError("logistic regression produced a non-finite coefficient")


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _require_case_id(case: CaseResult, side: str) -> str:
    if not isinstance(case.case_id, str) or case.case_id == "":
        raise ValueError(f"{side} case_id must be a non-empty string")
    return case.case_id


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _iterations(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("iterations must be an integer >= 1")
    return value


def _positive(label: str, value: float) -> float:
    number = _finite(value, label)
    if number <= 0.0:
        raise ValueError(f"{label} must be a finite number > 0")
    return number


def _non_negative(label: str, value: float) -> float:
    number = _finite(value, label)
    if number < 0.0:
        raise ValueError(f"{label} must be a finite number >= 0")
    return number


def _holdout_fraction(value: float) -> float:
    number = _finite(value, "holdout_fraction")
    if number <= 0.0 or number >= 1.0:
        raise ValueError("holdout_fraction must be a finite number in (0, 1)")
    return number


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        text = strip_utf8_bom_text(path.read_text(encoding="utf-8-sig"))
    except UnicodeError as exc:
        raise ValueError(f"{path}: not valid UTF-8: {exc}") from exc
    if text.strip() == "":
        raise ValueError(f"{path}: file is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        if _looks_like_history(text):
            raise ValueError(
                f"{path}: this is a history JSONL file of headline metrics, not a "
                "per-case comparison. Pass comparison reports written by "
                "compare --report-json."
            ) from exc
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: comparison report must be a JSON object")
    if "baseline" not in payload or "candidate" not in payload:
        if payload.get("command") in {"run", "compare"} and "metrics" in payload:
            raise ValueError(
                f"{path}: this is a history record of headline metrics, not a "
                "per-case comparison. Pass comparison reports written by "
                "compare --report-json."
            )
        raise ValueError(f"{path}: expected a comparison report with baseline and candidate")
    return payload


def _looks_like_history(text: str) -> bool:
    for line in text.splitlines():
        raw = line.strip().lstrip("\ufeff")
        if raw == "":
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            return False
        return (
            isinstance(row, dict) and row.get("command") in {"run", "compare"} and "metrics" in row
        )
    return False


def _run_side(payload: Mapping[str, object], side: str, path: Path) -> RunResult:
    raw = payload.get(side)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: comparison report needs an object field {side!r}")
    result = run_from_dict(raw, label=f"{path} {side}")
    results_raw = raw.get("results")
    if not isinstance(results_raw, list) or len(results_raw) != len(result.results):
        raise ValueError(f"{path}: {side}.results does not match the loaded run")
    updated: list[CaseResult] = []
    for case, item in zip(result.results, results_raw, strict=True):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: {side} results must be objects")
        if "case_input" not in item or item["case_input"] is None:
            updated.append(case)
            continue
        text = item["case_input"]
        if not isinstance(text, str):
            raise ValueError(f"{path}: {side} case {case.case_id!r} case_input must be a string")
        updated.append(replace(case, case_input=text))
    return replace(result, results=updated)
