"""Data model for suites, runs, and paired comparisons."""

import math
from dataclasses import dataclass, field
from typing import Any, Literal

Outcome = Literal["win", "loss", "tie"]
PairwiseSide = Literal["baseline", "candidate", "tie"]


@dataclass(frozen=True)
class AssertionSpec:
    """One scoring assertion applied to a model output.

    ``kind`` selects a scorer. ``value`` is the scorer argument (pattern,
    schema, expected number, target text, or a list of trace steps). When
    ``value`` is omitted, text scorers fall back to the case reference.
    ``weight`` is multiplied by the suite's per-scorer weight (default 1)
    when the case score is combined. ``params`` holds scorer-specific options
    such as ``tol``.
    """

    kind: str
    value: Any = None
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Case:
    """One eval item.

    ``source_line`` is the 1-based line of this case in the suite file when
    the loader recorded one, and ``None`` when the line is unknown.
    """

    id: str
    input: str
    reference: str | None
    assertions: list[AssertionSpec]
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_line: int | None = None


@dataclass(frozen=True)
class ScoringConfig:
    """How a suite combines its scorers.

    ``weights`` maps a scorer kind to a non-negative weight. Kinds left out
    of the map use 1. An empty map is equal weights: the case score stays the
    mean already produced from assertion weights (each of those defaults to 1).

    The case score multiplies each assertion weight by the weight of its kind.
    Repeating a kind still counts every assertion. Per-scorer subscores stay
    on ``CaseResult.scores`` as the unweighted mean of that kind.
    """

    weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Suite:
    """A named list of cases. Empty suites are rejected by the loader.

    ``scoring.weights`` reweights scorer kinds. An empty map means equal
    weights, which keeps the historical mean of assertion scores.
    """

    name: str
    cases: list[Case]
    description: str = ""
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


@dataclass(frozen=True)
class CaseResult:
    """Score for one case under one configuration.

    ``scores`` holds one subscore per scorer kind: the unweighted mean of that
    kind's assertions. ``score`` is the weighted mean of the assertions.
    Suite ``scoring.weights`` scales each kind and defaults to equal weights,
    so an omitted map does not change ``score``.

    ``latency_ms`` is the task-provider call only. A cache hit is ``0`` and
    ``cached`` is true, because the provider was not called. Token fields are
    whatever the provider reported. ``MockProvider`` uses character lengths.
    An OpenAI-compatible call uses the response ``usage`` object, or zeros
    when that object is absent. They are not estimated again at report time.

    A multi-sample run stores one score per execution on ``sample_scores``.
    ``score`` is the mean of those scores (the single score when ``n_samples``
    is 1). ``score_sd`` is the population standard deviation of
    ``sample_scores`` (sum of squared deviations divided by N), so one sample
    has sd 0. ``passed`` for one sample is the scorer's own decision. For
    more than one sample, ``passed`` is ``score >= 0.5``. ``output`` is the
    first sample's text, and ``sample_outputs`` keeps every sample in order.
    Latency and token fields sum the samples. ``cached`` is true only when
    every sample was a cache hit.

    ``judge_scores`` is empty unless the case used an ``llm_judge`` ensemble
    (``params.judges``). Each pair is ``(label, score)`` in judge order. A
    single context judge leaves this empty, so older results stay the same.

    ``case_input`` is the suite case input when the runner scored the case.
    It is empty for results built without a suite case, including runs loaded
    from JSON. The HTML report searches and expands it. The JSON report does
    not write the field.

    ``source_line`` is the 1-based line of the case in the suite file when
    the loader recorded one. It is not written to run JSON. ``None`` means
    the line is unknown (a case built in code, or a file the loader could
    not locate).
    """

    case_id: str
    output: str
    scores: dict[str, float]
    score: float
    passed: bool
    latency_ms: float
    error: str | None = None
    tags: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached: bool = False
    sample_scores: list[float] = field(default_factory=list)
    sample_outputs: list[str] = field(default_factory=list)
    score_sd: float = 0.0
    n_samples: int = 1
    judge_scores: tuple[tuple[str, float], ...] = ()
    case_input: str = ""
    source_line: int | None = None


@dataclass(frozen=True)
class CostEstimate:
    """USD estimate for one run under a pricing table.

    ``usd`` is ``None`` when the model is missing from the table and the run
    recorded tokens. Zero token counts are ``0.0`` even when the model is
    unknown. ``note`` explains an unknown model; it is ``None`` when the
    estimate is a number.
    """

    model: str
    usd: float | None
    note: str | None = None


@dataclass(frozen=True)
class RunResult:
    """Every case result for one suite under one named configuration.

    ``mean_score``, ``pass_rate``, ``total_latency_ms``, token totals, and
    ``mean_latency_ms`` / ``p95_latency_ms`` are derived from ``results`` so
    they cannot drift from the per-case rows. p95 uses the same linear
    interpolation as the bootstrap percentile: rank ``0.95 * (n - 1)``.

    ``cost`` is set only when a pricing table was applied. ``None`` means
    cost was not estimated, which is different from an unknown model (that
    is a :class:`CostEstimate` whose ``usd`` is ``None``).
    """

    suite_name: str
    config_name: str
    results: list[CaseResult]
    cost: CostEstimate | None = None

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.score for result in self.results) / len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        passed = sum(1 for result in self.results if result.passed)
        return passed / len(self.results)

    @property
    def total_latency_ms(self) -> float:
        return sum(result.latency_ms for result in self.results)

    @property
    def mean_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return self.total_latency_ms / len(self.results)

    @property
    def p95_latency_ms(self) -> float:
        """95th percentile of per-case latency, linear interpolation.

        An empty run is ``0``. One case returns that case's latency.
        """
        return _linear_percentile([result.latency_ms for result in self.results], 95.0)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(result.prompt_tokens for result in self.results)

    @property
    def total_completion_tokens(self) -> int:
        return sum(result.completion_tokens for result in self.results)

    @property
    def total_tokens(self) -> int:
        """Sum of per-case ``total_tokens``, not a fresh ``prompt + completion``."""
        return sum(result.total_tokens for result in self.results)

    @property
    def n_cached(self) -> int:
        return sum(1 for result in self.results if result.cached)

    @property
    def n_cases(self) -> int:
        return len(self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def mean_within_case_sd(self) -> float:
        """Mean of the per-case population score sds. An empty run is ``0``."""
        if not self.results:
            return 0.0
        return sum(result.score_sd for result in self.results) / len(self.results)

    def by_id(self) -> dict[str, CaseResult]:
        """Map case id to result. Later duplicates overwrite earlier ones."""
        return {result.case_id: result for result in self.results}


@dataclass(frozen=True)
class PairedRow:
    """One case scored under both the baseline and the candidate."""

    case_id: str
    baseline_output: str
    candidate_output: str
    baseline_score: float
    candidate_score: float
    diff: float
    baseline_passed: bool
    candidate_passed: bool
    outcome: Outcome
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SliceSummary:
    """Paired comparison summary for cases that carry one tag.

    A case with several tags is counted in each matching slice. ``n`` is the
    number of cases in the slice, not a partition of the suite.
    """

    tag: str
    n: int
    mean_diff: float
    ci_low: float
    ci_high: float
    wins: int
    losses: int
    ties: int


@dataclass(frozen=True)
class PairwiseCase:
    """One case after a two-order pairwise judgement.

    ``score`` is ``1`` when the candidate wins both presentation orders,
    ``0`` when it loses both, and ``0.5`` on a split or a tie. An unreadable
    judge reply counts as a tie for that order, so it cannot manufacture a
    double win, and the message is stored on ``error``.
    """

    case_id: str
    outcome: Outcome
    score: float
    order_baseline_first: PairwiseSide
    order_candidate_first: PairwiseSide
    error: str | None = None


@dataclass(frozen=True)
class DriftCase:
    """Semantic drift for one paired case. ``drift`` is in ``[0, 2]``."""

    case_id: str
    drift: float


@dataclass(frozen=True)
class DriftSummary:
    """Semantic drift between paired outputs.

    For each case, ``drift = 1 - cosine_similarity`` of the embedder vectors
    of the baseline and candidate outputs. Cosine is clamped to ``[-1, 1]``,
    so drift is in ``[0, 2]``. The value is reported raw. It is not divided
    by 2 and not clipped to ``[0, 1]``. Identical output strings have drift
    ``0`` and are not embedded. Two zero vectors have cosine ``1`` (drift
    ``0``). A zero vector paired with a non-zero vector has cosine ``0``
    (drift ``1``).

    ``per_case`` follows baseline order. ``mean_drift`` is the mean of those
    values (``0`` when there are no cases). ``max_drift`` is the maximum
    (``0`` when there are no cases). ``top`` is the up to five largest
    drifts, ties broken by case id ascending. The recorded case output is
    the first sample when a run used more than one sample.
    """

    embedder: str
    mean_drift: float
    max_drift: float
    per_case: list[DriftCase]
    top: list[DriftCase]


@dataclass(frozen=True)
class PairwiseSummary:
    """Pairwise preference of the candidate over the baseline.

    ``win_rate`` is the mean of the per-case scores, not the share of strict
    wins. An always-first judge therefore lands at ``0.5`` with every case a
    tie. ``ci_low`` and ``ci_high`` are the bootstrap interval for that mean.
    This summary does not change score diffs or the gate.
    """

    rubric: str
    wins: int
    losses: int
    ties: int
    win_rate: float
    ci_low: float
    ci_high: float
    cases: list[PairwiseCase]
    n_boot: int = 2000
    seed: int = 13
    alpha: float = 0.05
    ci_method: str = "percentile"


@dataclass(frozen=True)
class JudgeAgreement:
    """Agreement among ensemble judges on one side of a comparison.

    ``mean_pairwise_abs`` is the mean, across included cases, of the mean
    absolute difference between every pair of judge scores on that case.
    ``fleiss_kappa`` is Fleiss' kappa on pass/fail. A judge passes when its
    score is ``>= pass_threshold`` (the graded pass cut, 0.5). It is ``None``
    when chance agreement is already 1 and the judges are not in full
    agreement. ``labels`` and ``per_judge_mean`` line up by judge index.

    Cases with fewer than two judge scores are not an ensemble. Cases whose
    judge labels differ from the panel used here are counted in ``notes``
    and left out of the numbers. This summary does not change the gate.
    """

    side: str
    config_name: str
    n_cases: int
    n_judges: int
    mean_pairwise_abs: float
    fleiss_kappa: float | None
    pass_threshold: float
    labels: tuple[str, ...]
    per_judge_mean: tuple[float, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Comparison:
    """Paired comparison of a baseline run and a candidate run.

    ``mean_diff`` is the mean of ``candidate_score - baseline_score`` across
    matched cases. A negative value means the candidate scored lower.
    ``ci_low`` and ``ci_high`` are the bootstrap interval for that mean
    (percentile by default, or BCa when requested). ``wins`` counts cases
    where the candidate score is higher, ``losses`` where it is lower, and
    ``ties`` where the scores match. ``per_case`` follows baseline result
    order.

    McNemar counts use pass/fail, which is separate from the score outcome:
    ``mcnemar_b`` is baseline pass and candidate fail, ``mcnemar_c`` is
    baseline fail and candidate pass. Wilcoxon and the paired t-test are
    reported evidence on the score diffs; they do not drive the gate.

    ``pairwise``, when set, is a separate judge of the two outputs. It is
    not an input to ``mean_diff`` or to the gate.

    ``drift``, when set, is semantic drift of the paired outputs. It does
    not change ``mean_diff``. The gate reads it only when a mean-drift
    budget is set.

    ``judge_agreement`` is set when a run scored cases with an ``llm_judge``
    ensemble. It does not change ``mean_diff`` or the gate. An empty tuple
    means no ensemble was recorded.
    """

    baseline: RunResult
    candidate: RunResult
    mean_diff: float
    ci_low: float
    ci_high: float
    mcnemar_p: float
    wins: int
    losses: int
    ties: int
    per_case: list[PairedRow]
    mcnemar_b: int = 0
    mcnemar_c: int = 0
    alpha: float = 0.05
    ci_method: str = "percentile"
    wilcoxon_stat: float = 0.0
    wilcoxon_p: float = 1.0
    ttest_stat: float = 0.0
    ttest_df: float = 0.0
    ttest_p: float = 1.0
    cohens_dz: float = 0.0
    effect_size_label: str = "negligible"
    slices: list[SliceSummary] = field(default_factory=list)
    details: list[str] = field(default_factory=list)
    pairwise: PairwiseSummary | None = None
    drift: DriftSummary | None = None
    judge_agreement: tuple[JudgeAgreement, ...] = ()


def _linear_percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile. Empty input returns ``0``.

    Rank is ``pct/100 * (n - 1)``, the same rule as the bootstrap percentile
    in :mod:`promptregress.stats`. ``pct`` is in ``[0, 100]``.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight
