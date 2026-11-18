"""2PL item-response fit for a models-by-cases score matrix.

This is a small, deterministic gradient fit, not a production IRT package.
The simplifications are intentional and should be read with the numbers:

- The model is two-parameter logistic, ``P = sigmoid(a * (theta - b))``.
  There is no guessing parameter (that would be 3PL) and no graded-response
  steps. A score in ``[0, 1]`` is a soft Bernoulli label: the loss is binary
  cross-entropy with that score as ``y``. Binary pass/fail is the special
  case ``y in {0, 1}``.
- Parameters are fit by full-batch gradient descent for a fixed number of
  iterations and a fixed step size. There is no EM cycle, no Newton step,
  no quadrature, and no convergence test. The seed only jitters the start
  by ``±0.01``. The same seed repeats the same fit.
- Discrimination ``a`` is penalised by ``(prior_l2 / 2) * (a - 1)^2``, a
  Gaussian pull toward 1 rather than toward 0, and clipped to
  ``[0.05, 5]``. The pull stops ``a`` from exploding and keeps the scale
  from collapsing. A zero-mean penalty would shrink every item toward "no
  signal".
- After each step, abilities are recentered to mean 0 and every difficulty
  is shifted by the same constant, so ``theta - b`` is unchanged. That
  fixes location. Scale is only weakly identified, through the penalty on
  ``a`` and the clips. Abilities are not EAP or MAP estimates from a
  marginal likelihood.
- An item that everyone passes, or everyone fails, has no finite maximum
  likelihood. ``b`` is clipped to ``[-6, 6]``, so those items pile up at
  a rail. With two models the difficulties are a coarse ordering, not a
  standard error.
- ``low_discrimination``, ``very_hard``, and ``very_easy`` are threshold
  flags on the fitted numbers. They are not hypothesis tests.
"""

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from promptregress.models import CaseResult, CostEstimate, RunResult

_A_MIN = 0.05
_A_MAX = 5.0
_B_LIMIT = 6.0
_THETA_LIMIT = 6.0
_LOGIT_CLIP = 30.0
_DEFAULT_ITERATIONS = 800
_DEFAULT_LR = 0.35
_DEFAULT_PRIOR = 0.02
_DEFAULT_MIN_A = 0.5
_DEFAULT_HARD = 1.0
_DEFAULT_EASY = -1.0
_JITTER = 0.01

_NOTES = [
    "2PL joint gradient descent for a fixed number of steps, not marginal MLE.",
    "Graded scores are soft Bernoulli labels, not a graded-response model.",
    "L2 penalty pulls discrimination toward 1. Abilities are recentered to mean 0.",
    "Difficulty is clipped to [-6, 6]. All-correct and all-wrong items have no finite MLE.",
]


@dataclass(frozen=True)
class CaseIRT:
    """Fitted difficulty and discrimination for one case."""

    case_id: str
    a: float
    b: float
    low_discrimination: bool
    very_hard: bool
    very_easy: bool


@dataclass(frozen=True)
class ModelAbility:
    """Fitted ability. ``rank`` 1 is the highest theta.

    Equal thetas break ties by model name, ascending, and still receive
    distinct ranks so the table has a total order.
    """

    model: str
    theta: float
    rank: int


@dataclass(frozen=True)
class IRTResult:
    """A 2PL fit and the threshold flags derived from it."""

    cases: list[CaseIRT]
    abilities: list[ModelAbility]
    iterations: int
    seed: int
    learning_rate: float
    prior_l2: float
    min_discrimination: float
    hard_threshold: float
    easy_threshold: float
    details: list[str] = field(default_factory=list)


def load_run_report(path: str | Path) -> RunResult:
    """Load a run JSON file written by ``promptregress run``."""
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{target}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{target}: run report must be a JSON object")
    return run_from_dict(payload, label=str(target))


def run_from_dict(payload: dict[str, object], *, label: str = "run report") -> RunResult:
    """Build a :class:`RunResult` from the ``run`` JSON shape.

    Headline aggregates are recomputed from ``results``. Per-case ``output``
    is restored (it is part of the run report). Token counts, cache flags,
    sample lists, ``score_sd``, ``n_samples``, and ``cost`` are restored when
    present and otherwise keep the dataclass defaults, so an older report
    that omitted them still loads.
    """
    if "results" not in payload or "config_name" not in payload:
        raise ValueError(f"{label}: expected a run report with config_name and results")
    raw_results = payload["results"]
    if not isinstance(raw_results, list):
        raise ValueError(f"{label}: results must be a list")
    config_name = payload["config_name"]
    suite_name = payload.get("suite_name", "")
    if not isinstance(config_name, str) or not isinstance(suite_name, str):
        raise ValueError(f"{label}: config_name and suite_name must be strings")
    results: list[CaseResult] = []
    for index, item in enumerate(raw_results):
        results.append(_case_result(item, label, index))
    return RunResult(
        suite_name=suite_name,
        config_name=config_name,
        results=results,
        cost=_parse_cost(payload.get("cost"), label),
    )


def response_matrix(
    runs: Sequence[RunResult],
    *,
    response: str = "score",
) -> tuple[list[str], list[str], list[list[float]], list[str]]:
    """Align runs into ``matrix[model][case]``.

    Case order follows the first run. Every run must contain exactly that
    set of case ids. Model names are ``config_name``; a repeated name gets
    a ``#2`` suffix. ``response`` is ``score`` (the graded score) or
    ``passed`` (1 or 0). Values outside ``[0, 1]`` are clipped and noted.
    """
    if response not in {"score", "passed"}:
        raise ValueError("response must be 'score' or 'passed'")
    if len(runs) < 2:
        raise ValueError("IRT needs at least two runs")
    if not runs[0].results:
        raise ValueError("IRT needs at least one case")
    case_ids = [item.case_id for item in runs[0].results]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"Duplicate case id in {runs[0].config_name}")
    expected = set(case_ids)
    notes: list[str] = []
    matrix: list[list[float]] = []
    names: list[str] = []
    seen: dict[str, int] = {}
    for run in runs:
        found = run.by_id()
        if set(found) != expected or len(found) != len(run.results):
            raise ValueError(
                f"Run {run.config_name!r} case ids do not match {runs[0].config_name!r}. "
                "IRT does not impute missing cases."
            )
        row: list[float] = []
        for case_id in case_ids:
            result = found[case_id]
            if response == "passed":
                value = 1.0 if result.passed else 0.0
            else:
                value = float(result.score)
            if not math.isfinite(value):
                raise ValueError(f"Non-finite score for {run.config_name} case {case_id}")
            if value < 0.0 or value > 1.0:
                notes.append(
                    f"Clipped {run.config_name} case {case_id} from {value:.6f} into [0, 1]."
                )
                value = min(1.0, max(0.0, value))
            row.append(value)
        base = run.config_name
        seen[base] = seen.get(base, 0) + 1
        if seen[base] == 1:
            names.append(base)
        else:
            names.append(f"{base}#{seen[base]}")
        matrix.append(row)
    return names, case_ids, matrix, notes


def fit_2pl(
    matrix: Sequence[Sequence[float]],
    *,
    case_ids: Sequence[str],
    model_names: Sequence[str],
    seed: int = 13,
    iterations: int = _DEFAULT_ITERATIONS,
    learning_rate: float = _DEFAULT_LR,
    prior_l2: float = _DEFAULT_PRIOR,
    min_discrimination: float = _DEFAULT_MIN_A,
    hard_threshold: float = _DEFAULT_HARD,
    easy_threshold: float = _DEFAULT_EASY,
) -> IRTResult:
    """Fit 2PL parameters by seeded full-batch gradient descent.

    ``matrix`` is ``[model][case]`` with entries in ``[0, 1]``. See the
    module docstring for what this fit does not claim.
    """
    rows = _checked_matrix(matrix, case_ids, model_names)
    _check_int(seed, "seed")
    _check_int(iterations, "iterations")
    if iterations < 1:
        raise ValueError("iterations must be an integer >= 1")
    step = _positive("learning_rate", learning_rate)
    prior = _non_negative("prior_l2", prior_l2)
    min_a = _finite_number("min_discrimination", min_discrimination)
    hard = _finite_number("hard_threshold", hard_threshold)
    easy = _finite_number("easy_threshold", easy_threshold)
    if easy > hard:
        raise ValueError("easy_threshold must be <= hard_threshold")
    n_models = len(rows)
    n_cases = len(case_ids)
    rng = random.Random(seed)
    discrimination = [1.0 + rng.uniform(-_JITTER, _JITTER) for _ in range(n_cases)]
    difficulty = [rng.uniform(-_JITTER, _JITTER) for _ in range(n_cases)]
    ability = [rng.uniform(-_JITTER, _JITTER) for _ in range(n_models)]
    for _ in range(iterations):
        grad_a = [0.0] * n_cases
        grad_b = [0.0] * n_cases
        grad_theta = [0.0] * n_models
        for model_index, row in enumerate(rows):
            for case_index, observed in enumerate(row):
                gap = ability[model_index] - difficulty[case_index]
                probability = _sigmoid(discrimination[case_index] * gap)
                residual = probability - observed
                grad_theta[model_index] += residual * discrimination[case_index]
                grad_b[case_index] += residual * (-discrimination[case_index])
                grad_a[case_index] += residual * gap
        for case_index in range(n_cases):
            grad_a[case_index] = grad_a[case_index] / n_models + prior * (
                discrimination[case_index] - 1.0
            )
            grad_b[case_index] /= n_models
            discrimination[case_index] = _clip(
                discrimination[case_index] - step * grad_a[case_index],
                _A_MIN,
                _A_MAX,
            )
            difficulty[case_index] = _clip(
                difficulty[case_index] - step * grad_b[case_index],
                -_B_LIMIT,
                _B_LIMIT,
            )
        for model_index in range(n_models):
            grad_theta[model_index] /= n_cases
            ability[model_index] = _clip(
                ability[model_index] - step * grad_theta[model_index],
                -_THETA_LIMIT,
                _THETA_LIMIT,
            )
        center = sum(ability) / n_models
        for model_index in range(n_models):
            ability[model_index] -= center
        for case_index in range(n_cases):
            difficulty[case_index] = _clip(difficulty[case_index] - center, -_B_LIMIT, _B_LIMIT)
    cases = [
        CaseIRT(
            case_id=case_ids[index],
            a=discrimination[index],
            b=difficulty[index],
            low_discrimination=discrimination[index] < min_a,
            very_hard=difficulty[index] >= hard,
            very_easy=difficulty[index] <= easy,
        )
        for index in range(n_cases)
    ]
    order = sorted(
        range(n_models),
        key=lambda index: (-ability[index], model_names[index]),
    )
    abilities = [
        ModelAbility(model=model_names[index], theta=ability[index], rank=rank)
        for rank, index in enumerate(order, start=1)
    ]
    return IRTResult(
        cases=cases,
        abilities=abilities,
        iterations=iterations,
        seed=seed,
        learning_rate=step,
        prior_l2=prior,
        min_discrimination=min_a,
        hard_threshold=hard,
        easy_threshold=easy,
        details=list(_NOTES),
    )


def fit_runs(
    runs: Sequence[RunResult],
    *,
    response: str = "score",
    seed: int = 13,
    iterations: int = _DEFAULT_ITERATIONS,
    learning_rate: float = _DEFAULT_LR,
    prior_l2: float = _DEFAULT_PRIOR,
    min_discrimination: float = _DEFAULT_MIN_A,
    hard_threshold: float = _DEFAULT_HARD,
    easy_threshold: float = _DEFAULT_EASY,
) -> IRTResult:
    """Build the response matrix from run reports and fit 2PL."""
    names, case_ids, matrix, notes = response_matrix(runs, response=response)
    result = fit_2pl(
        matrix,
        case_ids=case_ids,
        model_names=names,
        seed=seed,
        iterations=iterations,
        learning_rate=learning_rate,
        prior_l2=prior_l2,
        min_discrimination=min_discrimination,
        hard_threshold=hard_threshold,
        easy_threshold=easy_threshold,
    )
    if not notes:
        return result
    return IRTResult(
        cases=result.cases,
        abilities=result.abilities,
        iterations=result.iterations,
        seed=result.seed,
        learning_rate=result.learning_rate,
        prior_l2=result.prior_l2,
        min_discrimination=result.min_discrimination,
        hard_threshold=result.hard_threshold,
        easy_threshold=result.easy_threshold,
        details=[*notes, *result.details],
    )


def irt_to_dict(result: IRTResult) -> dict[str, object]:
    """JSON-ready 2PL fit."""
    return {
        "iterations": result.iterations,
        "seed": result.seed,
        "learning_rate": result.learning_rate,
        "prior_l2": result.prior_l2,
        "min_discrimination": result.min_discrimination,
        "hard_threshold": result.hard_threshold,
        "easy_threshold": result.easy_threshold,
        "abilities": [
            {"model": item.model, "theta": item.theta, "rank": item.rank}
            for item in result.abilities
        ],
        "cases": [
            {
                "case_id": item.case_id,
                "a": item.a,
                "b": item.b,
                "low_discrimination": item.low_discrimination,
                "very_hard": item.very_hard,
                "very_easy": item.very_easy,
            }
            for item in result.cases
        ],
        "details": list(result.details),
    }


def format_irt(result: IRTResult) -> str:
    """Ability table, case table, and JSON. Ends with a newline."""
    lines = [
        "irt",
        f"iterations={result.iterations}",
        f"seed={result.seed}",
        "",
        "| rank | model | theta |",
        "| ---: | --- | ---: |",
    ]
    for ability in result.abilities:
        lines.append(f"| {ability.rank} | {ability.model} | {ability.theta:.6f} |")
    lines.extend(
        [
            "",
            "| case | a | b | flags |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for case in result.cases:
        lines.append(f"| {case.case_id} | {case.a:.6f} | {case.b:.6f} | {_flags(case)} |")
    if result.details:
        lines.extend(["", "details:"])
        lines.extend(f"- {note}" for note in result.details)
    lines.extend(["", json.dumps(irt_to_dict(result), indent=2), ""])
    return "\n".join(lines)


def _flags(item: CaseIRT) -> str:
    flags: list[str] = []
    if item.low_discrimination:
        flags.append("low_discrimination")
    if item.very_hard:
        flags.append("very_hard")
    if item.very_easy:
        flags.append("very_easy")
    if not flags:
        return "-"
    return ",".join(flags)


def _sigmoid(value: float) -> float:
    if value >= _LOGIT_CLIP:
        return 1.0
    if value <= -_LOGIT_CLIP:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _checked_matrix(
    matrix: Sequence[Sequence[float]],
    case_ids: Sequence[str],
    model_names: Sequence[str],
) -> list[list[float]]:
    if len(matrix) != len(model_names):
        raise ValueError("matrix must have one row per model")
    if not case_ids:
        raise ValueError("IRT needs at least one case")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case ids must be unique")
    if not model_names:
        raise ValueError("IRT needs at least one model")
    rows: list[list[float]] = []
    width = len(case_ids)
    for row in matrix:
        if len(row) != width:
            raise ValueError("every matrix row must have one entry per case")
        values: list[float] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("matrix entries must be finite numbers in [0, 1]")
            number = float(value)
            if not math.isfinite(number) or number < 0.0 or number > 1.0:
                raise ValueError("matrix entries must be finite numbers in [0, 1]")
            values.append(number)
        rows.append(values)
    return rows


def _case_result(item: object, label: str, index: int) -> CaseResult:
    if not isinstance(item, dict):
        raise ValueError(f"{label}: results[{index}] must be an object")
    try:
        case_id = item["case_id"]
        score = item["score"]
        passed = item["passed"]
    except KeyError as exc:
        raise ValueError(f"{label}: results[{index}] is missing {exc.args[0]}") from exc
    if not isinstance(case_id, str):
        raise ValueError(f"{label}: results[{index}].case_id must be a string")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"{label}: results[{index}].score must be a number")
    if not isinstance(passed, bool):
        raise ValueError(f"{label}: results[{index}].passed must be a boolean")
    output = item.get("output", "")
    if not isinstance(output, str):
        raise ValueError(f"{label}: results[{index}].output must be a string")
    raw_scores = item.get("scores", {})
    if not isinstance(raw_scores, dict):
        raise ValueError(f"{label}: results[{index}].scores must be an object")
    scores: dict[str, float] = {}
    for key, value in raw_scores.items():
        if (
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(f"{label}: results[{index}].scores must map strings to numbers")
        scores[key] = float(value)
    latency = item.get("latency_ms", 0.0)
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        raise ValueError(f"{label}: results[{index}].latency_ms must be a number")
    error = item.get("error")
    if error is not None and not isinstance(error, str):
        raise ValueError(f"{label}: results[{index}].error must be a string or null")
    raw_tags = item.get("tags", [])
    if not isinstance(raw_tags, list) or not all(isinstance(tag, str) for tag in raw_tags):
        raise ValueError(f"{label}: results[{index}].tags must be a list of strings")
    return CaseResult(
        case_id=case_id,
        output=output,
        scores=scores,
        score=float(score),
        passed=passed,
        latency_ms=float(latency),
        error=error,
        tags=list(raw_tags),
        prompt_tokens=_optional_count(item, "prompt_tokens", label, index),
        completion_tokens=_optional_count(item, "completion_tokens", label, index),
        total_tokens=_optional_count(item, "total_tokens", label, index),
        cached=_optional_bool(item, "cached", label, index),
        sample_scores=_optional_numbers(item, "sample_scores", label, index),
        sample_outputs=_optional_strings(item, "sample_outputs", label, index),
        score_sd=_optional_float(item, "score_sd", label, index),
        n_samples=_optional_samples(item, label, index),
    )


def _parse_cost(raw: object, label: str) -> CostEstimate | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{label}: cost must be an object or null")
    model = raw.get("model")
    if not isinstance(model, str):
        raise ValueError(f"{label}: cost.model must be a string")
    usd_raw = raw.get("usd")
    if usd_raw is None:
        usd = None
    elif isinstance(usd_raw, bool) or not isinstance(usd_raw, (int, float)):
        raise ValueError(f"{label}: cost.usd must be a number or null")
    else:
        usd = float(usd_raw)
    note = raw.get("note")
    if note is not None and not isinstance(note, str):
        raise ValueError(f"{label}: cost.note must be a string or null")
    return CostEstimate(model=model, usd=usd, note=note)


def _optional_count(item: dict[Any, Any], key: str, label: str, index: int) -> int:
    if key not in item:
        return 0
    value = item[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label}: results[{index}].{key} must be a non-negative integer")
    return value


def _optional_bool(item: dict[Any, Any], key: str, label: str, index: int) -> bool:
    if key not in item:
        return False
    value = item[key]
    if not isinstance(value, bool):
        raise ValueError(f"{label}: results[{index}].{key} must be a boolean")
    return value


def _optional_float(item: dict[Any, Any], key: str, label: str, index: int) -> float:
    if key not in item:
        return 0.0
    value = item[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: results[{index}].{key} must be a number")
    return float(value)


def _optional_numbers(item: dict[Any, Any], key: str, label: str, index: int) -> list[float]:
    if key not in item:
        return []
    value = item[key]
    if not isinstance(value, list):
        raise ValueError(f"{label}: results[{index}].{key} must be a list of numbers")
    numbers: list[float] = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise ValueError(f"{label}: results[{index}].{key} must be a list of numbers")
        numbers.append(float(entry))
    return numbers


def _optional_strings(item: dict[Any, Any], key: str, label: str, index: int) -> list[str]:
    if key not in item:
        return []
    value = item[key]
    if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
        raise ValueError(f"{label}: results[{index}].{key} must be a list of strings")
    return list(value)


def _optional_samples(item: dict[Any, Any], label: str, index: int) -> int:
    if "n_samples" not in item:
        return 1
    value = item["n_samples"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label}: results[{index}].n_samples must be an integer >= 1")
    return value


def _check_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")


def _finite_number(label: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _positive(label: str, value: float) -> float:
    number = _finite_number(label, value)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive")
    return number


def _non_negative(label: str, value: float) -> float:
    number = _finite_number(label, value)
    if number < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return number
