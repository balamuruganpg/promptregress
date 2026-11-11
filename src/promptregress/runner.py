"""Run a suite against a provider and score each case."""

import math
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

from promptregress.cache import ResponseCache
from promptregress.models import Case, CaseResult, RunResult, Suite
from promptregress.providers import (
    Completion,
    Provider,
    ProviderError,
    TokenUsage,
    provider_model_name,
)
from promptregress.scorers import ScoringContext, ensemble_judge_scores, score_case


def format_prompt(template: str, case: Case) -> str:
    """Replace ``{input}`` and ``{reference}`` without scanning inserted text.

    Other braces are left alone, so a case whose input contains ``{reference}``
    is not formatted twice. A missing reference becomes an empty string.
    """
    reference = "" if case.reference is None else case.reference
    parts: list[str] = []
    index = 0
    while index < len(template):
        if template.startswith("{input}", index):
            parts.append(case.input)
            index += len("{input}")
            continue
        if template.startswith("{reference}", index):
            parts.append(reference)
            index += len("{reference}")
            continue
        parts.append(template[index])
        index += 1
    return "".join(parts)


def run_suite(
    suite: Suite,
    provider: Provider,
    config_name: str,
    prompt_template: str = "{input}",
    cache: ResponseCache | None = None,
    max_workers: int = 1,
    context: ScoringContext | None = None,
    samples: int = 1,
) -> RunResult:
    """Format each case, call the provider, and score the output.

    When ``cache`` is set, a hit skips the provider. Latency is the provider
    call only, measured with ``time.perf_counter``, not scoring and not the
    cache write. A hit records ``latency_ms`` of ``0`` and ``cached`` true:
    the lookup is not model latency. Token counts from the provider are
    stored on the case and, on a miss, written into the cache so a later hit
    reports the same usage. A provider exception is stored on the case
    result with score 0, ``passed`` false, and zero tokens; it is not cached.
    Unknown scorers still raise. ``llm_judge`` failures do not raise: the
    assertion scores 0 and the note is stored on ``CaseResult.error`` (the
    model output is kept). ``context`` supplies the judge provider. Each
    case gets its own context, so notes from one case are not attached to
    another, including when ``max_workers > 1``. The caller's context is not
    modified. Judge time is scoring time, so it is not part of ``latency_ms``.

    ``max_workers > 1`` uses a thread pool and keeps case order.

    Case scores use ``suite.scoring.weights``. An empty map is equal weights.

    ``samples`` is how many times each case is executed. The default ``1``
    keeps the single-call score, pass bit, output, and token counts. The
    cache key is still provider, model, and prompt, so repeated samples of
    the same prompt share one entry: later samples are hits and replay the
    stored output. Variation across samples shows up on cache misses.
    Samples of one case run in order, and the provider is told the sample
    index when it implements ``complete_at``. With ``samples > 1`` the case
    score is the mean of the sample scores, ``passed`` is that mean ``>= 0.5``,
    and ``score_sd`` is the population standard deviation (divide by N).
    Latency and tokens sum the samples. ``output`` is the first sample.

    An ``llm_judge`` ensemble stores one ``(label, score)`` pair per judge
    on ``CaseResult.judge_scores``. A single context judge leaves that empty.
    With ``samples > 1``, each judge's scores are averaged across samples
    when every sample recorded the same labels. A missing or mismatched
    panel leaves ``judge_scores`` empty.
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be an integer >= 1")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be an integer >= 1")
    cases = list(suite.cases)
    judge = None if context is None else context.judge
    weights = suite.scoring.weights

    def evaluate(case: Case) -> CaseResult:
        taken = [
            _evaluate_case(
                case,
                provider,
                prompt_template,
                cache,
                judge,
                weights,
                sample_index=index,
            )
            for index in range(samples)
        ]
        if samples == 1:
            return taken[0]
        return _aggregate_samples(taken)

    if max_workers == 1 or len(cases) <= 1:
        results = [evaluate(case) for case in cases]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(evaluate, cases))
    return RunResult(suite_name=suite.name, config_name=config_name, results=results)


def _invoke(provider: Provider, prompt: str, *, sample_index: int | None = None) -> Completion:
    if sample_index is not None:
        complete_at = getattr(provider, "complete_at", None)
        if callable(complete_at):
            completion = complete_at(prompt, sample_index)
            if not isinstance(completion, Completion):
                raise ProviderError("complete_at must return a Completion")
            if not isinstance(completion.text, str):
                raise ProviderError("provider returned a non-string completion")
            return completion
    detailed = getattr(provider, "complete_detailed", None)
    if callable(detailed):
        completion = detailed(prompt)
        if not isinstance(completion, Completion):
            raise ProviderError("complete_detailed must return a Completion")
        if not isinstance(completion.text, str):
            raise ProviderError("provider returned a non-string completion")
        return completion
    text = provider.complete(prompt)
    if not isinstance(text, str):
        raise ProviderError("provider returned a non-string completion")
    return Completion(text=text, usage=TokenUsage(0, 0, 0))


def _evaluate_case(
    case: Case,
    provider: Provider,
    prompt_template: str,
    cache: ResponseCache | None,
    judge: Provider | None,
    weights: Mapping[str, float] | None = None,
    sample_index: int | None = None,
) -> CaseResult:
    prompt = format_prompt(prompt_template, case)
    model = provider_model_name(provider)
    if cache is not None:
        cached = cache.get_record(provider.name, model, prompt)
        if cached is not None:
            return _score_output(
                case,
                cached.response,
                latency_ms=0.0,
                prompt_tokens=cached.prompt_tokens,
                completion_tokens=cached.completion_tokens,
                total_tokens=cached.total_tokens,
                cached=True,
                judge=judge,
                weights=weights,
            )
    started = time.perf_counter()
    try:
        completion = _invoke(provider, prompt, sample_index=sample_index)
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        message = f"{type(exc).__name__}: {exc}"
        return CaseResult(
            case_id=case.id,
            output="",
            scores={},
            score=0.0,
            passed=False,
            latency_ms=latency_ms,
            error=message,
            tags=list(case.tags),
            cached=False,
            sample_scores=[0.0],
            sample_outputs=[""],
            score_sd=0.0,
            n_samples=1,
        )
    latency_ms = (time.perf_counter() - started) * 1000.0
    if cache is not None:
        cache.put(
            provider.name,
            model,
            prompt,
            completion.text,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            total_tokens=completion.usage.total_tokens,
        )
    return _score_output(
        case,
        completion.text,
        latency_ms=latency_ms,
        prompt_tokens=completion.usage.prompt_tokens,
        completion_tokens=completion.usage.completion_tokens,
        total_tokens=completion.usage.total_tokens,
        cached=False,
        judge=judge,
        weights=weights,
    )


def _score_output(
    case: Case,
    output: str,
    *,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cached: bool,
    judge: Provider | None,
    weights: Mapping[str, float] | None = None,
) -> CaseResult:
    # A fresh context per case keeps judge notes from crossing cases or threads.
    scored = score_case(case, output, context=ScoringContext(judge=judge), weights=weights)
    scores, score, passed = scored
    error = "; ".join(scored.notes) if scored.notes else None
    return CaseResult(
        case_id=case.id,
        output=output,
        scores=scores,
        score=score,
        passed=passed,
        latency_ms=latency_ms,
        error=error,
        tags=list(case.tags),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached=cached,
        sample_scores=[score],
        sample_outputs=[output],
        score_sd=0.0,
        n_samples=1,
        judge_scores=ensemble_judge_scores(scored.details),
    )


def _aggregate_samples(samples: Sequence[CaseResult]) -> CaseResult:
    """Mean score, population sd, and summed usage across samples of one case.

    ``passed`` is the mean score ``>= 0.5``. That is the multi-sample rule.
    It is not a fresh run of the exact-type 0.999 check. ``output`` is the
    first sample. Kind subscores average across samples, with a missing kind
    counted as 0. ``cached`` is true only when every sample was a cache hit.
    """
    count = len(samples)
    scores = [sample.score for sample in samples]
    mean = sum(scores) / count
    errors = [
        f"sample {index}: {sample.error}" for index, sample in enumerate(samples) if sample.error
    ]
    first = samples[0]
    return CaseResult(
        case_id=first.case_id,
        output=first.output,
        scores=_mean_scores(samples),
        score=mean,
        passed=mean >= 0.5,
        latency_ms=sum(sample.latency_ms for sample in samples),
        error="; ".join(errors) if errors else None,
        tags=list(first.tags),
        prompt_tokens=sum(sample.prompt_tokens for sample in samples),
        completion_tokens=sum(sample.completion_tokens for sample in samples),
        total_tokens=sum(sample.total_tokens for sample in samples),
        cached=all(sample.cached for sample in samples),
        sample_scores=scores,
        sample_outputs=[sample.output for sample in samples],
        score_sd=_population_sd(scores),
        n_samples=count,
        judge_scores=_mean_judge_scores(samples),
    )


def _mean_scores(samples: Sequence[CaseResult]) -> dict[str, float]:
    keys: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        for key in sample.scores:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    count = len(samples)
    return {key: sum(sample.scores.get(key, 0.0) for sample in samples) / count for key in keys}


def _mean_judge_scores(samples: Sequence[CaseResult]) -> tuple[tuple[str, float], ...]:
    """Average each judge across samples when every sample has the same labels."""
    panels = [sample.judge_scores for sample in samples]
    if not panels or any(not panel for panel in panels):
        return ()
    labels = tuple(label for label, _score in panels[0])
    if any(tuple(label for label, _score in panel) != labels for panel in panels):
        return ()
    count = float(len(panels))
    return tuple(
        (label, sum(panel[index][1] for panel in panels) / count)
        for index, label in enumerate(labels)
    )


def _population_sd(values: Sequence[float]) -> float:
    """Population standard deviation. Divide by N. An empty input is ``0``."""
    count = len(values)
    if count == 0:
        return 0.0
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    if variance < 0.0:
        variance = 0.0
    return math.sqrt(variance)
