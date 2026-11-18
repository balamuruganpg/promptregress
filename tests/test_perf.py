"""Timing budget for a 1000-case mock comparison with the response cache.

Three runs here took 2.42s, 2.28s, and 2.26s. A profile of one run put about
1.7s in the default 2000-resample bootstrap (overall interval plus one tag
slice) and about 0.6s in the two cache-backed provider runs. One thousand
cache puts with fsync were about 0.14s on their own. ``BUDGET_SECONDS`` is
7.5, about three times the slowest of those runs.
"""

import time
from pathlib import Path

import pytest

from promptregress.cache import ResponseCache
from promptregress.models import AssertionSpec, Case, Suite
from promptregress.providers import MockProvider
from promptregress.runner import run_suite
from promptregress.stats import compare_runs

pytestmark = pytest.mark.perf

# 3x the slowest measured run (2.42s): MockProvider, cache, 2000 resamples.
BUDGET_SECONDS = 7.5
N_CASES = 1000


def _suite(n: int) -> tuple[Suite, list[tuple[str, str]]]:
    cases: list[Case] = []
    rules: list[tuple[str, str]] = []
    for index in range(n):
        case_id = f"c{index:04d}"
        question = f"[{case_id}] What is the status of ticket {index}?"
        reference = f"Ticket {index} is open."
        cases.append(
            Case(
                id=case_id,
                input=question,
                reference=reference,
                assertions=[
                    AssertionSpec(kind="contains", value=f"Ticket {index}"),
                    AssertionSpec(kind="exact_match"),
                ],
                tags=["generated"],
            )
        )
        rules.append((question, reference))
    rules.sort(key=lambda item: len(item[0]), reverse=True)
    suite = Suite(name="perf", description="generated timing suite", cases=cases)
    return suite, rules


def test_thousand_case_compare_within_budget(tmp_path: Path) -> None:
    suite, rules = _suite(N_CASES)
    provider = MockProvider(rules=rules, default="unknown", name="mock", model="perf")
    cache = ResponseCache(tmp_path / "cache")
    started = time.perf_counter()
    baseline = run_suite(
        suite,
        provider,
        "baseline",
        prompt_template="Answer completely.\n{input}",
        cache=cache,
    )
    candidate = run_suite(
        suite,
        provider,
        "candidate",
        prompt_template="Answer briefly.\n{input}",
        cache=cache,
    )
    comparison = compare_runs(baseline, candidate, seed=13)
    elapsed = time.perf_counter() - started
    assert comparison.baseline.n_cases == N_CASES
    assert comparison.candidate.n_cases == N_CASES
    assert comparison.baseline.pass_rate == 1.0
    assert comparison.candidate.pass_rate == 1.0
    assert comparison.mean_diff == 0.0
    assert comparison.ci_low <= comparison.ci_high
    assert comparison.ties == N_CASES
    assert len(comparison.slices) == 1
    assert cache.misses == 2 * N_CASES
    assert cache.hits == 0
    assert any((tmp_path / "cache").glob("*.json"))
    assert elapsed < BUDGET_SECONDS, f"elapsed {elapsed:.3f}s exceeds budget {BUDGET_SECONDS:.1f}s"
