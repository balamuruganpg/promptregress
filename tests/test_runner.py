"""Prompt templates, cache integration, errors, and worker order."""

import threading
from pathlib import Path

import pytest

from promptregress.cache import ResponseCache
from promptregress.models import AssertionSpec, Case, Suite
from promptregress.providers import EchoProvider, MockProvider
from promptregress.runner import format_prompt, run_suite
from promptregress.scorers import ScoringContext


def _suite() -> Suite:
    return Suite(
        name="s",
        cases=[
            Case(
                id="a",
                input="hello",
                reference="ok",
                assertions=[AssertionSpec("exact_match", value="ok")],
            )
        ],
    )


def test_format_prompt_does_not_rescan_inserted_text() -> None:
    case = Case(id="c", input="{reference}", reference="SECRET", assertions=[])
    assert format_prompt("{input}", case) == "{reference}"
    assert format_prompt("X {reference} Y {input}", case) == "X SECRET Y {reference}"
    assert format_prompt("none", case) == "none"
    missing = Case(id="c", input="Q", reference=None, assertions=[])
    assert format_prompt("[{reference}]", missing) == "[]"


def test_runner_copies_case_tags() -> None:
    suite = Suite(
        name="s",
        cases=[
            Case(
                id="a",
                input="hello",
                reference="hello",
                assertions=[AssertionSpec("exact_match")],
                tags=["greet", "smoke"],
            )
        ],
    )
    result = run_suite(suite, EchoProvider(), "echo")
    assert result.results[0].tags == ["greet", "smoke"]


def test_template_is_applied() -> None:
    provider = MockProvider(rules=[("PREFIX hello", "ok")], default="no")
    result = run_suite(_suite(), provider, "cfg", prompt_template="PREFIX {input}")
    assert result.results[0].output == "ok"
    assert result.results[0].passed is True
    assert result.results[0].error is None
    assert result.mean_score == 1.0
    assert result.pass_rate == 1.0
    assert result.config_name == "cfg"
    assert result.suite_name == "s"
    assert result.n_cases == 1
    assert result.total_latency_ms >= 0


def test_reference_placeholder() -> None:
    suite = Suite(
        name="s",
        cases=[
            Case(
                id="a",
                input="Q",
                reference="R",
                assertions=[AssertionSpec("exact_match", value="I=Q;R=R")],
            )
        ],
    )
    result = run_suite(
        suite,
        EchoProvider(),
        "echo",
        prompt_template="I={input};R={reference}",
    )
    assert result.results[0].output == "I=Q;R=R"
    assert result.results[0].passed is True


def test_runner_uses_cache(tmp_path: Path) -> None:
    suite = Suite(
        name="s",
        cases=[
            Case(
                id="a",
                input="alpha",
                reference="yes",
                assertions=[AssertionSpec("exact_match", value="yes")],
            ),
            Case(
                id="b",
                input="beta",
                reference="yes",
                assertions=[AssertionSpec("exact_match", value="yes")],
            ),
        ],
    )
    provider = MockProvider(rules=[("alpha", "yes"), ("beta", "yes")], default="no")
    cache = ResponseCache(tmp_path)
    first = run_suite(suite, provider, "c", cache=cache)
    assert provider.call_count == 2
    assert cache.misses == 2
    assert cache.hits == 0
    second = run_suite(suite, provider, "c", cache=cache)
    assert provider.call_count == 2
    assert cache.hits == 2
    assert [row.output for row in second.results] == [row.output for row in first.results]
    assert [row.case_id for row in second.results] == ["a", "b"]


def test_provider_error_is_recorded_and_not_cached(tmp_path: Path) -> None:
    class Boom:
        name = "boom"
        model = "boom"

        def complete(self, prompt: str) -> str:
            raise RuntimeError("nope")

    suite = Suite(
        name="s",
        cases=[
            Case(
                id="a",
                input="q",
                reference="a",
                assertions=[AssertionSpec("exact_match")],
            )
        ],
    )
    cache = ResponseCache(tmp_path)
    result = run_suite(suite, Boom(), "c", cache=cache)
    assert result.results[0].passed is False
    assert result.results[0].score == 0.0
    assert result.results[0].error is not None
    assert "nope" in result.results[0].error
    assert result.pass_rate == 0.0
    assert cache.misses == 1
    assert list(tmp_path.glob("*.json")) == []


def test_max_workers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        run_suite(_suite(), EchoProvider(), "c", max_workers=0)


def test_workers_preserve_case_order() -> None:
    names = ["a", "b", "c", "d"]
    suite = Suite(
        name="s",
        cases=[
            Case(
                id=name,
                input=name,
                reference=name,
                assertions=[AssertionSpec("exact_match", value=f"out:{name}")],
            )
            for name in names
        ],
    )

    class Scripted:
        name = "scripted"
        model = "scripted"

        def __init__(self) -> None:
            self.started: list[str] = []
            self._lock = threading.Lock()
            self._release = threading.Event()
            self._started = 0
            self._expected = 4

        def complete(self, prompt: str) -> str:
            with self._lock:
                self.started.append(prompt)
                self._started += 1
                count = self._started
            if count == self._expected:
                self._release.set()
            elif not self._release.wait(timeout=5):
                raise TimeoutError("workers did not run concurrently")
            return f"out:{prompt}"

    provider = Scripted()
    result = run_suite(suite, provider, "parallel", prompt_template="{input}", max_workers=4)
    assert [row.case_id for row in result.results] == names
    assert [row.output for row in result.results] == [f"out:{name}" for name in names]
    assert all(row.passed and row.error is None for row in result.results)
    assert set(provider.started) == set(names)


def _judge_suite() -> Suite:
    return Suite(
        name="s",
        cases=[
            Case(
                id="bad",
                input="bad",
                reference="yes",
                assertions=[AssertionSpec("llm_judge", params={"rubric": "be correct"})],
            ),
            Case(
                id="good",
                input="good",
                reference="yes",
                assertions=[AssertionSpec("llm_judge", params={"rubric": "be correct"})],
            ),
        ],
    )


def test_run_suite_records_judge_parse_failure_on_the_case() -> None:
    judge = MockProvider(default="not-a-score")
    ctx = ScoringContext(judge=judge)
    result = run_suite(_judge_suite(), EchoProvider(), "cfg", context=ctx)
    assert [row.case_id for row in result.results] == ["bad", "good"]
    assert ctx.notes == []
    assert judge.call_count == 2
    for row in result.results:
        assert row.output == row.case_id
        assert row.score == 0.0
        assert row.passed is False
        assert row.error is not None
        assert "not-a-score" in row.error
        assert row.case_id in row.error


def test_run_suite_judge_success_and_provider_failure_stay_apart() -> None:
    class Selective:
        name = "sel"
        model = "sel"

        def complete(self, prompt: str) -> str:
            if "\nInput: bad\n" in prompt:
                raise RuntimeError("bad-judge")
            return "1"

    ctx = ScoringContext(judge=Selective())
    result = run_suite(_judge_suite(), EchoProvider(), "cfg", context=ctx, max_workers=2)
    assert [row.case_id for row in result.results] == ["bad", "good"]
    bad, good = result.results
    assert bad.output == "bad"
    assert bad.score == 0.0
    assert bad.passed is False
    assert bad.error is not None
    assert "bad-judge" in bad.error
    assert good.output == "good"
    assert good.score == 1.0
    assert good.passed is True
    assert good.error is None
    assert ctx.notes == []
