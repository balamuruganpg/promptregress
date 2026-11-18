"""Regression tests for bugs found in the independent v0.9.0 review cycle #22.

Scope is scorers, providers, and the response cache: JSON-schema enum
equality, judge-score parsing, TF-IDF of token-free text, oversized JSON
integers, non-finite retry settings, and cache keys that contain NUL.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from promptregress.cache import ResponseCache
from promptregress.models import AssertionSpec, Case
from promptregress.providers import (
    AnthropicProvider,
    MockProvider,
    OpenAICompatibleProvider,
)
from promptregress.scorers import ScoreError, ScoringContext, score_case


def _case(
    kind: str,
    output: str,
    value: object = None,
    *,
    params: dict[str, object] | None = None,
    weight: float = 1.0,
) -> tuple[Case, str]:
    case = Case(
        id="c",
        input="q",
        reference=None,
        assertions=[
            AssertionSpec(
                kind,
                value=value,
                weight=weight,
                params={} if params is None else params,
            )
        ],
    )
    return case, output


def _judge(reply: str) -> ScoringContext:
    return ScoringContext(judge=MockProvider(default=reply))


def test_json_schema_enum_does_not_treat_bool_as_number() -> None:
    """JSON Schema ``enum`` must not use Python's ``True == 1`` equality.

    ``type`` already refuses to treat booleans as numbers, but ``enum``
    used ``in``, so ``{"enum": [0, 1]}`` scored ``true`` and ``false`` as
    1, and ``{"enum": [[1]]}`` matched ``[true]``. A nested ``true`` also
    matched ``1``. Compare with JSON equality: booleans match only
    booleans, while ``1`` and ``1.0`` still match, and a value that is
    actually listed still matches.
    """

    def scored(schema: object, document: str) -> float:
        case, output = _case("json_schema", document, value=schema)
        return score_case(case, output)[1]

    assert scored({"enum": [0, 1]}, "true") == 0.0
    assert scored({"enum": [0, 1]}, "false") == 0.0
    assert scored({"enum": [0, 1]}, "0") == 1.0
    assert scored({"enum": [0, 1]}, "1") == 1.0
    assert scored({"enum": [True]}, "1") == 0.0
    assert scored({"enum": [True]}, "true") == 1.0
    assert scored({"type": "number", "enum": [True]}, "1") == 0.0
    assert scored({"type": "boolean", "enum": [1]}, "true") == 0.0
    assert scored({"enum": [1.0]}, "1") == 1.0
    assert scored({"enum": [[1]]}, "[true]") == 0.0
    assert scored({"enum": [[1]]}, "[1]") == 1.0
    assert scored({"enum": [{"ok": True}]}, '{"ok": 1}') == 0.0
    assert scored({"enum": [{"ok": True}]}, '{"ok": true}') == 1.0


def test_oversized_json_integer_scores_zero_instead_of_raising() -> None:
    """A JSON integer past the digit limit must score 0, not raise.

    CPython raises ``ValueError`` (not ``JSONDecodeError``) when an
    integer has more than 4300 digits. ``json_valid``, ``json_schema``,
    and ``trace_match`` only caught ``JSONDecodeError``, so a model
    output of five thousand 9s aborted the run. That failure is a decode
    error: those scorers score 0.
    """
    huge = "9" * 5000
    valid, valid_output = _case("json_valid", huge)
    assert score_case(valid, valid_output)[1] == 0.0
    schema, schema_output = _case("json_schema", "{" + huge + "}", value={"type": "integer"})
    assert score_case(schema, schema_output)[1] == 0.0
    wrapped, wrapped_output = _case("json_valid", '{"n": ' + huge + "}")
    assert score_case(wrapped, wrapped_output)[1] == 0.0
    trace, trace_output = _case("trace_match", "[" + huge + "]", value=[{"tool": "lookup"}])
    assert score_case(trace, trace_output)[1] == 0.0


def test_json_schema_number_accepts_integer_that_does_not_fit_in_float() -> None:
    """``type: number`` must accept a finite integer that does not fit in float.

    ``math.isfinite`` converts via ``float`` and raises ``OverflowError``
    on a 4000-digit integer (still under the 4300-digit parser cap). The
    value is a finite integer, so it is a number. The same overflow must
    not escape an assertion or suite weight: those raise ``ScoreError``.
    """
    digits = "9" * 4000
    case, output = _case("json_schema", digits, value={"type": "number"})
    assert score_case(case, output)[1] == 1.0
    integer, integer_output = _case("json_schema", digits, value={"type": "integer"})
    assert score_case(integer, integer_output)[1] == 1.0
    heavy, heavy_output = _case("exact_match", "a", value="a", weight=10**4000)
    with pytest.raises(ScoreError, match="assertion weight"):
        score_case(heavy, heavy_output)
    weighted, weighted_output = _case("exact_match", "a", value="a")
    with pytest.raises(ScoreError, match="non-negative and finite"):
        score_case(weighted, weighted_output, weights={"exact_match": 10**4000})


def test_llm_judge_large_integer_score_is_zero_not_an_exception() -> None:
    """A JSON score that does not fit in float is a bad score, not a crash.

    ``{"score": <4000 nines>}`` parses as a Python int. ``float`` of that
    int raises ``OverflowError``, which ``llm_judge`` did not catch, so
    the run died. The judge contract is to score 0 and record a note when
    the reply is not a finite number in ``[0, 1]``.
    """
    reply = '{"score": ' + ("9" * 4000) + "}"
    case, output = _case("llm_judge", "answer", params={"rubric": "grade it"})
    scored = score_case(case, output, _judge(reply))
    assert scored[1] == 0.0
    assert scored[2] is False
    assert scored.notes
    assert "not a number" in scored.notes[0]


def test_llm_judge_skips_unparsed_object_and_keeps_later_score() -> None:
    """A JSON object the decoder cannot read must not hide a later score.

    ``raw_decode`` raises ``ValueError`` on an integer past the digit
    limit. That was not a ``JSONDecodeError``, so the scan aborted and a
    following ``{"score": 0.4}`` scored 0. Skip that object the same way
    as other malformed ``{`` and use the later score field.
    """
    reply = '{"noise": ' + ("9" * 5000) + '} {"score": 0.4}'
    case, output = _case("llm_judge", "answer", params={"rubric": "grade it"})
    scored = score_case(case, output, _judge(reply))
    assert scored[1] == pytest.approx(0.4)
    assert scored.notes == ()


def test_llm_judge_fraction_or_percent_label_is_not_a_perfect_score() -> None:
    """``Score: 1/5`` must not be parsed as 1.0.

    The label regex took the leading number and ignored a following
    fraction or percent, so ``Score: 1/2`` and ``Score: 1/5`` both scored
    1 and passed the graded floor. ``{"score": "1/2"}`` already failed
    closed. A ``/`` or ``%`` after the label number is not a bare 0-1
    score, so the judge scores 0. A real ``Score: 0.8`` with trailing
    prose still counts.
    """
    case, output = _case("llm_judge", "answer", params={"rubric": "grade it"})
    for reply in ("Score: 1/5", "Score: 1 / 2", "Score: 0.5%"):
        scored = score_case(case, output, _judge(reply))
        assert scored[1] == 0.0, reply
        assert scored[2] is False
        assert scored.notes
    plain = score_case(case, output, _judge("Score: 0.8 because it helped"))
    assert plain[1] == pytest.approx(0.8)
    assert plain.notes == ()


def test_tfidf_word_mode_does_not_match_different_punctuation() -> None:
    """Word TF-IDF must not score different punctuation-only strings as 1.

    ``\\w+`` drops punctuation, so ``!!!`` and ``???`` were both empty
    token lists and took the both-empty branch. An empty target against
    ``???`` did the same, which contradicts "one side empty scores 0".
    Whitespace-only text is still empty. Two non-empty strings with no
    word tokens score 1 only when they match after stripping and
    casefolding.
    """
    different, different_output = _case("tfidf_cosine", "!!!", value="???")
    assert score_case(different, different_output)[1] == 0.0
    empty, empty_output = _case("tfidf_cosine", "???", value="")
    assert score_case(empty, empty_output)[1] == 0.0
    same, same_output = _case("tfidf_cosine", "!!!", value="!!!")
    assert score_case(same, same_output)[1] == 1.0
    folded, folded_output = _case("tfidf_cosine", "  !!!  ", value="!!!")
    assert score_case(folded, folded_output)[1] == 1.0
    blank, blank_output = _case("tfidf_cosine", "   ", value="")
    assert score_case(blank, blank_output)[1] == 1.0
    words, words_output = _case("tfidf_cosine", "red cat", value="red cat")
    assert score_case(words, words_output)[1] == pytest.approx(1.0)


def test_cache_key_nul_inside_a_field_does_not_alias_another_call(tmp_path: Path) -> None:
    """A NUL inside a model or prompt must not serve another call's response.

    The key is ``provider + NUL + model + NUL + prompt``. That separates
    ``ab`` / ``c`` from ``a`` / ``bc``, but a NUL inside a field shifts the
    same boundary: ``model="mock\\0a", prompt="b"`` hashed like
    ``model="mock", prompt="a\\0b"``, so a get returned the other text.
    Escape ``\\`` and NUL inside fields. Fields without those characters
    keep the historical hash.
    """
    cache = ResponseCache(tmp_path)
    cache.put("mock", "mock\0a", "b", "from-model-nul")
    assert cache.get("mock", "mock", "a\0b") is None
    cache.put("mock", "mock", "a\0b", "from-prompt-nul")
    assert cache.get("mock", "mock\0a", "b") == "from-model-nul"
    assert cache.get("mock", "mock", "a\0b") == "from-prompt-nul"
    material = b"prov\0model\0prompt"
    assert cache.key("prov", "model", "prompt") == hashlib.sha256(material).hexdigest()
    assert cache.key("ab", "c", "d") != cache.key("a", "bc", "d")


def test_provider_rejects_non_finite_backoff_and_timeout() -> None:
    """Non-finite retry settings must be rejected before any request.

    ``backoff_base=inf`` reaches ``time.sleep`` and never returns, so the
    provider error is never raised. ``backoff_base=nan`` makes ``sleep``
    raise ``ValueError`` and drops the original transport error.
    ``timeout=inf`` and ``timeout=nan`` are passed to the socket, which
    raises ``OverflowError`` or ``ValueError`` outside the retry handler
    instead of a ``ProviderError``. Construction rejects non-finite
    timeout and backoff. Zero backoff stays valid.
    """
    for cls in (OpenAICompatibleProvider, AnthropicProvider):
        with pytest.raises(ValueError, match="finite"):
            cls(api_key="k", model="m", timeout=float("inf"))
        with pytest.raises(ValueError, match="finite"):
            cls(api_key="k", model="m", timeout=float("nan"))
        with pytest.raises(ValueError, match="finite"):
            cls(api_key="k", model="m", backoff_base=float("inf"))
        with pytest.raises(ValueError, match="finite"):
            cls(api_key="k", model="m", backoff_base=float("nan"))
        provider = cls(api_key="k", model="m", timeout=1, backoff_base=0)
        assert provider.timeout == 1
        assert provider.backoff_base == 0
