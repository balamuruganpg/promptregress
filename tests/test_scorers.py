"""Deterministic scorers, the pass rule, the judge, and the plugin hook."""

import importlib.metadata

import pytest

from promptregress.models import AssertionSpec, Case
from promptregress.providers import MockProvider
from promptregress.scorers import (
    EXACT_KINDS,
    SCORERS,
    ScoreError,
    ScoringContext,
    load_scorer_plugins,
    register_scorer,
    score_case,
)


def _case(
    kind: str,
    value: object = None,
    params: dict | None = None,
    reference: str | None = "ref",
    assertions: list[AssertionSpec] | None = None,
) -> Case:
    if assertions is None:
        assertions = [AssertionSpec(kind=kind, value=value, params=params or {})]
    return Case(id="c", input="question", reference=reference, assertions=assertions)


def test_exact_kinds_are_the_binary_scorers() -> None:
    assert EXACT_KINDS == {"exact_match", "contains", "regex", "json_valid", "json_schema"}


def test_pass_rule_is_documented() -> None:
    doc = score_case.__doc__ or ""
    assert "0.999" in doc
    assert "0.5" in doc


def test_exact_match_normalizes() -> None:
    case = _case("exact_match", value="hello world")
    scores, score, passed = score_case(case, "  Hello   WORLD \n")
    assert scores == {"exact_match": 1.0}
    assert score == 1.0
    assert passed is True


def test_exact_match_case_sensitive() -> None:
    case = _case("exact_match", value="Hello", params={"case_sensitive": True})
    _, score, passed = score_case(case, "hello")
    assert score == 0.0
    assert passed is False
    _, score_same, passed_same = score_case(
        _case("exact_match", value="Hello", params={"case_sensitive": True}),
        "Hello",
    )
    assert score_same == 1.0
    assert passed_same is True


def test_exact_match_uses_reference_when_value_missing() -> None:
    case = _case("exact_match", reference="hello")
    _, score, passed = score_case(case, "Hello")
    assert score == 1.0
    assert passed is True


def test_exact_match_missing_target() -> None:
    case = _case("exact_match", reference=None)
    _, score, passed = score_case(case, "anything")
    assert score == 0.0
    assert passed is False


def test_contains_normalized_substring() -> None:
    case = _case("contains", value="hello world")
    _, score, passed = score_case(case, "say  Hello   WORLD  please")
    assert score == 1.0
    assert passed is True
    _, miss, miss_passed = score_case(case, "goodbye")
    assert miss == 0.0
    assert miss_passed is False


def test_contains_case_sensitive_and_empty_needle() -> None:
    sensitive = _case("contains", value="Hello", params={"case_sensitive": True})
    _, score, _ = score_case(sensitive, "hello")
    assert score == 0.0
    empty = _case("contains", value="")
    _, empty_score, _ = score_case(empty, "abc")
    assert empty_score == 1.0


def test_regex_pass_and_fail() -> None:
    case = _case("regex", value=r"a+")
    assert score_case(case, "baaac")[1] == 1.0
    assert score_case(case, "bbb")[1] == 0.0
    assert score_case(case, "bbb")[2] is False


def test_regex_ignore_case_and_invalid_pattern() -> None:
    case = _case("regex", value="cat", params={"ignore_case": True})
    assert score_case(case, "the CAT sat")[1] == 1.0
    bad = _case("regex", value="[")
    with pytest.raises(ScoreError, match="Invalid regex"):
        score_case(bad, "abc")


def test_json_valid() -> None:
    case = _case("json_valid")
    for output in (' {"a": 1} ', "null", "true", '"s"', "1", "[1, 2]"):
        assert score_case(case, output)[1] == 1.0
    assert score_case(case, "")[1] == 0.0
    assert score_case(case, "[] trailing")[1] == 0.0
    assert score_case(case, "{")[1] == 0.0


def test_json_schema_subset() -> None:
    schema = {
        "type": "object",
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "integer", "enum": [1, 2, 3]},
            }
        },
    }
    case = _case("json_schema", value=schema)
    assert score_case(case, '{"items": [1, 2]}')[2] is True
    assert score_case(case, '{"items": [1, 4]}')[1] == 0.0
    assert score_case(case, '{"items": [1, true]}')[1] == 0.0
    assert score_case(case, "{}")[1] == 0.0
    assert score_case(case, '{"items": "no"}')[1] == 0.0
    assert score_case(case, "not json")[1] == 0.0


def test_json_schema_type_list_and_ignored_keywords() -> None:
    union = _case("json_schema", value={"type": ["string", "null"]})
    assert score_case(union, '"hi"')[1] == 1.0
    assert score_case(union, "null")[1] == 1.0
    assert score_case(union, "1")[1] == 0.0
    number = _case("json_schema", value={"type": "number"})
    assert score_case(number, "true")[1] == 0.0
    assert score_case(number, "1")[1] == 1.0
    assert score_case(number, "1.5")[1] == 1.0
    integer = _case("json_schema", value={"type": "integer"})
    assert score_case(integer, "1.0")[1] == 0.0
    loose = _case("json_schema", value={"type": "string", "minLength": 5})
    assert score_case(loose, '"a"')[1] == 1.0
    extra = _case(
        "json_schema",
        value={"type": "object", "properties": {"a": {"type": "integer"}}},
    )
    assert score_case(extra, '{"a": 1, "b": 2}')[1] == 1.0


def test_json_schema_requires_object_schema() -> None:
    case = _case("json_schema", value=["not", "a", "schema"])
    with pytest.raises(ScoreError, match="schema object"):
        score_case(case, "{}")


def test_numeric_tolerance_edges() -> None:
    exact = _case("numeric_tolerance", value=5, params={"tol": 0})
    assert score_case(exact, "see 5 then 9")[1] == 1.0
    assert score_case(exact, "no digits")[1] == 0.0
    assert score_case(exact, "value is -3.5")[0]["numeric_tolerance"] == 0.0
    negative = _case("numeric_tolerance", value=-3.5, params={"tol": 0})
    assert score_case(negative, "value is -3.5 today")[1] == 1.0
    sci = _case("numeric_tolerance", value=0.001, params={"tol": 1e-12})
    assert score_case(sci, "about 1e-3 units")[1] == 1.0
    within = _case("numeric_tolerance", value=10, params={"tol": 1})
    assert score_case(within, "10.5")[1] == 1.0
    partial = score_case(within, "11.5")[1]
    assert partial == pytest.approx(0.5)
    assert score_case(within, "12")[1] == 0.0
    # tol 0 decays across a window of 1.
    assert score_case(exact, "5.2")[1] == pytest.approx(0.8)


def test_levenshtein() -> None:
    same = _case("levenshtein", value="abc")
    assert score_case(same, "abc")[1] == 1.0
    folded = _case("levenshtein", value="abc")
    assert score_case(folded, "ABC")[1] == 1.0
    sensitive = _case("levenshtein", value="abc", params={"case_sensitive": True})
    assert score_case(sensitive, "ABC")[1] < 1.0
    empty = _case("levenshtein", value="")
    assert score_case(empty, "   ")[1] == 1.0
    assert score_case(_case("levenshtein", value="abc"), "")[1] == 0.0
    kitten = _case("levenshtein", value="sitting")
    assert score_case(kitten, "kitten")[1] == pytest.approx(1.0 - 3 / 7)


def test_levenshtein_graded_threshold() -> None:
    # "abcd" vs "ab": distance 2, max len 4, similarity 0.5. Graded floor is 0.5.
    half = _case("levenshtein", value="ab", reference=None)
    _, score, passed = score_case(half, "abcd")
    assert score == pytest.approx(0.5)
    assert passed is True
    below = _case("levenshtein", value="ab", reference=None)
    _, low, low_passed = score_case(below, "abcde")
    assert low == pytest.approx(0.4)
    assert low_passed is False


def test_tfidf_cosine() -> None:
    same = _case("tfidf_cosine", value="red cat sat")
    assert score_case(same, "red cat sat")[1] == pytest.approx(1.0)
    empty = _case("tfidf_cosine", value="", reference=None)
    assert score_case(empty, "   ")[1] == 1.0
    one_empty = _case("tfidf_cosine", value="cat")
    assert score_case(one_empty, "")[1] == 0.0
    disjoint = _case("tfidf_cosine", value="dog")
    assert score_case(disjoint, "cat")[1] == 0.0
    partial = score_case(_case("tfidf_cosine", value="red dog"), "red cat")[1]
    assert 0.0 < partial < 1.0
    chars = _case("tfidf_cosine", value="abcd", params={"mode": "char", "ngram": 2})
    assert score_case(chars, "abcd")[1] == pytest.approx(1.0)
    assert (
        score_case(_case("tfidf_cosine", value="yz", params={"mode": "char", "ngram": 2}), "ab")[1]
        == 0.0
    )


def test_tfidf_rejects_bad_params() -> None:
    with pytest.raises(ScoreError, match="mode"):
        score_case(_case("tfidf_cosine", value="a", params={"mode": "byte"}), "a")
    with pytest.raises(ScoreError, match="ngram"):
        score_case(_case("tfidf_cosine", value="a", params={"ngram": 0}), "a")


def test_unknown_scorer() -> None:
    case = _case("not_a_scorer", value="be helpful")
    with pytest.raises(ScoreError, match="not_a_scorer"):
        score_case(case, "hello")


def test_no_assertions_does_not_pass() -> None:
    case = Case(id="c", input="q", reference=None, assertions=[])
    assert score_case(case, "hello") == ({}, 0.0, False)


def test_duplicate_kind_is_averaged_and_each_exact_must_pass() -> None:
    case = _case(
        "contains",
        assertions=[
            AssertionSpec("contains", value="alpha"),
            AssertionSpec("contains", value="missing"),
        ],
    )
    scores, score, passed = score_case(case, "alpha only")
    assert scores["contains"] == pytest.approx(0.5)
    assert score == pytest.approx(0.5)
    assert passed is False


def test_weighted_mean_and_mixed_pass_rule() -> None:
    both_exact_fail_weight = Case(
        id="c",
        input="q",
        reference=None,
        assertions=[
            AssertionSpec("contains", value="yes", weight=1),
            AssertionSpec("contains", value="no", weight=3),
        ],
    )
    _, weighted, passed = score_case(both_exact_fail_weight, "yes")
    assert weighted == pytest.approx(0.25)
    assert passed is False

    exact_and_low_graded = Case(
        id="c",
        input="q",
        reference=None,
        assertions=[
            AssertionSpec("exact_match", value="ok", weight=1),
            AssertionSpec("levenshtein", value="zzzzzzzz", weight=3),
        ],
    )
    scores, mixed, mixed_passed = score_case(exact_and_low_graded, "ok")
    assert scores["exact_match"] == 1.0
    assert scores["levenshtein"] == 0.0
    assert mixed == pytest.approx(0.25)
    assert mixed_passed is False

    exact_and_ok_graded = Case(
        id="c",
        input="q",
        reference=None,
        assertions=[
            AssertionSpec("exact_match", value="abcd"),
            AssertionSpec("levenshtein", value="ab"),
        ],
    )
    _, mean, ok = score_case(exact_and_ok_graded, "abcd")
    # exact 1.0 and levenshtein 0.5 -> weighted 0.75, exact passes, graded floor met.
    assert mean == pytest.approx(0.75)
    assert ok is True


def test_register_scorer_plugin_is_graded() -> None:
    saved_scorers = dict(SCORERS)
    saved_exact = set(EXACT_KINDS)
    try:

        @register_scorer("always_one")
        def always_one(output: str, case: Case, assertion: AssertionSpec) -> float:
            return 1.0

        assert SCORERS["always_one"] is always_one
        case = _case("always_one")
        _, score, passed = score_case(case, "ignored")
        assert score == 1.0
        assert passed is True

        @register_scorer("always_low")
        def always_low(output: str, case: Case, assertion: AssertionSpec) -> float:
            return 0.2

        _, low, low_passed = score_case(_case("always_low"), "x")
        assert low == pytest.approx(0.2)
        assert low_passed is False

        @register_scorer("clamped")
        def clamped(output: str, case: Case, assertion: AssertionSpec) -> float:
            return 1.4

        assert score_case(_case("clamped"), "x")[1] == 1.0

        @register_scorer
        def bare_name(output: str, case: Case, assertion: AssertionSpec) -> float:
            return 1.0

        assert "bare_name" in SCORERS
    finally:
        SCORERS.clear()
        SCORERS.update(saved_scorers)
        EXACT_KINDS.clear()
        EXACT_KINDS.update(saved_exact)


def _judge_case(
    rubric: str = "Reward a complete answer.",
    reference: str | None = "the reference",
    question: str = "question",
    case_id: str = "c",
) -> Case:
    return Case(
        id=case_id,
        input=question,
        reference=reference,
        assertions=[AssertionSpec("llm_judge", params={"rubric": rubric})],
    )


def test_llm_judge_is_graded() -> None:
    assert "llm_judge" in SCORERS
    assert "llm_judge" not in EXACT_KINDS


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("0.75", 0.75),
        ("  1\n", 1.0),
        ("0", 0.0),
        ("0.5", 0.5),
        ('{"score": 0.25}', 0.25),
        ('{"reason": "ok", "score": 1}', 1.0),
        ('{"score": "0.5"}', 0.5),
        ("Score: 0.9", 0.9),
        ("score:0.4", 0.4),
        ("Verdict\nScore: 0.8\nbecause it helped", 0.8),
        ('```json\n{"score": 0.6}\n```', 0.6),
        ('prefix {"score": 0.7} suffix', 0.7),
    ],
)
def test_llm_judge_parses_mock_replies(reply: str, expected: float) -> None:
    judge = MockProvider(default=reply)
    ctx = ScoringContext(judge=judge)
    scored = score_case(_judge_case(), "the answer", context=ctx)
    assert scored[1] == pytest.approx(expected)
    assert scored.notes == ()
    assert scored.details == ()
    assert ctx.notes == []
    assert judge.call_count == 1
    assert scored[2] is (expected >= 0.5)


@pytest.mark.parametrize(
    "reply",
    ["", "nope", "Score: 1.2", '{"score": 2}', '{"score": -0.1}', '{"score": true}', "2", "Score:"],
)
def test_llm_judge_bad_reply_scores_zero(reply: str) -> None:
    judge = MockProvider(default=reply)
    ctx = ScoringContext(judge=judge)
    scored = score_case(_judge_case(), "output", context=ctx)
    assert scored == ({"llm_judge": 0.0}, 0.0, False)
    assert scored.notes
    assert "llm_judge (c)" in scored.notes[0]
    assert scored.details[0]["kind"] == "llm_judge"
    assert scored.details[0]["case_id"] == "c"
    assert scored.details[0]["error"] == scored.notes[0]
    assert scored.details[0]["raw"] == reply
    assert ctx.notes == list(scored.notes)
    assert ctx.details == list(scored.details)
    assert judge.call_count == 1


def test_llm_judge_provider_failure_is_a_zero_with_a_note() -> None:
    class Down:
        name = "down"
        model = "down"

        def complete(self, prompt: str) -> str:
            raise RuntimeError("judge down")

    ctx = ScoringContext(judge=Down())
    scored = score_case(_judge_case(), "output", context=ctx)
    assert scored == ({"llm_judge": 0.0}, 0.0, False)
    assert "RuntimeError" in scored.notes[0]
    assert "judge down" in scored.notes[0]
    assert scored.details[0]["raw"] is None
    assert ctx.notes == list(scored.notes)


def test_llm_judge_without_provider_or_rubric_does_not_raise() -> None:
    missing_judge = score_case(_judge_case(), "hello")
    assert missing_judge == ({"llm_judge": 0.0}, 0.0, False)
    assert "no judge provider" in missing_judge.notes[0]
    assert missing_judge.details[0]["raw"] is None

    judge = MockProvider(default="1")
    ctx = ScoringContext(judge=judge)
    missing_rubric = score_case(_case("llm_judge"), "hello", context=ctx)
    assert missing_rubric == ({"llm_judge": 0.0}, 0.0, False)
    assert "rubric" in missing_rubric.notes[0]
    assert judge.call_count == 0

    blank = score_case(_judge_case(rubric="   "), "hello", context=ctx)
    assert blank[1] == 0.0
    assert "rubric" in blank.notes[0]
    assert judge.call_count == 0


def test_llm_judge_prompt_includes_rubric_input_reference_and_output() -> None:
    judge = MockProvider(rules=[("RUBRIC-UNIQUE-42", "1")], default="0")
    scored = score_case(
        _judge_case(rubric="RUBRIC-UNIQUE-42", question="Q-UNIQUE", reference="REF-UNIQUE"),
        "OUT-UNIQUE",
        context=ScoringContext(judge=judge),
    )
    assert scored[1] == 1.0

    judge = MockProvider(rules=[("\nInput: Q-UNIQUE-7\n", "1")], default="0")
    scored = score_case(
        _judge_case(question="Q-UNIQUE-7"),
        "answer",
        context=ScoringContext(judge=judge),
    )
    assert scored[1] == 1.0

    judge = MockProvider(rules=[("\nReference: REF-UNIQUE-7\n", "1")], default="0")
    scored = score_case(
        _judge_case(reference="REF-UNIQUE-7"),
        "answer",
        context=ScoringContext(judge=judge),
    )
    assert scored[1] == 1.0

    judge = MockProvider(rules=[("\nOutput: OUT-UNIQUE-7\n", "1")], default="0")
    scored = score_case(_judge_case(), "OUT-UNIQUE-7", context=ScoringContext(judge=judge))
    assert scored[1] == 1.0

    judge = MockProvider(rules=[("\nReference: \n", "1")], default="0")
    scored = score_case(
        _judge_case(reference=None),
        "answer",
        context=ScoringContext(judge=judge),
    )
    assert scored[1] == 1.0


def test_llm_judge_notes_from_one_call_do_not_include_older_notes() -> None:
    judge = MockProvider(default="nope")
    ctx = ScoringContext(judge=judge)
    first = score_case(_judge_case(case_id="one"), "a", context=ctx)
    second = score_case(_judge_case(case_id="two"), "b", context=ctx)
    assert len(first.notes) == 1
    assert len(second.notes) == 1
    assert "one" in first.notes[0]
    assert "two" in second.notes[0]
    assert len(ctx.notes) == 2


def test_score_case_still_equals_a_three_tuple_when_the_judge_fails() -> None:
    scored = score_case(_judge_case(), "hello")
    assert scored == ({"llm_judge": 0.0}, 0.0, False)
    scores, score, passed = scored
    assert scores == {"llm_judge": 0.0}
    assert score == 0.0
    assert passed is False


class _EntryPoint:
    def __init__(self, name: str, loaded: object) -> None:
        self.name = name
        self._loaded = loaded

    def load(self) -> object:
        return self._loaded


def _patch_entry_points(
    monkeypatch: pytest.MonkeyPatch,
    entries: list[_EntryPoint],
) -> None:
    def fake_entry_points(**kwargs: object) -> list[_EntryPoint]:
        if kwargs.get("group") == "promptregress.scorers":
            return list(entries)
        return []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)


def test_missing_scorer_entry_points_are_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch, [])
    before = dict(SCORERS)
    load_scorer_plugins()
    assert SCORERS == before
    assert "exact_match" in SCORERS


def test_entry_point_scorer_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    def plugin(output: str, case: Case, assertion: AssertionSpec) -> float:
        del output, case, assertion
        return 0.8

    saved_scorers = dict(SCORERS)
    saved_exact = set(EXACT_KINDS)
    try:
        _patch_entry_points(monkeypatch, [_EntryPoint("plugin_score", plugin)])
        load_scorer_plugins()
        load_scorer_plugins()
        assert SCORERS["plugin_score"] is plugin
        assert "plugin_score" not in EXACT_KINDS
        scored = score_case(_case("plugin_score"), "x")
        assert scored[1] == pytest.approx(0.8)
        assert scored[2] is True
    finally:
        SCORERS.clear()
        SCORERS.update(saved_scorers)
        EXACT_KINDS.clear()
        EXACT_KINDS.update(saved_exact)


def test_entry_point_does_not_replace_a_builtin(monkeypatch: pytest.MonkeyPatch) -> None:
    def evil(output: str, case: Case, assertion: AssertionSpec) -> float:
        del output, case, assertion
        return 0.0

    saved_scorers = dict(SCORERS)
    saved_exact = set(EXACT_KINDS)
    try:
        _patch_entry_points(monkeypatch, [_EntryPoint("exact_match", evil)])
        load_scorer_plugins()
        assert SCORERS["exact_match"] is not evil
        assert score_case(_case("exact_match", value="a"), "a")[1] == 1.0
    finally:
        SCORERS.clear()
        SCORERS.update(saved_scorers)
        EXACT_KINDS.clear()
        EXACT_KINDS.update(saved_exact)


def test_entry_point_keeps_exact_flag_from_self_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_exact() -> object:
        @register_scorer("exact_plugin", exact=True)
        def exact_plugin(output: str, case: Case, assertion: AssertionSpec) -> float:
            del output, case, assertion
            return 0.6

        return exact_plugin

    saved_scorers = dict(SCORERS)
    saved_exact = set(EXACT_KINDS)
    try:
        _patch_entry_points(monkeypatch, [_EntryPoint("exact_plugin", load_exact())])
        # load_exact already registered it; the loader must not downgrade exact=True.
        load_scorer_plugins()
        assert "exact_plugin" in EXACT_KINDS
        _, score, passed = score_case(_case("exact_plugin"), "x")
        assert score == pytest.approx(0.6)
        assert passed is False
    finally:
        SCORERS.clear()
        SCORERS.update(saved_scorers)
        EXACT_KINDS.clear()
        EXACT_KINDS.update(saved_exact)


def test_non_callable_entry_point_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    saved_scorers = dict(SCORERS)
    saved_exact = set(EXACT_KINDS)
    try:
        _patch_entry_points(monkeypatch, [_EntryPoint("bad_plugin", 123)])
        with pytest.raises(ScoreError, match="bad_plugin"):
            load_scorer_plugins()
        assert "bad_plugin" not in SCORERS
    finally:
        SCORERS.clear()
        SCORERS.update(saved_scorers)
        EXACT_KINDS.clear()
        EXACT_KINDS.update(saved_exact)
