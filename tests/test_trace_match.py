"""Hand-computed trace_match scores for agent tool traces."""

import json
from pathlib import Path

import pytest

from promptregress.loading import load_suite
from promptregress.models import AssertionSpec, Case
from promptregress.scorers import EXACT_KINDS, SCORERS, ScoreError, ScoreOutcome, score_case

SEARCH = {"tool": "search_kb", "args": {"query": "refund policy"}}
ORDER = {"tool": "get_order", "args": {"order_id": "1001"}}
REFUND = {"tool": "issue_refund", "args": {"order_id": "1001", "amount": "20.00"}}
EXPECTED = [SEARCH, ORDER, REFUND]


def _score(expected: object, output: str) -> ScoreOutcome:
    case = Case(
        id="trace",
        input="do the task",
        reference=None,
        assertions=[AssertionSpec(kind="trace_match", value=expected)],
    )
    return score_case(case, output)


def test_trace_match_is_graded() -> None:
    assert "trace_match" in SCORERS
    assert "trace_match" not in EXACT_KINDS


def test_trace_match_perfect() -> None:
    output = json.dumps(EXPECTED)
    assert _score(EXPECTED, output) == ({"trace_match": 1.0}, 1.0, True)
    assert _score(EXPECTED, f"\n  {output}\n")[1] == 1.0


def test_trace_match_reordered() -> None:
    """ORDER, SEARCH, REFUND against SEARCH, ORDER, REFUND scores 2/3.

    SEARCH then REFUND stays in order, and so does ORDER then REFUND.
    All three cannot be paired: ORDER occurs before SEARCH in the output.
    """
    output = json.dumps([ORDER, SEARCH, REFUND])
    assert _score(EXPECTED, output) == ({"trace_match": 2 / 3}, 2 / 3, True)


def test_trace_match_missing_step() -> None:
    """SEARCH, REFUND against SEARCH, ORDER, REFUND keeps the two that remain."""
    output = json.dumps([SEARCH, REFUND])
    assert _score(EXPECTED, output) == ({"trace_match": 2 / 3}, 2 / 3, True)


def test_trace_match_wrong_args() -> None:
    """A tool hit whose arg text is not a substring does not match that step."""
    wrong_query = {"tool": "search_kb", "args": {"query": "shipping"}}
    partial = json.dumps([wrong_query, ORDER, REFUND])
    assert _score(EXPECTED, partial) == ({"trace_match": 2 / 3}, 2 / 3, True)
    only_wrong = json.dumps([wrong_query])
    assert _score(EXPECTED, only_wrong) == ({"trace_match": 0.0}, 0.0, False)


def test_trace_match_extra_steps() -> None:
    """Inserted steps are ignored. Every expected step still matches."""
    noise = {"tool": "lookup_customer", "args": {"id": "9"}}
    output = json.dumps([noise, SEARCH, noise, ORDER, REFUND, noise])
    assert _score(EXPECTED, output) == ({"trace_match": 1.0}, 1.0, True)


def test_trace_match_does_not_reuse_actual_steps() -> None:
    """One actual step can satisfy only one expected step."""
    one = json.dumps([SEARCH])
    scored = _score([SEARCH, SEARCH], one)
    assert scored == ({"trace_match": 0.5}, 0.5, True)
    assert _score([SEARCH, SEARCH], json.dumps([SEARCH, SEARCH]))[1] == 1.0


def test_trace_match_longer_subsequence_not_greedy() -> None:
    """A loose early match must not block a longer later pairing.

    Expected queries ``a``, ``ab``, ``x`` against actual ``ab``, ``x``.
    Pairing ``a`` with ``ab`` leaves only ``x``, which matches the third
    step but not the second (1/3). Skipping ``a`` pairs ``ab`` and ``x``
    (2/3). The score is 2/3.
    """
    expected = [
        {"tool": "search", "args": {"q": "a"}},
        {"tool": "search", "args": {"q": "ab"}},
        {"tool": "search", "args": {"q": "x"}},
    ]
    actual = [
        {"tool": "search", "args": {"q": "ab"}},
        {"tool": "search", "args": {"q": "x"}},
    ]
    assert _score(expected, json.dumps(actual)) == ({"trace_match": 2 / 3}, 2 / 3, True)


def test_trace_match_arg_substring_is_case_sensitive() -> None:
    richer = json.dumps(
        [
            {
                "tool": "search_kb",
                "args": {"query": "refund policy", "note": "vip", "extra": "1"},
            }
        ]
    )
    # "" matches any present value. Extra actual keys are ignored.
    expected = [{"tool": "search_kb", "args": {"query": "refund", "note": ""}}]
    assert _score(expected, richer)[1] == 1.0
    assert _score([{"tool": "search_kb", "args": {"query": "Refund"}}], richer)[1] == 0.0
    assert _score([{"tool": "Search_kb", "args": {"query": "refund"}}], richer)[1] == 0.0
    missing_key = [{"tool": "search_kb", "args": {"query": "refund", "ticket": "1"}}]
    assert _score(missing_key, richer)[1] == 0.0

    numeric = json.dumps([{"tool": "get_order", "args": {"order_id": 1001}}])
    assert _score([{"tool": "get_order", "args": {"order_id": "1001"}}], numeric)[1] == 1.0
    flagged = json.dumps([{"tool": "notify", "args": {"ok": True}}])
    assert _score([{"tool": "notify", "args": {"ok": True}}], flagged)[1] == 1.0
    assert _score([{"tool": "notify", "args": {"ok": "True"}}], flagged)[1] == 0.0


@pytest.mark.parametrize(
    "output",
    [
        "",
        "not json",
        "{",
        "[",
        "[{]",
        "null",
        "42",
        '"[]"',
        '{"tool": "search_kb", "args": {"query": "refund policy"}}',
        "NaN",
        "Infinity",
        "```json\n[]\n```",
        json.dumps(EXPECTED) + " trailing",
    ],
)
def test_trace_match_malformed_json(output: str) -> None:
    assert _score(EXPECTED, output) == ({"trace_match": 0.0}, 0.0, False)


def test_trace_match_non_step_elements_do_not_match() -> None:
    output = json.dumps(
        [1, "search_kb", {"tool": 3}, {"args": {}}, {"tool": "search_kb", "args": []}]
    )
    assert _score(EXPECTED, output) == ({"trace_match": 0.0}, 0.0, False)
    assert _score(EXPECTED, "[]") == ({"trace_match": 0.0}, 0.0, False)


def test_trace_match_empty_expected_scores_one() -> None:
    """An empty expected trace is a vacuous match and does not parse the output."""
    assert _score([], "not json") == ({"trace_match": 1.0}, 1.0, True)
    assert _score([], "[]")[1] == 1.0


@pytest.mark.parametrize(
    "value",
    [
        None,
        {"tool": "search_kb"},
        "[]",
        [{"args": {}}],
        [{"tool": 1, "args": {}}],
        [{"tool": "search_kb", "args": []}],
        [{"tool": "search_kb", "args": {1: "a"}}],
    ],
)
def test_trace_match_bad_expectation_raises(value: object) -> None:
    with pytest.raises(ScoreError, match="trace_match"):
        _score(value, "[]")


def test_trace_match_loads_structured_assertion_value(tmp_path: Path) -> None:
    """Suite JSON already carries a list on assertion ``value``."""
    path = tmp_path / "traces.json"
    path.write_text(
        json.dumps(
            {
                "name": "mini",
                "cases": [
                    {
                        "id": "refund",
                        "input": "refund order 1001",
                        "reference": json.dumps(EXPECTED),
                        "assertions": [{"kind": "trace_match", "value": EXPECTED}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    suite = load_suite(path)
    assertion = suite.cases[0].assertions[0]
    assert assertion.kind == "trace_match"
    assert assertion.value == EXPECTED
    assert assertion.params == {}
    reference = suite.cases[0].reference or ""
    assert score_case(suite.cases[0], reference) == ({"trace_match": 1.0}, 1.0, True)
