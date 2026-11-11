"""Regression tests for bugs found in the independent v0.2.0 review."""

from promptregress.calibration import parse_preference, position_bias_audit, verbosity_bias
from promptregress.models import AssertionSpec, Case
from promptregress.scorers import score_case


def test_parse_preference_accepts_exact_output_with_surrounding_whitespace() -> None:
    """Exact-text preferences must match even when an output has padding whitespace.

    ``_side`` stripped the judge reply before comparing it to ``first`` /
    ``second``, so returning the exact model output failed whenever that
    output had leading or trailing whitespace.
    """
    first = "hello\n"
    second = "world\n"
    assert parse_preference(first, second, first) == "first"
    assert parse_preference(first, second, second) == "second"
    assert parse_preference(" answer ", "other", " answer ") == "first"
    assert parse_preference("other", " answer ", " answer ") == "second"
    # A judge that strips the reply still matches after both sides are stripped.
    assert parse_preference(first, second, "hello") == "first"
    assert parse_preference(first, second, "world") == "second"


def test_position_and_verbosity_audits_tolerate_padded_outputs() -> None:
    pairs = [("alpha\n", "beta\n"), ("gamma ", "delta")]

    def prefer_content(first: str, second: str) -> str:
        # Return the exact preferred output text (common judge callback shape).
        if "alpha" in first or "gamma" in first:
            return first
        return second

    audit = position_bias_audit(pairs, prefer_content)
    assert audit.n_pairs == 2
    assert audit.n_flips == 0
    assert audit.flagged is False

    bias = verbosity_bias([("long text\n", "x")], lambda a, b: a)
    assert bias.preferences == [1.0]


def test_json_valid_rejects_nan_and_infinity() -> None:
    """RFC 8259 JSON has no NaN/Infinity; Python's json.loads accepts them by default."""
    case = Case(
        id="j",
        input="q",
        reference=None,
        assertions=[AssertionSpec(kind="json_valid")],
    )
    for output in ("NaN", "Infinity", "-Infinity", '{"a": NaN}', "[1, Infinity]"):
        scores, score, passed = score_case(case, output)
        assert scores["json_valid"] == 0.0
        assert score == 0.0
        assert passed is False
    # Still accept ordinary JSON, including null.
    assert score_case(case, "null")[1] == 1.0
    assert score_case(case, "1.5")[1] == 1.0
    assert score_case(case, '{"a": 1}')[1] == 1.0


def test_json_schema_rejects_nan_and_infinity_numbers() -> None:
    case = Case(
        id="j",
        input="q",
        reference=None,
        assertions=[AssertionSpec(kind="json_schema", value={"type": "number"})],
    )
    for output in ("NaN", "Infinity", "-Infinity"):
        assert score_case(case, output)[1] == 0.0
    nested = Case(
        id="j2",
        input="q",
        reference=None,
        assertions=[
            AssertionSpec(
                kind="json_schema",
                value={"type": "object", "properties": {"a": {"type": "number"}}},
            )
        ],
    )
    assert score_case(nested, '{"a": NaN}')[1] == 0.0
    assert score_case(case, "1.5")[1] == 1.0
