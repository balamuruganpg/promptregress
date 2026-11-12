"""Scorers. Each returns a float in ``[0.0, 1.0]``.

Pass rule used by :func:`score_case`:

- Exact-type scorers (registered with ``exact=True``: ``exact_match``,
  ``contains``, ``regex``, ``json_valid``, ``json_schema``) must each score
  at least ``0.999``.
- If any graded scorer is present (``numeric_tolerance``, ``levenshtein``,
  ``tfidf_cosine``, ``llm_judge``, ``trace_match``, and plugins registered
  without ``exact=True``), the weighted mean must also be at least ``0.5``.
- With only exact-type scorers, the weighted-mean check is not applied.

A case with no assertions scores ``0.0`` and does not pass.

Suite ``scoring.weights`` scales each assertion by its scorer kind. Kinds
left out of the map use weight 1. Omitting the map, or setting every weight
equal, leaves the case score unchanged. The per-kind subscores in the
returned ``scores`` dict stay the unweighted mean of that kind.

``llm_judge`` is graded. It calls a provider carried on a
:class:`ScoringContext`. A missing judge, a provider error, or an
unparseable reply scores ``0.0`` and records the failure on the result.
It does not raise. ``params.judges`` is an ensemble: each entry has its
own provider and rubric, and the assertion score is the unweighted mean.
Omitting ``judges`` keeps the single context judge, with the same score
and the same failure records.

Importing this module also loads callables from the
``promptregress.scorers`` entry-point group. An empty group is normal.
"""

import contextvars
import importlib.metadata
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast, overload

from promptregress.models import AssertionSpec, Case
from promptregress.providers import (
    EchoProvider,
    MockProvider,
    OpenAICompatibleProvider,
    Provider,
    provider_model_name,
)

Scorer = Callable[[str, Case, AssertionSpec], float]

SCORERS: dict[str, Scorer] = {}
EXACT_KINDS: set[str] = set()

EXACT_PASS_FLOOR = 0.999
GRADED_MEAN_FLOOR = 0.5

_WHITESPACE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_WORD_RE = re.compile(r"\w+")
_PLAIN_NUMBER = re.compile(r"[-+]?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_SCORE_LABEL = re.compile(rf"(?i)\bscore\s*:\s*({_PLAIN_NUMBER.pattern})")
_NOTE_LIMIT = 180


class ScoreError(ValueError):
    """An assertion or scorer result cannot be scored."""


class _JudgeFailure(Exception):
    """Internal: ``llm_judge`` could not produce a score."""

    def __init__(self, message: str, raw: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.raw = raw


@dataclass
class ScoringContext:
    """Judge provider and failure records for one scoring call.

    Pass this to :func:`score_case` or to ``run_suite``. ``llm_judge`` reads
    ``judge``. On a failure it appends a human-readable string to ``notes``
    and a dict to ``details`` (keys ``kind``, ``case_id``, ``error``, and
    ``raw``). The assertion still scores 0.0.

    ``run_suite`` reads ``judge`` and stores joined notes on
    ``CaseResult.error``. It does not append to the context you pass, so
    that object can be shared across worker threads when the judge provider
    itself is safe to call concurrently. Do not share one context across
    concurrent :func:`score_case` calls; those calls append to ``notes``.
    """

    judge: Provider | None = None
    notes: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)


class ScoreOutcome(tuple[dict[str, float], float, bool]):
    """``(scores, weighted_mean, passed)`` plus judge notes.

    Unpacks and compares like a 3-tuple, so existing callers keep working.
    ``notes`` are human-readable failures. ``details`` are the structured
    records also appended to :class:`ScoringContext`. Both are empty when
    scoring did not record a failure or an ensemble panel.
    """

    notes: tuple[str, ...]
    details: tuple[dict[str, Any], ...]

    def __new__(
        cls,
        scores: dict[str, float],
        score: float,
        passed: bool,
        notes: tuple[str, ...] = (),
        details: tuple[dict[str, Any], ...] = (),
    ) -> "ScoreOutcome":
        outcome = tuple.__new__(cls, (scores, score, passed))
        outcome.notes = notes
        outcome.details = details
        return outcome

    def __repr__(self) -> str:
        return (
            f"ScoreOutcome(scores={self[0]!r}, score={self[1]!r}, passed={self[2]!r}, "
            f"notes={self.notes!r}, details={self.details!r})"
        )


_SCORING_CONTEXT: contextvars.ContextVar[ScoringContext | None] = contextvars.ContextVar(
    "promptregress_scoring_context",
    default=None,
)


def _store(name: str, fn: Scorer, exact: bool) -> Scorer:
    SCORERS[name] = fn
    if exact:
        EXACT_KINDS.add(name)
    else:
        EXACT_KINDS.discard(name)
    return fn


@overload
def register_scorer(name: Scorer, *, exact: bool = ...) -> Scorer: ...


@overload
def register_scorer(name: str, *, exact: bool = ...) -> Callable[[Scorer], Scorer]: ...


def register_scorer(
    name: str | Scorer,
    *,
    exact: bool = False,
) -> Scorer | Callable[[Scorer], Scorer]:
    """Register a scorer.

    ``@register_scorer("kind")`` stores the function under ``kind``.
    ``@register_scorer`` without arguments uses the function name.
    ``exact=True`` opts the kind into the exact-type pass rule. Plugins are
    graded unless they pass ``exact=True``.
    """
    if isinstance(name, str):

        def deco(fn: Scorer) -> Scorer:
            return _store(name, fn, exact)

        return deco
    return _store(name.__name__, name, exact)


def score_case(
    case: Case,
    output: str,
    context: ScoringContext | None = None,
    *,
    weights: Mapping[str, float] | None = None,
) -> ScoreOutcome:
    """Score ``output`` against every assertion on ``case``.

    Returns a :class:`ScoreOutcome` ``(scores, weighted_mean, passed)``.
    It unpacks and compares like that 3-tuple. ``notes`` and ``details``
    carry judge failures and are empty when every scorer succeeds, except
    that an ``llm_judge`` ensemble also records each judge's score.

    ``scores`` maps each assertion kind to the unweighted mean of that kind
    when a case repeats a kind. Those subscores do not depend on weights.
    ``weighted_mean`` uses every assertion. Its effective weight is
    ``assertion.weight * weights.get(kind, 1)``. ``weights`` is the suite
    ``scoring.weights`` map. ``None`` or an empty map is equal weights, which
    is the mean of the assertion weights alone. Keys for kinds this case does
    not use are ignored. A negative or non-finite weight raises
    ``ScoreError``. Scores from a scorer are clamped to ``[0.0, 1.0]``.

    Pass rule:

    - Every exact-type assertion must score at least 0.999.
    - If any graded scorer is present, the weighted mean must also be at
      least 0.5.
    - With only exact-type scorers, the weighted-mean check is not applied.

    A case with no assertions returns ``({}, 0.0, False)``.

    ``context`` supplies the judge provider for ``llm_judge``. When it is
    omitted, ``llm_judge`` scores 0.0 and records an error instead of
    raising. The same notes are appended to ``context`` when one is passed.
    Notes on the returned outcome are only those recorded by this call.
    """
    ctx = context if context is not None else ScoringContext()
    note_start = len(ctx.notes)
    detail_start = len(ctx.details)
    token = _SCORING_CONTEXT.set(ctx)
    try:
        if not case.assertions:
            scores: dict[str, float] = {}
            weighted = 0.0
            passed = False
        else:
            scores, weighted, passed = _score_assertions(case, output, weights)
        return ScoreOutcome(
            scores,
            weighted,
            passed,
            notes=tuple(ctx.notes[note_start:]),
            details=tuple(ctx.details[detail_start:]),
        )
    finally:
        _SCORING_CONTEXT.reset(token)


def _score_assertions(
    case: Case,
    output: str,
    weights: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], float, bool]:
    parts: list[tuple[str, float, float]] = []
    buckets: dict[str, list[float]] = {}
    for assertion in case.assertions:
        value = _score_one(case, output, assertion)
        # Suite weights scale a kind. Equal scales cancel, so the historical
        # mean of assertion weights is unchanged when every kind weighs the same.
        effective = assertion.weight * _scorer_weight(assertion.kind, weights)
        parts.append((assertion.kind, value, effective))
        buckets.setdefault(assertion.kind, []).append(value)
    scores = {kind: sum(values) / len(values) for kind, values in buckets.items()}
    weight_sum = sum(weight for _, _, weight in parts)
    if weight_sum <= 0:
        weighted = 0.0
    else:
        weighted = sum(value * weight for _, value, weight in parts) / weight_sum
    exact_ok = all(value >= EXACT_PASS_FLOOR for kind, value, _ in parts if kind in EXACT_KINDS)
    graded_present = any(kind not in EXACT_KINDS for kind, _, _ in parts)
    if graded_present:
        passed = exact_ok and weighted >= GRADED_MEAN_FLOOR
    else:
        passed = exact_ok
    return scores, weighted, passed


def _scorer_weight(kind: str, weights: Mapping[str, float] | None) -> float:
    """Weight for ``kind``. Missing maps and missing kinds use 1."""
    if not weights or kind not in weights:
        return 1.0
    raw = weights[kind]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ScoreError(f"Scorer weight for {kind!r} must be a non-negative finite number")
    try:
        weight = float(raw)
    except OverflowError as exc:
        raise ScoreError(f"Scorer weight for {kind!r} must be non-negative and finite") from exc
    if not math.isfinite(weight) or weight < 0:
        raise ScoreError(f"Scorer weight for {kind!r} must be non-negative and finite")
    return weight


def _score_one(case: Case, output: str, assertion: AssertionSpec) -> float:
    kind = assertion.kind
    scorer = SCORERS.get(kind)
    if scorer is None:
        known = ", ".join(sorted(SCORERS))
        raise ScoreError(f"Unknown scorer {kind!r}. Known scorers: {known}")
    try:
        weight_finite = math.isfinite(assertion.weight)
    except OverflowError:
        weight_finite = False
    if not weight_finite or assertion.weight < 0:
        raise ScoreError(f"Case {case.id} assertion weight must be non-negative and finite")
    raw = scorer(output, case, assertion)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ScoreError(f"Scorer {kind!r} returned a non-numeric score") from exc
    if not math.isfinite(value):
        raise ScoreError(f"Scorer {kind!r} returned a non-finite score")
    return min(1.0, max(0.0, value))


def _case_sensitive(params: dict[str, Any]) -> bool:
    raw = params.get("case_sensitive", False)
    if not isinstance(raw, bool):
        raise ScoreError("case_sensitive must be a boolean")
    return raw


def _flag(params: dict[str, Any], name: str) -> bool:
    raw = params.get(name, False)
    if not isinstance(raw, bool):
        raise ScoreError(f"{name} must be a boolean")
    return raw


def _normalize(text: str, *, case_sensitive: bool) -> str:
    collapsed = _WHITESPACE.sub(" ", text.strip())
    if case_sensitive:
        return collapsed
    return collapsed.casefold()


def _target(case: Case, assertion: AssertionSpec) -> str | None:
    """Assertion value, otherwise the case reference. ``None`` if both missing."""
    if assertion.value is not None:
        return str(assertion.value)
    return case.reference


@register_scorer("exact_match", exact=True)
def exact_match(output: str, case: Case, assertion: AssertionSpec) -> float:
    """1.0 when normalized output equals the value or the reference.

    Normalization strips ends, collapses internal whitespace, and casefolds.
    ``params.case_sensitive`` (default ``False``) skips casefolding.
    """
    target = _target(case, assertion)
    if target is None:
        return 0.0
    sensitive = _case_sensitive(assertion.params)
    left = _normalize(output, case_sensitive=sensitive)
    right = _normalize(target, case_sensitive=sensitive)
    return 1.0 if left == right else 0.0


@register_scorer("contains", exact=True)
def contains(output: str, case: Case, assertion: AssertionSpec) -> float:
    """1.0 when the normalized value or reference is a substring of the output.

    Uses the same normalization as ``exact_match``. An empty needle matches.
    A missing value and reference scores 0.0.
    """
    target = _target(case, assertion)
    if target is None:
        return 0.0
    sensitive = _case_sensitive(assertion.params)
    needle = _normalize(target, case_sensitive=sensitive)
    haystack = _normalize(output, case_sensitive=sensitive)
    return 1.0 if needle in haystack else 0.0


@register_scorer("regex", exact=True)
def regex(output: str, case: Case, assertion: AssertionSpec) -> float:
    """1.0 when ``re.search`` finds ``value`` in the raw output.

    The pattern is not normalized. ``params.ignore_case`` sets
    ``re.IGNORECASE``. An invalid pattern raises ``ScoreError``.
    """
    if not isinstance(assertion.value, str) or assertion.value == "":
        raise ScoreError("regex assertion requires a non-empty string pattern")
    flags = re.IGNORECASE if _flag(assertion.params, "ignore_case") else 0
    try:
        pattern = re.compile(assertion.value, flags)
    except re.error as exc:
        raise ScoreError(f"Invalid regex pattern: {exc}") from exc
    return 1.0 if pattern.search(output) is not None else 0.0


def _loads_rfc8259(text: str) -> Any:
    """Parse JSON, rejecting ``NaN`` / ``Infinity`` (not valid per RFC 8259).

    CPython's ``json.loads`` accepts those tokens by default. Scorers that
    claim JSON validity must not treat them as success. An integer past the
    configured digit limit raises ``ValueError`` rather than
    ``JSONDecodeError``; that is turned into ``JSONDecodeError`` so callers
    score 0 instead of aborting the run.
    """

    def reject_nonfinite(token: str) -> None:
        raise json.JSONDecodeError(f"{token} is not valid JSON", text, 0)

    try:
        return json.loads(text, parse_constant=reject_nonfinite)
    except json.JSONDecodeError:
        raise
    except ValueError as exc:
        raise json.JSONDecodeError(str(exc), text, 0) from exc


@register_scorer("json_valid", exact=True)
def json_valid(output: str, case: Case, assertion: AssertionSpec) -> float:
    """1.0 when the stripped output parses as one RFC 8259 JSON value.

    ``NaN``, ``Infinity``, and ``-Infinity`` are rejected even though
    CPython's ``json.loads`` accepts them by default.
    """
    try:
        _loads_rfc8259(output.strip())
    except json.JSONDecodeError:
        return 0.0
    return 1.0


@register_scorer("json_schema", exact=True)
def json_schema(output: str, case: Case, assertion: AssertionSpec) -> float:
    """1.0 when the output is JSON and satisfies a small schema subset.

    Supported keywords: ``type``, ``required``, ``properties``, ``items``,
    and ``enum``. ``type`` may be a string or a list of strings. Supported
    types are ``object``, ``array``, ``string``, ``number``, ``integer``,
    ``boolean``, and ``null``. Booleans are not numbers or integers.
    ``items`` must be a single schema object, not a list of schemas.
    ``NaN`` / ``Infinity`` are rejected at parse time (not RFC 8259 JSON).
    ``enum`` uses JSON equality: ``true`` does not match ``1``, and
    ``false`` does not match ``0``. ``1`` and ``1.0`` still match.

    Unsupported keywords (``format``, ``pattern``, ``minimum``, ``maximum``,
    ``minLength``, ``maxLength``, ``additionalProperties``, ``oneOf``,
    ``anyOf``, ``allOf``, ``$ref``, and others) are ignored. Extra object
    properties are allowed. This is a subset validator, not full JSON Schema.
    """
    if not isinstance(assertion.value, dict):
        raise ScoreError("json_schema assertion value must be a schema object")
    try:
        instance = _loads_rfc8259(output.strip())
    except json.JSONDecodeError:
        return 0.0
    return 1.0 if _schema_ok(instance, assertion.value) else 0.0


def _is_type(value: Any, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        # Every Python int is finite. ``math.isfinite`` converts through
        # float and raises OverflowError once the integer no longer fits.
        if isinstance(value, int):
            return True
        return math.isfinite(value)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    return False


def _type_matches(value: Any, type_spec: Any) -> bool:
    if isinstance(type_spec, list):
        return any(_type_matches(value, item) for item in type_spec)
    if not isinstance(type_spec, str):
        return False
    return _is_type(value, type_spec)


def _json_equal(left: Any, right: Any) -> bool:
    """JSON equality. Booleans are not numbers; ``1`` and ``1.0`` are equal."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, dict) and isinstance(right, dict):
        if len(left) != len(right):
            return False
        return all(key in right and _json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_json_equal(item, other) for item, other in zip(left, right))
    return bool(left == right)


def _schema_ok(instance: Any, schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    if "type" in schema and not _type_matches(instance, schema["type"]):
        return False
    if "enum" in schema:
        enum_values = schema["enum"]
        if not isinstance(enum_values, list) or not any(
            _json_equal(instance, item) for item in enum_values
        ):
            return False
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or not isinstance(instance, dict):
            return False
        for key in required:
            if not isinstance(key, str) or key not in instance:
                return False
    if "properties" in schema:
        properties = schema["properties"]
        if not isinstance(properties, dict) or not isinstance(instance, dict):
            return False
        for key, subschema in properties.items():
            if key in instance and not _schema_ok(instance[key], subschema):
                return False
    if "items" in schema:
        item_schema = schema["items"]
        if not isinstance(instance, list) or not isinstance(item_schema, dict):
            return False
        for item in instance:
            if not _schema_ok(item, item_schema):
                return False
    return True


def _expected_number(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        raise ScoreError("numeric_tolerance value must be a number")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError as exc:
            raise ScoreError("numeric_tolerance value must be a number") from exc
    else:
        raise ScoreError("numeric_tolerance value must be a number")
    if not math.isfinite(number):
        raise ScoreError("numeric_tolerance value must be finite")
    return number


def _tolerance(params: dict[str, Any]) -> float:
    raw = params.get("tol", 0.0)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ScoreError("numeric_tolerance tol must be a number")
    tol = float(raw)
    if not math.isfinite(tol) or tol < 0:
        raise ScoreError("numeric_tolerance tol must be non-negative")
    return tol


@register_scorer("numeric_tolerance")
def numeric_tolerance(output: str, case: Case, assertion: AssertionSpec) -> float:
    """Compare the first number in the output to ``value``.

    The first match of an optional sign, a decimal or integer, and an optional
    exponent is used. Thousands separators are not parsed (``1,000`` yields
    ``1``). If no number is present the score is 0.0.

    Let ``delta = abs(found - expected)`` and ``tol = params.tol`` (default
    ``0.0``, absolute). The score is 1.0 when ``delta <= tol``. Otherwise it
    decays linearly to 0 across an additional window of width ``tol`` (or
    ``1.0`` when ``tol`` is 0):

        max(0.0, 1.0 - (delta - tol) / window)

    A match that overflows ``float`` or is not finite (``inf``, ``nan``)
    scores 0.0. It does not raise.
    """
    expected = _expected_number(assertion.value)
    tol = _tolerance(assertion.params)
    match = _NUMBER_RE.search(output)
    if match is None:
        return 0.0
    try:
        found = float(match.group(0))
    except (ValueError, OverflowError):
        return 0.0
    if not math.isfinite(found):
        return 0.0
    delta = abs(found - expected)
    if delta <= tol:
        return 1.0
    window = tol if tol > 0.0 else 1.0
    return max(0.0, 1.0 - (delta - tol) / window)


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


@register_scorer("levenshtein")
def levenshtein(output: str, case: Case, assertion: AssertionSpec) -> float:
    """Normalized similarity ``1 - edit_distance / max(len(left), len(right))``.

    Both strings are stripped. They are casefolded unless
    ``params.case_sensitive`` is true. Two empty strings score 1.0. Internal
    whitespace is not collapsed, so those characters count as edits.
    """
    target = _target(case, assertion)
    if target is None:
        return 0.0
    if _case_sensitive(assertion.params):
        left = output.strip()
        right = target.strip()
    else:
        left = output.strip().casefold()
        right = target.strip().casefold()
    denom = max(len(left), len(right))
    if denom == 0:
        return 1.0
    return 1.0 - _edit_distance(left, right) / denom


def _tfidf_settings(params: dict[str, Any]) -> tuple[str, int]:
    mode = params.get("mode", "word")
    if not isinstance(mode, str):
        raise ScoreError("tfidf_cosine mode must be a string")
    if mode not in {"word", "char"}:
        raise ScoreError(f"tfidf_cosine mode must be 'word' or 'char', got {mode!r}")
    ngram = params.get("ngram", 1)
    if isinstance(ngram, bool) or not isinstance(ngram, int) or ngram < 1:
        raise ScoreError("tfidf_cosine ngram must be an integer >= 1")
    return mode, ngram


def _tokens(text: str, mode: str, ngram: int) -> list[str]:
    if mode == "char":
        folded = text.casefold()
        if not folded:
            return []
        if len(folded) < ngram:
            return [folded]
        return [folded[index : index + ngram] for index in range(len(folded) - ngram + 1)]
    words = _WORD_RE.findall(text.casefold())
    if not words:
        return []
    if len(words) < ngram:
        return [" ".join(words)]
    return [" ".join(words[index : index + ngram]) for index in range(len(words) - ngram + 1)]


def _tfidf_vectors(docs: list[list[str]]) -> list[dict[str, float]]:
    """Smoothed TF-IDF so identical documents still have cosine 1.

    With two documents, unsmoothed IDF is 0 for every shared term, which
    would zero the vectors of identical texts. The IDF used here is
    ``log((1 + N) / (1 + df)) + 1`` with ``N`` equal to the number of
    documents. An empty document yields an empty vector. An empty ``docs``
    returns ``[]``.
    """
    if not docs:
        return []
    document_frequency: dict[str, int] = {}
    counts: list[dict[str, int]] = []
    for tokens in docs:
        seen: dict[str, int] = {}
        for token in tokens:
            seen[token] = seen.get(token, 0) + 1
        counts.append(seen)
        for token in seen:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    n_docs = float(len(docs))
    vectors: list[dict[str, float]] = []
    for seen in counts:
        total = float(sum(seen.values()))
        vector: dict[str, float] = {}
        if total > 0.0:
            for token, count in seen.items():
                tf = count / total
                idf = math.log((1.0 + n_docs) / (1.0 + document_frequency[token])) + 1.0
                vector[token] = tf * idf
        vectors.append(vector)
    return vectors


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    dot = sum(weight * right.get(token, 0.0) for token, weight in left.items())
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    score = dot / (left_norm * right_norm)
    return min(1.0, max(0.0, score))


@register_scorer("tfidf_cosine")
def tfidf_cosine(output: str, case: Case, assertion: AssertionSpec) -> float:
    """Cosine similarity of smoothed TF-IDF vectors over the output and target.

    ``params.mode`` is ``word`` (default) or ``char``. ``params.ngram`` is the
    n-gram width (default 1). Tokens are casefolded. Both sides empty scores
    1.0. One side empty scores 0.0. In word mode, text with no word characters
    is empty only when it is empty after stripping. Two non-empty strings with
    no word tokens score 1.0 when they match after stripping and casefolding,
    and 0.0 when they differ. The target is ``value``, else the reference.
    """
    target = _target(case, assertion)
    if target is None:
        target = ""
    mode, ngram = _tfidf_settings(assertion.params)
    left_tokens = _tokens(output, mode, ngram)
    right_tokens = _tokens(target, mode, ngram)
    if not left_tokens and not right_tokens:
        # Word mode drops punctuation, so "!!!" and "???" both have no tokens.
        # That is not "both sides empty" unless the stripped text is empty.
        left_blank = output.strip() == ""
        right_blank = target.strip() == ""
        if left_blank and right_blank:
            return 1.0
        if left_blank or right_blank:
            return 0.0
        left_folded = output.strip().casefold()
        right_folded = target.strip().casefold()
        return 1.0 if left_folded == right_folded else 0.0
    if not left_tokens or not right_tokens:
        return 0.0
    vectors = _tfidf_vectors([left_tokens, right_tokens])
    return _cosine(vectors[0], vectors[1])


def _excerpt(text: str, limit: int = _NOTE_LIMIT) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _record_judge_failure(
    case: Case,
    message: str,
    raw: str | None,
    label: str | None = None,
) -> None:
    if label is None:
        note = f"llm_judge ({case.id}): {message}"
    else:
        note = f"llm_judge ({case.id}, {label}): {message}"
    detail: dict[str, Any] = {
        "kind": "llm_judge",
        "case_id": case.id,
        "error": note,
        "raw": raw,
    }
    if label is not None:
        detail["label"] = label
    ctx = _SCORING_CONTEXT.get()
    if ctx is not None:
        ctx.notes.append(note)
        ctx.details.append(detail)


def _coerce_finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError:
            return None
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except (ValueError, OverflowError):
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def _unit_interval(number: float) -> float | None:
    if 0.0 <= number <= 1.0:
        return number
    return None


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body).strip()


def _score_field(payload: dict[Any, Any]) -> tuple[bool, float | None]:
    for key, value in payload.items():
        if isinstance(key, str) and key.casefold() == "score":
            number = _coerce_finite(value)
            if number is None:
                return True, None
            return True, _unit_interval(number)
    return False, None


def _json_score(text: str) -> tuple[bool, float | None]:
    """Return ``(found_score_field, score_in_unit_interval_or_none)``."""
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        start = text.find("{", index)
        if start < 0:
            return False, None
        try:
            payload, end = decoder.raw_decode(text, start)
        except ValueError:
            # JSONDecodeError is a ValueError. An integer past the digit
            # limit is a bare ValueError. Both mean this '{' is not a
            # usable object; keep scanning so a later score can still win.
            index = start + 1
            continue
        index = end if end > start else start + 1
        if not isinstance(payload, dict):
            continue
        present, number = _score_field(payload)
        if present:
            return True, number
    return False, None


def _parse_judge_score(text: str) -> float:
    """Parse a score in ``[0, 1]`` from judge text.

    Accepted forms, in order: the whole reply is a number; a JSON object
    with a ``score`` field (optionally inside a code fence or surrounding
    text); a ``Score:`` label. The first JSON ``score`` field wins and is
    not replaced by a later label. Values outside ``[0, 1]`` are rejected.
    A label number glued to ``/`` or ``%`` (``Score: 1/5``, ``Score: 0.5%``)
    is not a bare score. An integer that does not fit in float is not a
    finite score in ``[0, 1]``.
    """
    stripped = _strip_fences(text.strip())
    if _PLAIN_NUMBER.fullmatch(stripped):
        return _bare_unit_score(stripped)
    found, score = _json_score(stripped)
    if found:
        if score is None:
            raise ValueError("JSON score is not a number in [0, 1]")
        return score
    match = _SCORE_LABEL.search(stripped)
    if match is not None:
        token = match.group(1)
        tail = stripped[match.end() :].lstrip()
        if tail.startswith(("/", "%")):
            raise ValueError(f"score {token}{tail[0]} is not a bare number in [0, 1]")
        return _bare_unit_score(token)
    excerpt = _excerpt(stripped)
    raise ValueError(f"could not parse a 0-1 score from judge output: {excerpt!r}")


def _bare_unit_score(token: str) -> float:
    """``token`` as a finite float in ``[0, 1]``, or ``ValueError``."""
    try:
        number = float(token)
    except OverflowError as exc:
        raise ValueError(f"score {token[:24]} is outside [0, 1]") from exc
    if not math.isfinite(number) or _unit_interval(number) is None:
        raise ValueError(f"score {token} is outside [0, 1]")
    return number


def _judge_prompt(case: Case, output: str, rubric: str) -> str:
    reference = "" if case.reference is None else case.reference
    return (
        "You are grading one model output. Use the rubric and reply with "
        'only a number from 0 to 1, a JSON object {"score": <number>}, '
        "or a line of the form Score: <number>.\n"
        f"Case: {case.id}\n"
        f"Rubric: {rubric}\n"
        f"Input: {case.input}\n"
        f"Reference: {reference}\n"
        f"Output: {output}\n"
    )


def _rubric_text(assertion: AssertionSpec) -> str:
    rubric = assertion.params.get("rubric")
    if not isinstance(rubric, str) or rubric.strip() == "":
        raise _JudgeFailure("params.rubric must be a non-empty string")
    return rubric.strip()


def _judge_provider() -> Provider:
    ctx = _SCORING_CONTEXT.get()
    if ctx is None or ctx.judge is None:
        raise _JudgeFailure("no judge provider on the scoring context")
    return ctx.judge


@register_scorer("llm_judge")
def llm_judge(output: str, case: Case, assertion: AssertionSpec) -> float:
    """Ask one judge, or the mean of an ensemble, for a score in ``[0, 1]``.

    Without ``params.judges``, ``params.rubric`` is the grading instructions
    and the provider is ``ScoringContext.judge`` from the active
    :func:`score_case` call. That path is unchanged: the same prompt, the
    same parsed score, and the same failure records.

    ``params.judges`` is a non-empty list of judge objects. Each object
    follows the single-judge settings: ``provider`` (``mock``, ``echo``, or
    ``openai``), optional ``model``, ``rubric`` (falls back to
    ``params.rubric``), and for ``mock`` optional ``mock_rules`` and
    ``mock_default``. ``openai`` also accepts ``api_key``, ``base_url``, and
    ``timeout``. The context judge is not called. The assertion score is the
    unweighted mean of the judge scores. A judge that cannot be scored
    contributes 0 and records an error. The outcome details include one
    ``llm_judge_ensemble`` record with each judge's score.

    The reply may be a plain number (``0.75``), JSON with a score field
    (``{"score": 0.75}``, a string score is accepted), or a line such as
    ``Score: 0.75``. Surrounding prose is fine for the JSON and label
    forms. A code fence around the JSON is stripped first.

    A missing rubric, a missing judge, a provider exception, or a reply
    that is not a finite score in ``[0, 1]`` scores 0.0 for that judge.
    The failure is recorded on the scoring context and on the
    :class:`ScoreOutcome`. This function does not raise for those failures.
    """
    if "judges" in assertion.params:
        return _llm_judge_ensemble(output, case, assertion)
    return _llm_judge_single(output, case, assertion)


def _llm_judge_single(output: str, case: Case, assertion: AssertionSpec) -> float:
    """Score with the context judge. Failure records match the historical shape."""
    raw: str | None = None
    try:
        rubric = _rubric_text(assertion)
        judge = _judge_provider()
        prompt = _judge_prompt(case, output, rubric)
        try:
            raw = judge.complete(prompt)
        except Exception as exc:
            raise _JudgeFailure(f"judge provider failed: {type(exc).__name__}: {exc}") from exc
        if not isinstance(raw, str):
            raise _JudgeFailure("judge provider returned a non-string")
        try:
            return _parse_judge_score(raw)
        except ValueError as exc:
            raise _JudgeFailure(str(exc), raw=raw) from exc
    except _JudgeFailure as exc:
        recorded = exc.raw
        if recorded is None and isinstance(raw, str):
            recorded = raw
        _record_judge_failure(case, exc.message, recorded)
        return 0.0


def _llm_judge_ensemble(output: str, case: Case, assertion: AssertionSpec) -> float:
    """Mean of ``params.judges``. A malformed list scores 0 and calls nobody."""
    try:
        specs = _parse_judge_list(assertion)
    except _JudgeFailure as exc:
        _record_judge_failure(case, exc.message, None)
        return 0.0
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for index, spec in enumerate(specs):
        score, error, raw = _score_configured_judge(case, output, spec)
        rows.append(
            {
                "index": index,
                "provider": spec.provider_name,
                "model": spec.model,
                "label": spec.label,
                "score": score,
                "error": error,
            }
        )
        scores.append(score)
        if error is not None:
            _record_judge_failure(case, error, raw, label=spec.label)
    mean = sum(scores) / len(scores)
    _record_ensemble(case, rows, mean)
    return mean


@dataclass(frozen=True)
class _JudgeSpec:
    provider: Provider
    provider_name: str
    model: str
    rubric: str
    label: str


def _parse_judge_list(assertion: AssertionSpec) -> list[_JudgeSpec]:
    raw = assertion.params.get("judges")
    if not isinstance(raw, list) or not raw:
        raise _JudgeFailure("params.judges must be a non-empty list")
    specs: list[_JudgeSpec] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise _JudgeFailure(f"params.judges[{index}] must be an object")
        specs.append(_parse_judge_spec(item, index, assertion))
    return _disambiguate_labels(specs)


def _parse_judge_spec(
    item: Mapping[str, Any],
    index: int,
    assertion: AssertionSpec,
) -> _JudgeSpec:
    kind = item.get("provider")
    if kind not in {"mock", "echo", "openai"}:
        raise _JudgeFailure(f"params.judges[{index}].provider must be 'mock', 'echo', or 'openai'")
    model = item.get("model")
    if model is not None and not isinstance(model, str):
        raise _JudgeFailure(f"params.judges[{index}].model must be a string")
    model_text = None if model is None or model.strip() == "" else model
    rubric = _entry_rubric(item, assertion, index)
    provider = _build_judge_provider(item, index, str(kind), model_text)
    resolved_model = provider_model_name(provider)
    return _JudgeSpec(
        provider=provider,
        provider_name=str(kind),
        model=resolved_model,
        rubric=rubric,
        label=f"{kind}/{resolved_model}",
    )


def _entry_rubric(item: Mapping[str, Any], assertion: AssertionSpec, index: int) -> str:
    if "rubric" in item and item["rubric"] is not None:
        rubric = item["rubric"]
    else:
        rubric = assertion.params.get("rubric")
    if not isinstance(rubric, str) or rubric.strip() == "":
        raise _JudgeFailure(
            f"params.judges[{index}].rubric must be a non-empty string "
            "(or set params.rubric as the fallback)"
        )
    return rubric.strip()


def _build_judge_provider(
    item: Mapping[str, Any],
    index: int,
    kind: str,
    model: str | None,
) -> Provider:
    if kind == "mock":
        return MockProvider(
            rules=_judge_mock_rules(item.get("mock_rules"), index),
            default=_judge_mock_default(item, index),
            name="mock",
            model=model or "mock",
        )
    if kind == "echo":
        return EchoProvider()
    timeout = _judge_timeout(item, index)
    api_key = _optional_judge_str(item, "api_key", index)
    base_url = _optional_judge_str(item, "base_url", index)
    try:
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            name="openai",
        )
    except ValueError as exc:
        raise _JudgeFailure(f"params.judges[{index}] {exc}") from exc


def _judge_mock_default(item: Mapping[str, Any], index: int) -> str:
    if "mock_default" not in item or item["mock_default"] is None:
        return ""
    default = item["mock_default"]
    if not isinstance(default, str):
        raise _JudgeFailure(f"params.judges[{index}].mock_default must be a string")
    return default


def _judge_mock_rules(raw: Any, index: int) -> list[tuple[str, str | tuple[str, ...]]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise _JudgeFailure(f"params.judges[{index}].mock_rules must be a list of pairs")
    rules: list[tuple[str, str | tuple[str, ...]]] = []
    for rule_index, rule in enumerate(raw):
        if not isinstance(rule, (list, tuple)) or len(rule) != 2 or not isinstance(rule[0], str):
            raise _JudgeFailure(
                f"params.judges[{index}].mock_rules[{rule_index}] must be "
                "[needle, response] with a string needle"
            )
        response = rule[1]
        if isinstance(response, str):
            rules.append((rule[0], response))
            continue
        if (
            isinstance(response, list)
            and response
            and all(isinstance(part, str) for part in response)
        ):
            rules.append((rule[0], tuple(response)))
            continue
        raise _JudgeFailure(
            f"params.judges[{index}].mock_rules[{rule_index}] response must be "
            "a string or a non-empty list of strings"
        )
    return rules


def _judge_timeout(item: Mapping[str, Any], index: int) -> float:
    if "timeout" not in item or item["timeout"] is None:
        return 60.0
    raw = item["timeout"]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise _JudgeFailure(f"params.judges[{index}].timeout must be a positive number")
    timeout = float(raw)
    if not math.isfinite(timeout) or timeout <= 0:
        raise _JudgeFailure(f"params.judges[{index}].timeout must be a positive number")
    return timeout


def _optional_judge_str(item: Mapping[str, Any], key: str, index: int) -> str | None:
    if key not in item or item[key] is None:
        return None
    value = item[key]
    if not isinstance(value, str):
        raise _JudgeFailure(f"params.judges[{index}].{key} must be a string")
    return value


def _disambiguate_labels(specs: list[_JudgeSpec]) -> list[_JudgeSpec]:
    labels = [spec.label for spec in specs]
    if len(set(labels)) == len(labels):
        return specs
    renamed: list[_JudgeSpec] = []
    for index, spec in enumerate(specs):
        renamed.append(
            _JudgeSpec(
                provider=spec.provider,
                provider_name=spec.provider_name,
                model=spec.model,
                rubric=spec.rubric,
                label=f"{spec.label}#{index}",
            )
        )
    return renamed


def _score_configured_judge(
    case: Case,
    output: str,
    spec: _JudgeSpec,
) -> tuple[float, str | None, str | None]:
    """Return ``(score, error, raw)``. A failure scores 0 and does not record."""
    raw: str | None = None
    try:
        prompt = _judge_prompt(case, output, spec.rubric)
        try:
            raw = spec.provider.complete(prompt)
        except Exception as exc:
            raise _JudgeFailure(f"judge provider failed: {type(exc).__name__}: {exc}") from exc
        if not isinstance(raw, str):
            raise _JudgeFailure("judge provider returned a non-string")
        try:
            return _parse_judge_score(raw), None, raw
        except ValueError as exc:
            raise _JudgeFailure(str(exc), raw=raw) from exc
    except _JudgeFailure as exc:
        recorded = exc.raw
        if recorded is None and isinstance(raw, str):
            recorded = raw
        return 0.0, exc.message, recorded


def _record_ensemble(case: Case, rows: list[dict[str, Any]], mean: float) -> None:
    detail: dict[str, Any] = {
        "kind": "llm_judge_ensemble",
        "case_id": case.id,
        "aggregation": "mean",
        "score": mean,
        "scores": rows,
    }
    ctx = _SCORING_CONTEXT.get()
    if ctx is not None:
        ctx.details.append(detail)


def ensemble_judge_scores(
    details: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, float], ...]:
    """``(label, score)`` pairs from ensemble records, in record order.

    Single-judge failure records are ignored. An empty result means the
    case did not use ``params.judges``.
    """
    found: list[tuple[str, float]] = []
    for detail in details:
        if detail.get("kind") != "llm_judge_ensemble":
            continue
        rows = detail.get("scores")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            label = row.get("label")
            score = row.get("score")
            if (
                not isinstance(label, str)
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
            ):
                continue
            found.append((label, float(score)))
    return tuple(found)


def _arg_text(value: Any) -> str | None:
    """Comparison text for one trace arg.

    A string is used unchanged, so the needle is not JSON-quoted. Any other
    value uses ``json.dumps`` (``true``, ``false``, ``null``, numbers, and
    objects with sorted keys). ``None`` means the value is not JSON text.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return None


def _trace_expected(value: Any) -> list[tuple[str, dict[str, str]]]:
    """Parse the assertion value into ``(tool, arg text)`` steps.

    Raises ``ScoreError`` when the value is not a list of step objects.
    ``args`` may be omitted or null and then means ``{}``.
    """
    if not isinstance(value, list):
        raise ScoreError("trace_match assertion value must be a list of steps {tool, args}")
    steps: list[tuple[str, dict[str, str]]] = []
    for index, step in enumerate(value, start=1):
        if not isinstance(step, dict):
            raise ScoreError(f"trace_match expected step {index} must be an object")
        tool = step.get("tool")
        if not isinstance(tool, str):
            raise ScoreError(f"trace_match expected step {index} tool must be a string")
        raw_args = step.get("args", {})
        if raw_args is None:
            raw_args = {}
        if not isinstance(raw_args, dict):
            raise ScoreError(f"trace_match expected step {index} args must be an object")
        args: dict[str, str] = {}
        for key, arg_value in raw_args.items():
            if not isinstance(key, str):
                raise ScoreError(f"trace_match expected step {index} arg keys must be strings")
            text = _arg_text(arg_value)
            if text is None:
                raise ScoreError(f"trace_match expected step {index} arg {key!r} is not JSON text")
            args[key] = text
        steps.append((tool, args))
    return steps


def _trace_output(output: str) -> list[Any] | None:
    """Parse a JSON array of steps. ``None`` means the output is not a trace."""
    try:
        parsed = _loads_rfc8259(output.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return parsed


def _trace_actual(value: Any) -> tuple[str, dict[str, str]] | None:
    """One actual step, or ``None`` when that element cannot match."""
    if not isinstance(value, dict):
        return None
    tool = value.get("tool")
    if not isinstance(tool, str):
        return None
    raw_args = value.get("args", {})
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, dict):
        return None
    args: dict[str, str] = {}
    for key, arg_value in raw_args.items():
        if not isinstance(key, str):
            return None
        text = _arg_text(arg_value)
        if text is None:
            return None
        args[key] = text
    return tool, args


def _trace_steps_match(
    expected: tuple[str, dict[str, str]],
    actual: tuple[str, dict[str, str]] | None,
) -> bool:
    """Tool names equal, and every expected arg text is a substring."""
    if actual is None:
        return False
    expected_tool, expected_args = expected
    actual_tool, actual_args = actual
    if expected_tool != actual_tool:
        return False
    for key, needle in expected_args.items():
        haystack = actual_args.get(key)
        if haystack is None or needle not in haystack:
            return False
    return True


def _trace_matched(expected: list[tuple[str, dict[str, str]]], actual: list[Any]) -> int:
    """Largest in-order pairing of expected steps to distinct actual steps.

    Skip an expected step, skip an actual step, or pair them when they
    match. A greedy left-to-right scan is not used: it can spend a loose
    early step and miss a longer later match.
    """
    parsed = [_trace_actual(step) for step in actual]
    n_expected = len(expected)
    n_actual = len(parsed)
    previous = [0] * (n_actual + 1)
    for i in range(1, n_expected + 1):
        current = [0] * (n_actual + 1)
        exp = expected[i - 1]
        for j in range(1, n_actual + 1):
            best = previous[j]
            if current[j - 1] > best:
                best = current[j - 1]
            if _trace_steps_match(exp, parsed[j - 1]):
                take = previous[j - 1] + 1
                if take > best:
                    best = take
            current[j] = best
        previous = current
    return previous[n_actual]


@register_scorer("trace_match")
def trace_match(output: str, case: Case, assertion: AssertionSpec) -> float:
    """Fraction of expected trace steps matched in order.

    The expected trace is ``assertion.value``: a list of step objects
    ``{"tool": "<name>", "args": {"<key>": <value>, ...}}``. ``args`` may be
    omitted or null and means ``{}``. Only ``tool`` and ``args`` are read.
    This is the same ``value`` field ``json_schema`` uses for a structured
    document, so suite JSON and YAML already carry it. ``params`` is not
    read. A value that is not a list of steps raises ``ScoreError``.

    The output, after stripping ends, must be one JSON array of step objects
    (RFC 8259: ``NaN`` and ``Infinity`` are rejected). Anything else,
    including a JSON object, a JSON string, or text around the array, scores
    0.0. An element that is not an object, whose ``tool`` is not a string,
    or whose ``args`` is present and not an object, never matches.

    A step matches when all of the following hold:

    - Tool names are equal and case-sensitive. ``Search`` does not match
      ``search``.
    - Every expected arg key is present on the actual step. Extra actual
      keys are ignored. A missing key does not match, even when the expected
      text is empty.
    - The expected arg's comparison text is a case-sensitive substring of
      the actual arg's comparison text. A string is that text unchanged, so
      ``"20"`` matches ``"20.00"`` and ``"200"``. Any other JSON value uses
      ``json.dumps`` (``true``, ``false``, ``null``, numbers, and objects
      with sorted keys). An expected ``""`` matches any present value.

    The score is ``matched / len(expected)``. ``matched`` is the size of the
    largest in-order pairing: expected indexes ``i1 < i2 < ... < ik`` and
    actual indexes ``j1 < j2 < ... < jk`` where step ``i_t`` matches step
    ``j_t``. Extra actual steps are skipped. Each actual step is used at
    most once. The count is that subsequence recurrence (skip an expected
    step, skip an actual step, or pair them when they match), not a greedy
    left-to-right scan. A greedy scan can spend an early loose step and
    miss a longer match. A missing middle step still credits the steps
    before and after it when those stay in order.

    Worked counts for expected tools A, B, C:

    - Actual A, B, C scores 1.
    - Actual A, C (B missing) scores 2/3.
    - Actual C, B, A (reversed) scores 1/3.
    - Actual with extra steps inserted around A, B, C scores 1.

    An empty expected list scores 1.0 and does not parse the output. The
    scorer is graded: a partial score passes the case only when the weighted
    mean is at least 0.5 and every exact-type assertion on the case passes.
    """
    expected = _trace_expected(assertion.value)
    if not expected:
        return 1.0
    parsed = _trace_output(output)
    if parsed is None:
        return 0.0
    return _trace_matched(expected, parsed) / len(expected)


def load_scorer_plugins() -> None:
    """Register scorers from the ``promptregress.scorers`` entry-point group.

    No entry points is normal and does nothing. Each entry point is loaded
    once per call. Its name is the scorer kind. If that name is already
    registered — a built-in, or a callable that registered itself when it
    was imported — the existing scorer is kept, including an ``exact=True``
    registration. Otherwise the loaded object must be a callable
    ``(output, case, assertion) -> float`` and is registered as a graded
    scorer. A non-callable raises :class:`ScoreError`. An exception raised
    while loading an entry point propagates.
    """
    discovered = importlib.metadata.entry_points(group="promptregress.scorers")
    for entry_point in discovered:
        loaded = entry_point.load()
        name = entry_point.name
        if name in SCORERS:
            continue
        if not callable(loaded):
            raise ScoreError(
                f"Entry point {name!r} in group 'promptregress.scorers' "
                "did not load a callable scorer"
            )
        register_scorer(name)(cast(Scorer, loaded))


load_scorer_plugins()
