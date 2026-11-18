"""Seeded property checks for loaders, paired stats, and deterministic scorers.

No hypothesis. ``random.Random`` uses fixed seeds. A few hundred draws each.
Malformed JSON, YAML, and JSONL must load or raise ``SuiteError``.
"""

import json
import math
import random
from pathlib import Path

from promptregress.loading import SuiteError, load_suite
from promptregress.models import AssertionSpec, Case
from promptregress.scorers import SCORERS, score_case
from promptregress.stats import (
    cohens_dz,
    mcnemar_p_value,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)

LOADER_SEED = 401
STATS_SEED = 402
SCORER_SEED = 403
DRAWS = 300

_ALPHABET = "abcXYZ \n\t{}[]\"':,-.01eE+/\\<>[1]é你"
_DETERMINISTIC = (
    "exact_match",
    "contains",
    "regex",
    "json_valid",
    "json_schema",
    "numeric_tolerance",
    "levenshtein",
    "tfidf_cosine",
    "trace_match",
)
_REGEXES = (r"a", r"\w+", r"[0-9]+", r"a?", r"^", r"(?i)abc", r"colou?r")
_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["id"],
    "properties": {"id": {"type": "string"}, "n": {"type": "integer"}},
}


def _text(rng: random.Random, limit: int = 40) -> str:
    count = rng.randrange(0, limit + 1)
    return "".join(rng.choice(_ALPHABET) for _ in range(count))


def _json_value(rng: random.Random, depth: int = 0) -> object:
    roll = rng.randrange(6 if depth >= 3 else 0, 7)
    if roll == 0:
        return {_text(rng, 8): _json_value(rng, depth + 1) for _ in range(rng.randrange(0, 3))}
    if roll == 1:
        return [_json_value(rng, depth + 1) for _ in range(rng.randrange(0, 3))]
    if roll == 2:
        return rng.choice([None, True, False])
    if roll == 3:
        return rng.choice(["", "id", "cases", "input", 0, 1, -3, 1.5])
    if roll == 4:
        return _text(rng, 24)
    return rng.randrange(-4, 5)


def _valid_suite_text() -> str:
    return json.dumps(
        {
            "name": "fuzz",
            "cases": [
                {
                    "id": "a",
                    "input": "hello",
                    "reference": "hello",
                    "tags": ["fuzz"],
                    "assertions": [{"kind": "contains", "value": "hello"}],
                }
            ],
        }
    )


def _fixed_payloads() -> list[tuple[str, str | bytes]]:
    huge = "9" * 5000
    deep_json = "[" * 10000 + "]" * 10000
    deep_yaml = "a: " + ("[" * 500) + ("]" * 500)
    return [
        (".json", _valid_suite_text()),
        (".yaml", "name: fuzz\ncases:\n- id: a\n  input: hello\n  reference: hello\n"),
        (".jsonl", '{"id": "a", "input": "hello", "reference": "hello"}\n'),
        (".json", ""),
        (".json", "{"),
        (".json", "[]"),
        (".json", "null"),
        (".yaml", ":\n  - ["),
        (".yaml", ""),
        (".jsonl", "\n\n{nope\n"),
        (".json", b"\xff\xfe{"),
        (".yaml", b"\x80\x81"),
        (".jsonl", b"\xff"),
        (
            ".json",
            '{"name":"n","cases":[{"id":"a","input":"q","assertions":[{"kind":"contains","weight":'
            + huge
            + "}]}]}",
        ),
        (".json", '{"name":"n","cases":' + deep_json + "}"),
        (".yaml", deep_yaml),
        (
            ".yaml",
            "name: n\ncases:\n- id: a\n  input: q\n  assertions:\n  - kind: contains\n    weight: "
            + huge
            + "\n",
        ),
        (
            ".json",
            json.dumps(
                {
                    "name": "dup",
                    "cases": [{"id": "a", "input": "1"}, {"id": "a", "input": "2"}],
                }
            ),
        ),
        (".jsonl", '{"id":"a","input":"q"}\n{"id":"a","input":"z"}\n'),
    ]


def _random_payload(rng: random.Random) -> tuple[str, str | bytes]:
    kind = rng.randrange(8)
    if kind == 0:
        raw = _text(rng, 60)
        return (rng.choice([".json", ".yaml", ".jsonl"]), raw)
    if kind == 1:
        blob = json.dumps(_json_value(rng))
        if rng.random() < 0.5 and blob:
            blob = blob[: rng.randrange(0, len(blob))]
        return (".json", blob)
    if kind == 2:
        case = _json_value(rng)
        return (
            ".json",
            json.dumps({"name": _json_value(rng), "cases": case, "description": _json_value(rng)}),
        )
    if kind == 3:
        lines = []
        for _ in range(rng.randrange(0, 4)):
            if rng.random() < 0.5:
                lines.append(_text(rng, 30))
            else:
                lines.append(json.dumps({"id": _text(rng, 6), "input": _text(rng, 12)}))
        return (".jsonl", "\n".join(lines))
    if kind == 4:
        return (".yaml", _text(rng, 80))
    if kind == 5:
        return (rng.choice([".json", ".yaml", ".jsonl"]), rng.randbytes(rng.randrange(1, 12)))
    if kind == 6:
        return (".json", _valid_suite_text())
    weight = rng.choice([0, 1, -1, True, "x", None, 1.5])
    body = {
        "name": rng.choice(["fuzz", "", 1, None]),
        "cases": [
            {
                "id": rng.choice(["a", "", 2, None]),
                "input": rng.choice(["q", 1, None]),
                "reference": rng.choice(["r", None, 3]),
                "tags": rng.choice([["t"], ["t", 1], "tag", None]),
                "assertions": [{"kind": rng.choice(["contains", "", 1]), "weight": weight}],
            }
        ],
    }
    return (".json", json.dumps(body))


def _feed(path: Path, payload: str | bytes) -> str:
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    try:
        load_suite(path)
    except SuiteError:
        return "error"
    except Exception as exc:
        snippet = payload[:180] if isinstance(payload, str) else payload[:40]
        raise AssertionError(
            f"{path.name} raised {type(exc).__name__}: {exc} payload={snippet!r}"
        ) from exc
    return "load"


def test_loaders_survive_random_inputs(tmp_path: Path) -> None:
    rng = random.Random(LOADER_SEED)
    loads = 0
    errors = 0
    payloads: list[tuple[str, str | bytes]] = list(_fixed_payloads())
    payloads.extend(_random_payload(rng) for _ in range(DRAWS))
    for index, (suffix, payload) in enumerate(payloads):
        path = tmp_path / f"suite{suffix}"
        try:
            outcome = _feed(path, payload)
        except AssertionError:
            raise
        except Exception as exc:
            raise AssertionError(
                f"iter {index} crashed before load: {type(exc).__name__}: {exc}"
            ) from exc
        if outcome == "load":
            loads += 1
        else:
            errors += 1
    assert loads > 0
    assert errors > 50
    assert loads + errors == len(payloads)


def _diff_vector(rng: random.Random) -> list[float]:
    kind = rng.randrange(6)
    if kind == 0:
        return []
    if kind == 1:
        return [rng.choice([-2.0, -0.5, 0.0, 0.25, 1.0])]
    if kind == 2:
        value = rng.choice([-3.0, -1.0, 0.0, 0.5, 2.0])
        return [value] * rng.randrange(2, 18)
    if kind == 3:
        return [0.0] * rng.randrange(2, 12)
    width = rng.randrange(2, 36)
    return [rng.uniform(-4.0, 4.0) for _ in range(width)]


def test_stats_invariants_on_random_diffs() -> None:
    rng = random.Random(STATS_SEED)
    for index in range(DRAWS):
        diffs = _diff_vector(rng)
        for method in ("percentile", "bca"):
            low, high = paired_bootstrap_ci(
                diffs,
                n_boot=200,
                seed=rng.randrange(0, 50_000),
                method=method,
            )
            assert math.isfinite(low) and math.isfinite(high)
            assert low <= high, f"iter {index} {method} CI inverted: {low} > {high}"
        _stat, wilcoxon_p = wilcoxon_signed_rank(diffs)
        assert math.isfinite(wilcoxon_p)
        assert 0.0 <= wilcoxon_p <= 1.0
        dz = cohens_dz(diffs)
        if len(diffs) < 2:
            assert dz == 0.0
        else:
            mean = sum(diffs) / len(diffs)
            if mean == 0.0:
                assert dz == 0.0
            elif mean > 0.0:
                assert dz > 0.0
            else:
                assert dz < 0.0
        discordant_b = rng.randrange(0, 120)
        discordant_c = rng.randrange(0, 120)
        p_value = mcnemar_p_value(discordant_b, discordant_c)
        assert math.isfinite(p_value)
        assert 0.0 <= p_value <= 1.0


def _assertion(rng: random.Random, kind: str, reference: str) -> AssertionSpec:
    if kind == "regex":
        return AssertionSpec(
            kind=kind, value=rng.choice(_REGEXES), params={"ignore_case": rng.choice([True, False])}
        )
    if kind == "json_schema":
        return AssertionSpec(kind=kind, value=dict(_SCHEMA))
    if kind == "json_valid":
        return AssertionSpec(kind=kind)
    if kind == "numeric_tolerance":
        return AssertionSpec(
            kind=kind,
            value=rng.uniform(-50.0, 50.0),
            params={"tol": rng.choice([0.0, 0.01, 1.0, 3.0])},
        )
    if kind == "tfidf_cosine":
        return AssertionSpec(
            kind=kind,
            value=reference,
            params={"mode": rng.choice(["word", "char"]), "ngram": rng.choice([1, 2, 3])},
        )
    if kind == "trace_match":
        return AssertionSpec(kind=kind, value=[{"tool": "lookup", "args": {"q": reference}}])
    return AssertionSpec(
        kind=kind,
        value=reference,
        params={"case_sensitive": rng.choice([True, False])},
    )


def _assert_unit(kind: str, output: str, case: Case, assertion: AssertionSpec) -> None:
    raw = SCORERS[kind](output, case, assertion)
    value = float(raw)
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0
    outcome = score_case(case, output)
    assert 0.0 <= outcome[1] <= 1.0
    assert math.isfinite(outcome[1])
    for sub in outcome[0].values():
        assert 0.0 <= sub <= 1.0


def test_scorers_survive_random_strings() -> None:
    for kind in _DETERMINISTIC:
        assert kind in SCORERS
    rng = random.Random(SCORER_SEED)
    hostile = (
        "",
        "1e999999",
        "1e309",
        "nan",
        "inf",
        "-inf",
        "\x00",
        "{",
        "]",
        "9" * 400,
        "[[[",
        "Score: 9",
        " " * 8,
    )
    for index in range(DRAWS):
        reference = _text(rng, 32)
        if index % 5 == 0:
            output = hostile[index % len(hostile)]
        else:
            output = _text(rng, 32)
        # Levenshtein is quadratic. Keep that pair short; other scorers see
        # the full hostile string, including overflowing numbers.
        short_output = output[:32]
        short_reference = reference[:32]
        for kind in _DETERMINISTIC:
            if kind == "levenshtein":
                seen_output = short_output
                seen_reference = short_reference
            else:
                seen_output = output
                seen_reference = reference
            assertion = _assertion(rng, kind, seen_reference)
            case = Case(
                id=f"fuzz-{index}",
                input=_text(rng, 16),
                reference=seen_reference,
                assertions=[assertion],
                tags=["fuzz"],
            )
            try:
                _assert_unit(kind, seen_output, case, assertion)
            except Exception as exc:
                preview = seen_output[:40]
                raise AssertionError(
                    f"iter {index} {kind} raised {type(exc).__name__}: {exc} output={preview!r}"
                ) from exc
    overflow = Case(
        id="overflow",
        input="n",
        reference="0",
        assertions=[AssertionSpec(kind="numeric_tolerance", value=0, params={"tol": 0})],
    )
    assert SCORERS["numeric_tolerance"]("1e999999", overflow, overflow.assertions[0]) == 0.0
    assert SCORERS["numeric_tolerance"]("9" * 5000, overflow, overflow.assertions[0]) == 0.0
    assert SCORERS["numeric_tolerance"]("1e309", overflow, overflow.assertions[0]) == 0.0
