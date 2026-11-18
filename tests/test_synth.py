"""Adversarial suite synthesis: operators, metadata, and the synth CLI."""

import json
import random
from pathlib import Path

import pytest

from promptregress.cli import main
from promptregress.loading import load_suite
from promptregress.models import AssertionSpec, Case, ScoringConfig, Suite
from promptregress.providers import EchoProvider
from promptregress.runner import run_suite
from promptregress.synth import (
    DISTRACTORS,
    HOMOGLYPHS,
    OPERATOR_NAMES,
    OPERATORS,
    PUNCTUATION_SWAPS,
    SYNONYM_MAP,
    apply_operator,
    distractor,
    homoglyph,
    instruction_rephrase,
    number_format,
    punctuation,
    synonym,
    synthesize_suite,
    variant_seed,
    whitespace_casing,
    write_suite,
)


def _toy_suite() -> Suite:
    return Suite(
        name="toy",
        description="A toy suite.",
        cases=[
            Case(
                id="q1",
                input="Please begin quickly. The large total is 1,000.",
                reference="start",
                assertions=[
                    AssertionSpec(kind="contains", value="start"),
                    AssertionSpec(
                        kind="numeric_tolerance",
                        value=1000,
                        params={"tol": 0.5},
                    ),
                ],
                tags=["demo"],
                metadata={"priority": "high"},
            ),
            Case(
                id="q2",
                input="Can you assist with a simple purchase?",
                reference="help",
                assertions=[AssertionSpec(kind="exact_match", value="help")],
                tags=["billing"],
                metadata={},
            ),
        ],
        scoring=ScoringConfig(weights={"contains": 2.0}),
    )


def test_operator_names_are_the_registered_functions() -> None:
    assert OPERATOR_NAMES == (
        "whitespace_casing",
        "punctuation",
        "synonym",
        "instruction_rephrase",
        "distractor",
        "homoglyph",
        "number_format",
    )
    for name in OPERATOR_NAMES:
        assert OPERATORS[name].__name__ == name


def test_whitespace_casing_applies_skips_and_follows_the_seed() -> None:
    assert whitespace_casing("Ab Cd", 0) == "ab  Cd"
    assert whitespace_casing("Ab Cd", 1) == "AB  Cd"
    assert whitespace_casing("Ab Cd", 0) == whitespace_casing("Ab Cd", 0)
    assert whitespace_casing("Hello", 0) == "hello"
    assert whitespace_casing("   ", 0) == "    "
    assert whitespace_casing("123", 0) == "123"
    assert whitespace_casing("123", 4) == "123"
    assert whitespace_casing("", 0) == ""
    assert whitespace_casing("é", 0) == "é"
    assert whitespace_casing("a\tb", 0) == "A\t\tb"


def test_punctuation_applies_skips_and_ignores_the_seed() -> None:
    assert punctuation("Hello.", 0) == "Hello?"
    assert punctuation("Hello?", 1) == "Hello."
    assert punctuation("Hello!", 2) == "Hello."
    assert punctuation("Wait; go", 3) == "Wait, go"
    assert punctuation("a: b", 4) == "a; b"
    assert punctuation("Hello.", 0) == punctuation("Hello.", 9)
    assert punctuation("1.5", 0) == "1.5"
    assert punctuation("e.g", 0) == "e.g"
    assert punctuation("no punct", 0) == "no punct"
    assert set(PUNCTUATION_SWAPS) == {".", "?", "!", ";", ":"}


def test_synonym_applies_skips_and_ignores_the_seed() -> None:
    assert synonym("Please begin.", 0) == "Please start."
    assert synonym("BEGIN", 1) == "START"
    assert synonym("Begin", 2) == "Start"
    assert synonym("begin", 0) == synonym("begin", 8)
    assert synonym("beginning purchases", 0) == "beginning purchases"
    assert synonym("nothing here", 0) == "nothing here"
    text = "Please begin quickly. The large total is 1,000."
    assert synonym(text, 0) == "Please start rapidly. The big total is 1,000."
    assert set(SYNONYM_MAP) == {
        "quickly",
        "large",
        "little",
        "begin",
        "purchase",
        "difficult",
        "simple",
        "glad",
        "assist",
        "additional",
    }


def test_instruction_rephrase_applies_skips_and_follows_the_seed() -> None:
    assert instruction_rephrase("Hi", 0) == "Please answer: Hi"
    assert instruction_rephrase("Hi", 1) == "Q: Hi A:"
    assert instruction_rephrase("Hi", 2) == "Please answer: Hi"
    assert instruction_rephrase("Hi", 0) == instruction_rephrase("Hi", 4)
    assert instruction_rephrase("   ", 0) == "   "
    assert instruction_rephrase("", 1) == ""
    assert instruction_rephrase("  Hi  ", 0) == "Please answer:   Hi  "


def test_distractor_applies_skips_and_follows_the_seed() -> None:
    assert distractor("Hi", 0) == f"{DISTRACTORS[0]} Hi"
    assert distractor("Hi", 1) == f"{DISTRACTORS[1]} Hi"
    assert distractor("Hi", 4) == f"Hi {DISTRACTORS[0]}"
    assert distractor("Hi", 0) == distractor("Hi", 0)
    assert distractor("Hi", 0) != distractor("Hi", 1)
    assert distractor(" \n\t", 3) == " \n\t"
    assert distractor("", 0) == ""
    assert len(DISTRACTORS) == 4


def test_homoglyph_applies_skips_and_follows_the_seed() -> None:
    assert homoglyph("aeo", 0) == "\u0430eo"
    assert homoglyph("aeo", 1) == "a\u0435o"
    assert homoglyph("aeo", 2) == "ae\u043e"
    assert homoglyph("aeo", 3) == homoglyph("aeo", 0)
    assert homoglyph("Hello", 0) != "Hello"
    assert len(homoglyph("Hello", 0)) == len("Hello")
    assert homoglyph("1234-!?", 0) == "1234-!?"
    assert homoglyph("bd", 2) == "bd"
    swapped = homoglyph("Hello", 0)
    assert sum(left != right for left, right in zip("Hello", swapped, strict=True)) == 1
    assert any(char in HOMOGLYPHS.values() for char in swapped)


def test_number_format_applies_skips_and_ignores_the_seed() -> None:
    assert number_format("cost 1,000 dollars", 0) == "cost 1000 dollars"
    assert number_format("cost 1000 dollars", 0) == "cost 1,000 dollars"
    assert number_format("1,000", 1) == number_format("1,000", 7)
    assert number_format(number_format("1,000"), 0) == "1,000"
    assert number_format(number_format("1000"), 0) == "1000"
    assert number_format("see 1000.50 now", 0) == "see 1,000.50 now"
    assert number_format("see 1,000.50 now", 0) == "see 1000.50 now"
    assert number_format("Total is 1,000.", 0) == "Total is 1000."
    assert number_format("1000 and 1,000", 0) == "1,000 and 1000"
    assert number_format("12", 0) == "12"
    assert number_format("1.000", 0) == "1.000"
    assert number_format("1e1000", 0) == "1e1000"
    assert number_format("A1000", 0) == "A1000"
    assert number_format("01000", 0) == "01000"
    assert number_format("no digits", 0) == "no digits"


def test_operators_do_not_read_the_global_random_stream() -> None:
    text = "Please begin quickly. The large total is 1,000."
    random.seed(1)
    first = {name: OPERATORS[name](text, 3) for name in OPERATOR_NAMES}
    random.seed(2)
    second = {name: OPERATORS[name](text, 3) for name in OPERATOR_NAMES}
    assert first == second


def test_seed_must_be_an_integer() -> None:
    with pytest.raises(ValueError, match="seed must be an integer"):
        whitespace_casing("Hello", True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed must be an integer"):
        punctuation("Hello.", "nope")  # type: ignore[arg-type]


def test_unknown_operator_and_bad_operator_list() -> None:
    with pytest.raises(ValueError, match="unknown operator 'nope'"):
        apply_operator("nope", "Hello", 0)
    suite = _toy_suite()
    with pytest.raises(ValueError, match="unknown operator 'nope'"):
        synthesize_suite(suite, operators=["punctuation", "nope"])
    with pytest.raises(ValueError, match="repeated"):
        synthesize_suite(suite, operators=["punctuation", "punctuation"])
    with pytest.raises(ValueError, match="not a string"):
        synthesize_suite(suite, operators="punctuation")
    with pytest.raises(ValueError, match="empty"):
        synthesize_suite(suite, operators=["  "])
    with pytest.raises(ValueError, match="per-case"):
        synthesize_suite(suite, per_case=0)
    with pytest.raises(ValueError, match="empty"):
        synthesize_suite(Suite(name="empty", cases=[]))


def test_variant_seed_is_stable_and_depends_on_each_part() -> None:
    first = variant_seed(13, "q1", "punctuation", 0)
    assert first == variant_seed(13, "q1", "punctuation", 0)
    assert first != variant_seed(14, "q1", "punctuation", 0)
    assert first != variant_seed(13, "q2", "punctuation", 0)
    assert first != variant_seed(13, "q1", "synonym", 0)
    assert first != variant_seed(13, "q1", "punctuation", 1)
    assert first >= 0


def test_metadata_preserves_the_case_and_records_the_operator() -> None:
    suite = _toy_suite()
    out = synthesize_suite(suite, operators=["punctuation", "synonym"], per_case=2, seed=13)
    assert out.name == "toy-synth"
    assert "not proven" in out.description
    assert out.scoring.weights == {"contains": 2.0}
    assert len(out.cases) == len(suite.cases) * 2 * 2
    sources = {case.id: case for case in suite.cases}
    ids = [case.id for case in out.cases]
    assert len(ids) == len(set(ids))
    for case in out.cases:
        source_id = case.metadata["source_case_id"]
        operator = case.metadata["operator"]
        index = case.metadata["variant_index"]
        source = sources[str(source_id)]
        assert operator in {"punctuation", "synonym"}
        assert index in {0, 1}
        assert case.id == f"{source_id}__{operator}__{index}"
        assert case.reference == source.reference
        assert case.assertions == source.assertions
        assert case.tags == source.tags
        for key, value in source.metadata.items():
            assert case.metadata[key] == value
        mixed = variant_seed(13, source.id, str(operator), int(index))
        assert case.input == apply_operator(str(operator), source.input, mixed)
        assert case.metadata["changed"] is (case.input != source.input)
    single = synthesize_suite(suite, operators=["number_format"], per_case=1, seed=0)
    by_id = {case.id: case for case in single.cases}
    assert by_id["q1__number_format"].input == "Please begin quickly. The large total is 1000."
    assert by_id["q1__number_format"].metadata["changed"] is True
    assert by_id["q2__number_format"].input == "Can you assist with a simple purchase?"
    assert by_id["q2__number_format"].metadata["changed"] is False
    assert by_id["q2__number_format"].id == "q2__number_format"


def test_synthesize_is_deterministic() -> None:
    suite = _toy_suite()
    assert synthesize_suite(suite, seed=13) == synthesize_suite(suite, seed=13)
    assert synthesize_suite(suite, seed=13) != synthesize_suite(suite, seed=14)


def test_suite_round_trip_json_yaml_jsonl_and_csv(tmp_path: Path) -> None:
    suite = _toy_suite()
    json_path = tmp_path / "toy.json"
    write_suite(suite, json_path)
    loaded = load_suite(json_path)
    assert loaded.name == "toy"
    assert loaded.description == "A toy suite."
    assert loaded.scoring.weights == {"contains": 2.0}
    assert loaded.cases[0].input == suite.cases[0].input
    assert loaded.cases[0].assertions[1].params == {"tol": 0.5}
    assert loaded.cases[0].assertions[1].value == 1000
    assert loaded.cases[0].metadata == {"priority": "high"}

    yaml_path = tmp_path / "toy.yaml"
    write_suite(suite, yaml_path)
    yaml_loaded = load_suite(yaml_path)
    assert yaml_loaded.cases[0].id == "q1"
    assert yaml_loaded.cases[0].metadata["priority"] == "high"
    assert yaml_loaded.scoring.weights["contains"] == 2.0

    jsonl_path = tmp_path / "toy.jsonl"
    write_suite(suite, jsonl_path)
    jsonl_loaded = load_suite(jsonl_path)
    assert jsonl_loaded.name == "toy"
    assert jsonl_loaded.cases[1].metadata == {}
    assert jsonl_loaded.cases[0].assertions[0].kind == "contains"

    csv_path = tmp_path / "toy.csv"
    write_suite(suite, csv_path)
    csv_loaded = load_suite(csv_path)
    assert [case.id for case in csv_loaded.cases] == ["q1", "q2"]
    assert csv_loaded.cases[0].input == suite.cases[0].input
    assert csv_loaded.cases[0].assertions[0].kind == "exact_match"

    with pytest.raises(ValueError, match="Unsupported suite format"):
        write_suite(suite, tmp_path / "toy.txt")


def test_cli_round_trip_count_and_input_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "toy.json"
    write_suite(_toy_suite(), source)
    before = source.read_bytes()
    out = tmp_path / "nested" / "out.json"
    out.parent.mkdir()
    code = main(
        [
            "synth",
            "--suite",
            str(source),
            "--out",
            str(out),
            "--operators",
            "punctuation, synonym",
            "--per-case",
            "2",
            "--seed",
            "13",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert source.read_bytes() == before
    assert "out_cases=8" in captured.out
    assert "in_cases=2" in captured.out
    loaded = load_suite(out)
    original = load_suite(source)
    assert len(loaded.cases) == len(original.cases) * 2 * 2
    assert {case.metadata["operator"] for case in loaded.cases} == {"punctuation", "synonym"}
    assert {case.metadata["source_case_id"] for case in loaded.cases} == {"q1", "q2"}
    assert all(isinstance(case.metadata["changed"], bool) for case in loaded.cases)
    assert all(isinstance(case.metadata["variant_index"], int) for case in loaded.cases)
    assert loaded.name == "toy-synth"
    assert loaded.scoring.weights == {"contains": 2.0}
    result = run_suite(loaded, EchoProvider(), "echo")
    assert result.n_cases == len(loaded.cases)
    assert [item.case_id for item in result.results] == [case.id for case in loaded.cases]


def test_cli_unknown_operator_leaves_the_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "toy.json"
    write_suite(_toy_suite(), source)
    before = source.read_bytes()
    out = tmp_path / "out.json"
    code = main(["synth", "--suite", str(source), "--out", str(out), "--operators", "nope"])
    err = capsys.readouterr().err
    assert code == 2
    assert "unknown operator" in err
    assert "nope" in err
    assert not out.exists()
    assert source.read_bytes() == before


def test_cli_empty_suite_and_unwritable_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "empty.json"
    source.write_text(json.dumps({"name": "empty", "cases": []}) + "\n", encoding="utf-8")
    before = source.read_bytes()
    out = tmp_path / "out.json"
    code = main(["synth", "--suite", str(source), "--out", str(out)])
    assert code == 2
    assert "empty" in capsys.readouterr().err
    assert not out.exists()
    assert source.read_bytes() == before

    toy = tmp_path / "toy.json"
    write_suite(_toy_suite(), toy)
    toy_before = toy.read_bytes()
    directory = tmp_path / "out.json"
    directory.mkdir()
    code = main(["synth", "--suite", str(toy), "--out", str(directory)])
    assert code == 2
    assert "cannot write suite" in capsys.readouterr().err
    assert directory.is_dir()
    assert toy.read_bytes() == toy_before

    missing = tmp_path / "missing" / "out.json"
    code = main(["synth", "--suite", str(toy), "--out", str(missing)])
    assert code == 2
    assert "cannot write suite" in capsys.readouterr().err
    assert not missing.exists()
    assert toy.read_bytes() == toy_before

    code = main(["synth", "--suite", str(toy), "--out", str(toy), "--operators", "punctuation"])
    assert code == 2
    assert "different path" in capsys.readouterr().err
    assert toy.read_bytes() == toy_before

    code = main(
        [
            "synth",
            "--suite",
            str(toy),
            "--out",
            str(tmp_path / "zero.json"),
            "--per-case",
            "0",
        ]
    )
    assert code == 2
    assert "per-case" in capsys.readouterr().err
    assert not (tmp_path / "zero.json").exists()
    assert toy.read_bytes() == toy_before
