"""Real branches that the rest of the suite left uncovered.

Each test checks the value or the error the function produces.
"""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from helpers import comparison_with, run_with
from promptregress.annotations import annotation_lines
from promptregress.baseline import first_differing_case_id, list_baselines, load_baseline
from promptregress.cli import main
from promptregress.drift import cosine_similarity, measure_drift
from promptregress.gate import GateVerdict, evaluate_gate
from promptregress.history import (
    append_history,
    ascii_sparkline,
    format_history_table,
    load_history,
    make_compare_record,
    render_trend_html,
    suite_points,
)
from promptregress.junit_xml import failing_assertion, format_score, write_run_junit
from promptregress.loading import SuiteError, load_suite
from promptregress.matrix import MatrixError, load_matrix
from promptregress.models import (
    CaseResult,
    CostEstimate,
    PairedRow,
    PairwiseSummary,
    RunResult,
)
from promptregress.pairwise import pairwise_judge
from promptregress.power import (
    achieved_power,
    analytic_power,
    format_power,
    load_report_diffs,
    required_n,
    simulate_power,
)
from promptregress.pricing import PricingError, cost_delta, estimate_cost, load_pricing
from promptregress.providers import (
    Completion,
    MockProvider,
    ProviderError,
    TokenUsage,
    provider_model_name,
    usage_from_text_lengths,
)
from promptregress.runner import _invoke, _mean_judge_scores, _population_sd, run_suite
from promptregress.scorers import EXACT_KINDS


class _Vectors:
    name = "fixed"

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.vectors


def test_power_edges_and_report_loader(tmp_path: Path) -> None:
    infinite = required_n(math.inf, n_sim=20, seed=1)
    assert infinite.n == 1
    assert infinite.analytic_power == 1.0
    assert infinite.simulated_power == 1.0
    assert "inf" in format_power([infinite])
    assert analytic_power(0.0, 8) == pytest.approx(0.05)
    assert analytic_power(math.inf, 8) == 1.0
    assert achieved_power([1.0]) == 0.0
    assert achieved_power([0.2, 0.2]) == 1.0
    assert "required_n=undefined" in format_power([], achieved=0.0, observed_dz=0.0)
    with pytest.raises(ValueError, match="effect_dz"):
        required_n(0.0)
    with pytest.raises(ValueError, match="effect_dz"):
        analytic_power(True, 4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="effect_dz"):
        analytic_power(float("nan"), 4)
    with pytest.raises(ValueError, match="alpha"):
        required_n(0.5, alpha=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="power"):
        required_n(0.5, power=1.0)
    with pytest.raises(ValueError, match="n_sim"):
        simulate_power(0.5, 4, n_sim=0)
    with pytest.raises(ValueError, match="seed"):
        simulate_power(0.5, 4, n_sim=5, seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n must be"):
        analytic_power(0.5, 0)
    with pytest.raises(ValueError, match="diffs must be finite"):
        achieved_power([1.0, math.inf])

    broken = tmp_path / "nope.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_report_diffs(broken)
    broken.write_text(json.dumps({"per_case": "no"}), encoding="utf-8")
    with pytest.raises(ValueError, match="per_case"):
        load_report_diffs(broken)
    broken.write_text(json.dumps({"per_case": [{"score": 1}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="needs a diff"):
        load_report_diffs(broken)
    broken.write_text(json.dumps({"per_case": [{"diff": True}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a number"):
        load_report_diffs(broken)
    broken.write_text(json.dumps({"per_case": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no per-case"):
        load_report_diffs(broken)
    broken.write_text(json.dumps({"per_case": [{"diff": 0.25}]}), encoding="utf-8")
    assert load_report_diffs(broken) == [0.25]


def test_matrix_file_errors(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(MatrixError, match="not found"):
        load_matrix(missing)
    bad = tmp_path / "bad.yaml"
    bad.write_text(":\n  - [", encoding="utf-8")
    with pytest.raises(MatrixError, match="not valid YAML"):
        load_matrix(bad)
    cases = {
        "[]\n": "at least two",
        "- not-a-mapping\n- {name: b, provider: echo, template: '{input}'}\n": "must be a mapping",
        "- {provider: echo, template: '{input}'}\n- {name: b, provider: echo, template: x}\n": (
            "non-empty name"
        ),
        "echo: not-a-body\nother: {provider: echo, template: '{input}'}\n": "must be a mapping",
        "'': {provider: echo, template: '{input}'}\n": "non-empty strings",
        "just a string\n": "list or a mapping",
    }
    for text, message in cases.items():
        path = tmp_path / "matrix.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(MatrixError, match=message):
            load_matrix(path)
    path.write_text(
        "a: {name: other, provider: echo, template: '{input}'}\n"
        "b: {provider: echo, template: '{input}'}\n",
        encoding="utf-8",
    )
    with pytest.raises(MatrixError, match="does not match"):
        load_matrix(path)
    path.write_text(
        "- {name: a, provider: no, template: '{input}'}\n"
        "- {name: b, provider: echo, template: '{input}'}\n",
        encoding="utf-8",
    )
    with pytest.raises(MatrixError, match="provider must be"):
        load_matrix(path)
    path.write_text(
        "- {name: a, provider: echo}\n- {name: b, provider: echo, template: '{input}'}\n",
        encoding="utf-8",
    )
    with pytest.raises(MatrixError, match="string template"):
        load_matrix(path)
    path.write_text(
        "- {name: a, provider: echo, template: '{input}', model: 1}\n"
        "- {name: b, provider: echo, template: '{input}'}\n",
        encoding="utf-8",
    )
    with pytest.raises(MatrixError, match="model must be a string"):
        load_matrix(path)
    path.write_text(
        "- {name: a, provider: mock, template: '{input}', timeout: 0}\n"
        "- {name: b, provider: echo, template: '{input}'}\n",
        encoding="utf-8",
    )
    with pytest.raises(MatrixError, match="timeout"):
        load_matrix(path)
    path.write_text(
        "- {name: a, provider: mock, template: '{input}', mock_rules: 'no'}\n"
        "- {name: b, provider: echo, template: '{input}'}\n",
        encoding="utf-8",
    )
    with pytest.raises(MatrixError, match="mock_rules"):
        load_matrix(path)
    path.write_text(
        "- {name: a, provider: mock, template: '{input}', mock_rules: [[1, 'x']]}\n"
        "- {name: b, provider: echo, template: '{input}'}\n",
        encoding="utf-8",
    )
    with pytest.raises(MatrixError, match="pair of strings"):
        load_matrix(path)
    path.write_text(
        "a: {provider: echo, template: '{input}'}\na: {provider: echo, template: 'other'}\n",
        encoding="utf-8",
    )
    with pytest.raises(MatrixError, match="at least two"):
        load_matrix(path)
    path.write_text(
        "- {name: a, provider: echo, template: '{input}'}\n"
        "- {name: a, provider: echo, template: '{input}'}\n",
        encoding="utf-8",
    )
    with pytest.raises(MatrixError, match="Duplicate matrix config name"):
        load_matrix(path)


def test_history_rows_sparkline_and_trend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    comparison = replace(
        comparison_with(ci_low=0.0, ci_high=0.0, mean_diff=0.0, candidate_rows=[("a", 1.0, True)]),
        pairwise=PairwiseSummary("rubric", 1, 0, 0, 0.5, 0.0, 1.0, []),
    )
    record = make_compare_record(
        comparison,
        provider="mock",
        model="mock",
        baseline_template="{input}",
        candidate_template="{input}",
        gate_passed=True,
        timestamp="2026-11-12T00:00:00Z",
    )
    assert record["metrics"]["pairwise_win_rate"] == 0.5
    path = tmp_path / "history.jsonl"
    append_history(record, path)
    append_history({"note": "a\nb"}, path)
    loaded_rows, loaded_notes = load_history(path)
    assert loaded_notes == []
    assert loaded_rows[-1]["note"] == "a\nb"
    assert path.read_text(encoding="utf-8").count("\n") == 2

    def _multiline(_payload: object, ensure_ascii: bool = False) -> str:
        return "{\n}"

    monkeypatch.setattr("promptregress.history.json.dumps", _multiline)
    with pytest.raises(ValueError, match="one line"):
        append_history({"note": "x"}, path)
    path.write_text(
        "\n"
        "[]\n"
        '{"suite": "other", "metrics": {"mean_score": 1}}\n'
        '{"suite": "s", "metrics": []}\n'
        '{"suite": "s", "metrics": {"mean_score": true}}\n'
        '{"suite": "s", "metrics": {"mean_score": NaN}}\n'
        '{"suite": "s", "metrics": {"mean_score": 0.5, "pairwise_win_rate": 0.25}}\n'
        '{"suite": "s", "timestamp": 3, "metrics": {"mean_score": 1}}\n',
        encoding="utf-8",
    )
    rows, notes = load_history(path)
    assert any("not an object" in note for note in notes)
    points = suite_points(rows, "s")
    assert points == [("", 0.5), ("", 1.0)]
    table = format_history_table(
        [{"timestamp": "t", "command": "run", "suite": "s", "metrics": {"pairwise_win_rate": 0.5}}]
    )
    assert "pairwise_win_rate=0.500000" in table
    wrapped = format_history_table([{"timestamp": "a\nb", "command": None}])
    assert "a b" in wrapped
    assert "a\nb" not in wrapped
    assert ascii_sparkline([True, math.nan, 0.0, 1.0]) == "··▁█"  # type: ignore[list-item]
    html = render_trend_html("s", [("", 0.2), ("", 0.8)])
    assert "<polyline" in html
    assert render_trend_html("s", []).count("<circle") == 0
    assert "<circle" in render_trend_html("only", [("", 0.4)])


def test_drift_cosine_edges_and_empty_summary() -> None:
    with pytest.raises(ValueError, match="dimension"):
        cosine_similarity([1.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="non-empty"):
        cosine_similarity([], [])
    assert cosine_similarity([math.inf], [math.inf]) == 0.0
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 1.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    # Float error pushes these slightly outside [-1, 1]; the function clamps.
    assert cosine_similarity([0.5, 1.0], [0.5, 0.9999999999999999]) == 1.0
    assert cosine_similarity([1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]) == -1.0
    comparison = comparison_with(
        ci_low=0.0, ci_high=0.0, mean_diff=0.0, candidate_rows=[("a", 1.0, True)]
    )
    empty = replace(comparison, per_case=[])
    summary = measure_drift(empty, _Vectors([]), embedder_name="hash")
    assert summary.mean_drift == 0.0
    assert summary.max_drift == 0.0
    assert summary.per_case == []
    with pytest.raises(ValueError, match="embedder_name"):
        measure_drift(empty, _Vectors([]), embedder_name="  ")
    differed = replace(
        comparison,
        per_case=[PairedRow("a", "left", "right", 1.0, 0.0, -1.0, True, False, "loss")],
    )
    with pytest.raises(ValueError, match="different number of vectors"):
        measure_drift(differed, _Vectors([[1.0, 0.0]]), embedder_name="hash")


def test_pricing_rejections_and_note_delta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = run_with([("a", 1.0, True)], "c")
    run = replace(
        run,
        results=[replace(run.results[0], prompt_tokens=10, completion_tokens=0, total_tokens=10)],
    )
    with pytest.raises(ValueError, match="model"):
        estimate_cost(run, "", {})
    unknown = estimate_cost(run, "missing", {})
    assert unknown.usd is None
    assert unknown.note == "unknown model 'missing'"
    delta, note = cost_delta(
        CostEstimate("a", None, "unknown model 'a'"),
        CostEstimate("b", 1.0, "extra"),
    )
    assert delta is None
    assert note == "unknown model 'a'; extra"
    priced, clean = cost_delta(CostEstimate("a", 1.5), CostEstimate("b", 0.5))
    assert priced == pytest.approx(-1.0)
    assert clean is None
    text = tmp_path / "prices.txt"
    text.write_text("{}", encoding="utf-8")
    with pytest.raises(PricingError, match="json"):
        load_pricing(text)
    broken = tmp_path / "prices.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(PricingError, match="Malformed"):
        load_pricing(broken)
    yaml_path = tmp_path / "prices.yaml"
    yaml_path.write_text(":\n - [", encoding="utf-8")
    with pytest.raises(PricingError, match="Malformed"):
        load_pricing(yaml_path)
    yaml_path.write_text("- 1\n", encoding="utf-8")
    with pytest.raises(PricingError, match="mapping of model"):
        load_pricing(yaml_path)
    yaml_path.write_text("1: {prompt_per_1k: 0, completion_per_1k: 0}\n", encoding="utf-8")
    with pytest.raises(PricingError, match="non-empty strings"):
        load_pricing(yaml_path)
    yaml_path.write_text("mock: 1\n", encoding="utf-8")
    with pytest.raises(PricingError, match="must be a mapping"):
        load_pricing(yaml_path)
    nested = tmp_path / "deep.json"
    nested.write_text("{}", encoding="utf-8")

    def _too_deep(_text: str) -> object:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr("promptregress.pricing.json.loads", _too_deep)
    with pytest.raises(PricingError, match="too deeply nested"):
        load_pricing(nested)


def test_suite_loader_rejects_the_remaining_shapes(tmp_path: Path) -> None:
    def write(name: str, text: str) -> Path:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    with pytest.raises(SuiteError, match="description must be a string"):
        load_suite(write("d.json", json.dumps({"name": "s", "description": 1, "cases": []})))
    with pytest.raises(SuiteError, match="missing a header"):
        load_suite(write("empty.csv", ""))
    with pytest.raises(SuiteError, match="missing id"):
        load_suite(write("id.csv", "id,input,reference\n,hi,hi\n"))
    short = load_suite(write("short.csv", "id,input,reference\na,hello\n"))
    assert short.cases[0].reference == ""
    assert short.cases[0].input == "hello"
    body = {"name": "s", "cases": [{"id": "a", "input": "q"}]}
    with pytest.raises(SuiteError, match="scoring must be an object"):
        load_suite(write("sc.json", json.dumps({**body, "scoring": 1})))
    kept = load_suite(write("sc0.json", json.dumps({**body, "scoring": {"weights": None}})))
    assert kept.scoring.weights == {}
    with pytest.raises(SuiteError, match="weights must be an object"):
        load_suite(write("sc1.json", json.dumps({**body, "scoring": {"weights": []}})))
    shapes = [
        ({"cases": ["nope"]}, "not an object"),
        ({"cases": [{}]}, "missing id"),
        ({"cases": [{"id": "a"}]}, "missing input"),
        ({"cases": [{"id": "a", "input": 1}]}, "input must be a string"),
        ({"cases": [{"id": "a", "input": "q", "reference": 1}]}, "reference must be"),
        ({"cases": [{"id": "a", "input": "q", "assertions": "no"}]}, "assertions must be"),
        ({"cases": [{"id": "a", "input": "q", "assertions": ["no"]}]}, "not an object"),
        (
            {
                "cases": [
                    {
                        "id": "a",
                        "input": "q",
                        "assertions": [{"kind": "exact_match", "weight": True}],
                    }
                ]
            },
            "must be a number",
        ),
        (
            {
                "cases": [
                    {
                        "id": "a",
                        "input": "q",
                        "assertions": [{"kind": "exact_match", "params": []}],
                    }
                ]
            },
            "params must be",
        ),
        ({"cases": [{"id": "a", "input": "q", "tags": "no"}]}, "tags must be"),
        ({"cases": [{"id": "a", "input": "q", "metadata": []}]}, "metadata must be"),
    ]
    for index, (extra, message) in enumerate(shapes):
        document = {"name": "s", **extra}
        with pytest.raises(SuiteError, match=message):
            load_suite(write(f"bad-{index}.json", json.dumps(document)))
    none_ok = load_suite(
        write(
            "nones.json",
            json.dumps(
                {
                    "name": "s",
                    "description": None,
                    "cases": [
                        {
                            "id": "a",
                            "input": "q",
                            "assertions": None,
                            "tags": None,
                            "metadata": None,
                        }
                    ],
                }
            ),
        )
    )
    assert none_ok.description == ""
    assert none_ok.cases[0].assertions == []
    assert none_ok.cases[0].tags == []
    assert none_ok.cases[0].metadata == {}
    quoted = load_suite(write("quoted.yaml", "name: s\ncases:\n  - {id: 'plain', input: q}\n"))
    assert quoted.cases[0].id == "plain"
    assert quoted.cases[0].source_line == 3


def test_runner_provider_contract_and_judge_means() -> None:
    from promptregress.models import AssertionSpec, Case, Suite

    suite = Suite(
        name="s",
        cases=[
            Case(
                id="a",
                input="q",
                reference=None,
                assertions=[AssertionSpec("contains", "x")],
            )
        ],
    )

    class BadComplete:
        name = "bad"

        def complete(self, prompt: str) -> str:
            return 1  # type: ignore[return-value]

    with pytest.raises(ProviderError, match="non-string"):
        _invoke(BadComplete(), "q")  # type: ignore[arg-type]
    stored = run_suite(suite, BadComplete(), "c")  # type: ignore[arg-type]
    assert stored.results[0].passed is False
    assert stored.results[0].error is not None
    assert "non-string" in stored.results[0].error

    class BadDetailed:
        name = "bad"

        def complete(self, prompt: str) -> str:
            return "x"

        def complete_detailed(self, prompt: str) -> Completion:
            return "nope"  # type: ignore[return-value]

    with pytest.raises(ProviderError, match="complete_detailed must return"):
        _invoke(BadDetailed(), "q")  # type: ignore[arg-type]

    class BadText:
        name = "bad"

        def complete(self, prompt: str) -> str:
            return "x"

        def complete_detailed(self, prompt: str) -> Completion:
            return Completion(text=1, usage=TokenUsage(0, 0, 0))  # type: ignore[arg-type]

    with pytest.raises(ProviderError, match="non-string"):
        _invoke(BadText(), "q")  # type: ignore[arg-type]

    class BadAt:
        name = "bad"

        def complete(self, prompt: str) -> str:
            return "x"

        def complete_at(self, prompt: str, sample_index: int) -> Completion:
            return "nope"  # type: ignore[return-value]

    with pytest.raises(ProviderError, match="complete_at must return"):
        _invoke(BadAt(), "q", sample_index=0)  # type: ignore[arg-type]
    sampled = run_suite(suite, BadAt(), "c", samples=2)  # type: ignore[arg-type]
    assert sampled.results[0].n_samples == 2
    assert sampled.results[0].error is not None
    assert "complete_at must return" in sampled.results[0].error

    left = CaseResult(
        "a", "x", {"exact_match": 1.0}, 1.0, True, 1.0, judge_scores=(("j", 1.0), ("k", 0.0))
    )
    right = CaseResult(
        "a", "y", {"exact_match": 0.0}, 0.0, False, 1.0, judge_scores=(("j", 0.0), ("k", 1.0))
    )
    assert _mean_judge_scores([left, right]) == (("j", 0.5), ("k", 0.5))
    swapped = replace(right, judge_scores=(("k", 1.0), ("j", 0.0)))
    assert _mean_judge_scores([left, swapped]) == ()
    assert _population_sd([]) == 0.0


def test_small_model_gate_and_annotation_branches(tmp_path: Path) -> None:
    empty = RunResult("s", "c", [])
    assert empty.n_passed == 0
    assert empty.mean_within_case_sd == 0.0
    latencies = [float(index) for index in range(21)]
    run = RunResult(
        "s",
        "c",
        [
            CaseResult(str(index), "x", {}, 1.0, True, latency)
            for index, latency in enumerate(latencies)
        ],
    )
    assert run.p95_latency_ms == 19.0
    assert format_score(float("nan")) == "nan"
    assert format_score(float("inf")) == "inf"
    assert format_score(float("-inf")) == "-inf"
    graded = CaseResult("a", "x", {"levenshtein": 0.1}, 0.1, False, 1.0)
    assert "levenshtein" not in EXACT_KINDS
    assert failing_assertion(graded) == "levenshtein=0.100000"
    write_run_junit(
        tmp_path / "junit.xml",
        replace(empty, results=[replace(graded, latency_ms=math.nan)]),
    )
    text = (tmp_path / "junit.xml").read_text(encoding="utf-8")
    assert 'time="0.000000"' in text
    comparison = comparison_with(
        ci_low=0.0, ci_high=0.0, mean_diff=0.0, candidate_rows=[("a", 1.0, True)]
    )
    with pytest.raises(ValueError, match="max_mean_drift"):
        evaluate_gate(comparison, max_mean_drift=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_mean_drift"):
        evaluate_gate(comparison, max_mean_drift=float("nan"))
    assert annotation_lines([], "s.json", GateVerdict(False, ["not a failure"])) == []
    blank = evaluate_gate(
        replace(
            comparison,
            candidate=replace(
                comparison.candidate,
                cost=CostEstimate("m", None, ""),
            ),
        ),
        max_cost_usd=1.0,
    )
    assert any("unknown model" in reason for reason in blank.reasons)
    with pytest.raises(ValueError, match="non-negative"):
        TokenUsage(1, -1, 0)
    with pytest.raises(TypeError, match="strings"):
        usage_from_text_lengths(1, "x")  # type: ignore[arg-type]
    assert provider_model_name(type("P", (), {"name": "echo", "model": ""})()) == "echo"
    with pytest.raises(TypeError, match="vary"):
        MockProvider(vary=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="default"):
        MockProvider(default=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="latency_ms"):
        MockProvider(latency_ms=-1)
    assert first_differing_case_id(["a", "b"], ["a"]) == "b"


def test_baseline_list_skips_illegal_names_and_wraps_bad_json(tmp_path: Path) -> None:
    directory = tmp_path / ".promptregress" / "baselines"
    directory.mkdir(parents=True)
    (directory / "has space.json").write_text("{}", encoding="utf-8")
    (directory / "ok.json").write_text("{", encoding="utf-8")
    with pytest.raises(Exception, match="ok"):
        list_baselines(tmp_path)
    (directory / "ok.json").unlink()
    assert list_baselines(tmp_path) == []
    (directory / "gone.json").write_text("{", encoding="utf-8")
    with pytest.raises(Exception, match="invalid JSON|Expecting"):
        load_baseline("gone", tmp_path)


def test_pairwise_edges() -> None:
    comparison = comparison_with(
        ci_low=0.0, ci_high=0.0, mean_diff=0.0, candidate_rows=[("missing", 1.0, True)]
    )
    from promptregress.models import Case, Suite

    suite = Suite("s", [Case(id="other", input="q", reference=None, assertions=[])])
    missing = replace(
        comparison,
        per_case=[PairedRow("missing", "left", "right", 1.0, 1.0, 0.0, True, True, "tie")],
    )
    with pytest.raises(ValueError, match="rubric"):
        pairwise_judge(missing, suite, MockProvider(default="tie"), "  ")
    with pytest.raises(ValueError, match="no case"):
        pairwise_judge(missing, suite, MockProvider(default="tie"), "which is better")

    class Boom:
        name = "boom"

        def complete(self, prompt: str) -> str:
            raise RuntimeError("down")

    present = replace(
        comparison,
        per_case=[],
    )
    row = PairedRow("a", "left", "right", 1.0, 1.0, 0.0, True, True, "tie")
    paired = replace(comparison, per_case=[row])
    suite = Suite("s", [Case(id="a", input="q", reference=None, assertions=[])])
    failed = pairwise_judge(paired, suite, Boom(), "which")  # type: ignore[arg-type]
    assert failed.cases[0].score == 0.5
    assert failed.cases[0].error is not None
    assert "RuntimeError" in failed.cases[0].error

    class NotText:
        name = "nt"

        def complete(self, prompt: str) -> str:
            return 1  # type: ignore[return-value]

    weird = pairwise_judge(paired, suite, NotText(), "which")  # type: ignore[arg-type]
    assert weird.cases[0].error == (
        "judge provider returned a non-string; judge provider returned a non-string"
    )
    empty = pairwise_judge(present, suite, MockProvider(default="tie"), "which")
    assert empty.win_rate == 0.0
    assert empty.cases == []


def test_cli_budget_note_cache_and_power_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [
                    {
                        "id": "a",
                        "input": "hi",
                        "assertions": [{"kind": "contains", "value": "yes"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    prices = tmp_path / "prices.json"
    prices.write_text(
        json.dumps({"other": {"prompt_per_1k": 1, "completion_per_1k": 1}}),
        encoding="utf-8",
    )
    code = main(
        [
            "run",
            str(suite),
            "--provider",
            "mock",
            "--mock-default",
            "yes",
            "--config-name",
            "c",
            "--out",
            str(tmp_path / "run.json"),
            "--pricing",
            str(prices),
            "--cache",
            str(tmp_path / "cache"),
            "--no-history",
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "estimated_cost_usd=unknown" in text
    assert "unknown model 'mock'" in text
    code = main(
        [
            "compare",
            str(suite),
            "--provider",
            "echo",
            "--candidate-template",
            "{input}",
            "--no-history",
        ]
    )
    assert code == 2
    assert "baseline-template" in capsys.readouterr().err
    report = tmp_path / "cmp.json"
    report.write_text(
        json.dumps(
            {
                "per_case": [
                    {"case_id": "a", "diff": 0.0},
                    {"case_id": "b", "diff": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    code = main(["power", "--from-report", str(report), "--out", str(tmp_path / "power.json")])
    assert code == 0
    assert "required_n=undefined" in capsys.readouterr().out
    code = main(["power"])
    assert code == 2
    assert "requires --effect" in capsys.readouterr().err
    code = main(["flaky", "", "--suite", ""])
    assert code == 2
    assert "requires a suite name" in capsys.readouterr().err
