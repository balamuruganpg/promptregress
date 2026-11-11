"""Offline branches in scorers, reports, and the IRT loader.

These tests stay off the network. They call the public functions and check
the value each branch returns, not only that the line ran.
"""

import json
import math
from pathlib import Path

import pytest

from helpers import case_result
from promptregress.gate import evaluate_gate
from promptregress.irt import (
    _sigmoid,
    fit_2pl,
    fit_runs,
    format_irt,
    load_run_report,
    response_matrix,
    run_from_dict,
)
from promptregress.models import (
    AssertionSpec,
    Case,
    CaseResult,
    Comparison,
    CostEstimate,
    DriftSummary,
    JudgeAgreement,
    PairedRow,
    RunResult,
)
from promptregress.report import comparison_to_dict, format_text, render_html, render_markdown
from promptregress.scorers import (
    SCORERS,
    ScoreError,
    ScoringContext,
    ensemble_judge_scores,
    register_scorer,
    score_case,
)


def _case(
    kind: str,
    output: str,
    *,
    value: object = None,
    params: dict[str, object] | None = None,
    reference: str | None = "ref",
    weight: float = 1.0,
) -> tuple[Case, str]:
    case = Case(
        id="c",
        input="in",
        reference=reference,
        assertions=[
            AssertionSpec(kind, value=value, weight=weight, params={} if params is None else params)
        ],
    )
    return case, output


def test_score_outcome_repr_and_zero_assertion_weight() -> None:
    case, output = _case("exact_match", "hello", reference="Hello", weight=0)
    outcome = score_case(case, output)
    assert outcome[0]["exact_match"] == 1.0
    assert outcome[1] == 0.0
    assert outcome[2] is True
    assert "ScoreOutcome" in repr(outcome)
    assert outcome.notes == ()


def test_scorer_weight_and_return_checks() -> None:
    case, output = _case("exact_match", "ref")
    with pytest.raises(ScoreError, match="must be a non-negative finite number"):
        score_case(case, output, weights={"exact_match": True})
    with pytest.raises(ScoreError, match="non-negative and finite"):
        score_case(case, output, weights={"exact_match": float("nan")})
    with pytest.raises(ScoreError, match="non-negative and finite"):
        score_case(case, output, weights={"exact_match": -0.1})

    unknown, text = _case("missing_kind", "x")
    with pytest.raises(ScoreError, match="Unknown scorer"):
        score_case(unknown, text)

    nan_weight, text = _case("exact_match", "ref", weight=float("nan"))
    with pytest.raises(ScoreError, match="assertion weight"):
        score_case(nan_weight, text)

    def non_numeric(output: str, case: Case, assertion: AssertionSpec) -> str:
        del output, case, assertion
        return "nope"

    def non_finite(output: str, case: Case, assertion: AssertionSpec) -> float:
        del output, case, assertion
        return float("nan")

    register_scorer("cov_non_numeric")(non_numeric)
    register_scorer("cov_non_finite")(non_finite)
    try:
        bad, text = _case("cov_non_numeric", "x")
        with pytest.raises(ScoreError, match="non-numeric"):
            score_case(bad, text)
        worse, text = _case("cov_non_finite", "x")
        with pytest.raises(ScoreError, match="non-finite"):
            score_case(worse, text)
    finally:
        SCORERS.pop("cov_non_numeric", None)
        SCORERS.pop("cov_non_finite", None)


def test_text_scorer_option_and_target_errors() -> None:
    sensitive, text = _case("exact_match", "A", params={"case_sensitive": "yes"})
    with pytest.raises(ScoreError, match="case_sensitive"):
        score_case(sensitive, text)
    flags, text = _case("regex", "A", value="a", params={"ignore_case": "yes"})
    with pytest.raises(ScoreError, match="ignore_case"):
        score_case(flags, text)
    empty, text = _case("regex", "A", value="")
    with pytest.raises(ScoreError, match="non-empty"):
        score_case(empty, text)
    invalid, text = _case("regex", "A", value="[")
    with pytest.raises(ScoreError, match="Invalid regex"):
        score_case(invalid, text)
    missing, text = _case("contains", "hello", value=None, reference=None)
    assert score_case(missing, text)[1] == 0.0
    edit, text = _case("levenshtein", "hello", value=None, reference=None)
    assert score_case(edit, text)[1] == 0.0


def test_json_schema_rejects_malformed_keywords() -> None:
    schema_case, text = _case("json_schema", "{}", value="not-a-schema")
    with pytest.raises(ScoreError, match="schema object"):
        score_case(schema_case, text)

    def scored(schema: object, document: str) -> float:
        case, output = _case("json_schema", document, value=schema)
        return score_case(case, output)[1]

    assert scored({"type": "boolean"}, "true") == 1.0
    assert scored({"type": "boolean"}, "1") == 0.0
    assert scored({"type": "date"}, '"x"') == 0.0
    assert scored({"type": 1}, '"x"') == 0.0
    assert scored({"type": ["string", "null"]}, "null") == 1.0
    assert scored({"properties": {"a": "nope"}}, '{"a": 1}') == 0.0
    assert scored({"properties": []}, "{}") == 0.0
    assert scored({"enum": "a"}, '"a"') == 0.0
    assert scored({"required": "id"}, "{}") == 0.0
    assert scored({"required": ["id"]}, '"x"') == 0.0
    assert scored({"required": [1]}, "{}") == 0.0
    assert scored({"items": ["nope"]}, "[1]") == 0.0
    assert scored({"items": {"type": "string"}}, '"x"') == 0.0
    assert scored({"items": {"type": "integer"}}, "[1, 2]") == 1.0


def test_numeric_and_tfidf_input_checks() -> None:
    for value in (True, None, "nope", {"n": 1}, float("inf")):
        case, output = _case("numeric_tolerance", "1", value=value)
        with pytest.raises(ScoreError, match="numeric_tolerance value"):
            score_case(case, output)
    for tol in (True, -1, float("nan")):
        case, output = _case("numeric_tolerance", "1", value=1, params={"tol": tol})
        with pytest.raises(ScoreError, match="tol"):
            score_case(case, output)
    huge, output = _case("numeric_tolerance", "1e9999", value=0)
    assert score_case(huge, output)[1] == 0.0

    for params in ({"mode": 1}, {"mode": "byte"}, {"ngram": True}, {"ngram": 0}):
        case, output = _case("tfidf_cosine", "ab", value="ab", params=params)
        with pytest.raises(ScoreError, match="tfidf_cosine"):
            score_case(case, output)
    missing, output = _case("tfidf_cosine", "hello", value=None, reference=None)
    assert score_case(missing, output)[1] == 0.0
    both_empty, output = _case("tfidf_cosine", "", value=None, reference=None)
    assert score_case(both_empty, output)[1] == 1.0
    chars, output = _case("tfidf_cosine", "ab", value="ab", params={"mode": "char", "ngram": 3})
    assert score_case(chars, output)[1] == 1.0


def test_judge_parse_failures_and_non_string_provider() -> None:
    class _Odd:
        name = "odd"
        model = "odd"

        def complete(self, prompt: str) -> object:
            del prompt
            return 1

    case, output = _case("llm_judge", "out", params={"rubric": "grade it"})
    odd = score_case(case, output, ScoringContext(judge=_Odd()))  # type: ignore[arg-type]
    assert odd[1] == 0.0
    assert "non-string" in odd.notes[0]

    long_reply = "x" * 400

    class _Prose:
        name = "prose"
        model = "prose"

        def complete(self, prompt: str) -> str:
            del prompt
            return long_reply

    prose = score_case(case, output, ScoringContext(judge=_Prose()))
    assert prose[1] == 0.0
    assert "..." in prose.notes[0]
    assert long_reply not in prose.notes[0]

    def judged(reply: str) -> tuple[float, str]:
        class _Fixed:
            name = "fixed"
            model = "fixed"

            def complete(self, prompt: str) -> str:
                del prompt
                return reply

        scored = score_case(case, output, ScoringContext(judge=_Fixed()))
        return scored[1], scored.notes[0] if scored.notes else ""

    assert judged('{"score": true}')[0] == 0.0
    assert "not a number" in judged('{"score": true}')[1]
    assert judged('{"score": "nope"}')[0] == 0.0
    assert judged('{"score": {"nested": 1}}')[0] == 0.0
    assert judged('{"score": Infinity}')[0] == 0.0
    score, note = judged('prefix {not json {"score": 0.25}')
    assert score == 0.25
    assert note == ""


def test_ensemble_config_errors_and_score_filter() -> None:
    def ensemble(judges: object, rubric: str | None = "shared") -> tuple[float, tuple[str, ...]]:
        params: dict[str, object] = {"judges": judges}
        if rubric is not None:
            params["rubric"] = rubric
        case, output = _case("llm_judge", "MARKER", params=params, reference=None)
        scored = score_case(case, output)
        return scored[1], scored.notes

    assert ensemble(["nope"])[0] == 0.0
    assert "must be an object" in ensemble(["nope"])[1][0]
    assert "provider" in ensemble([{"provider": "nope", "rubric": "r"}])[1][0]
    assert "model" in ensemble([{"provider": "mock", "model": 1, "rubric": "r"}])[1][0]
    assert "rubric" in ensemble([{"provider": "mock"}], rubric=None)[1][0]
    assert (
        "mock_default" in ensemble([{"provider": "mock", "rubric": "r", "mock_default": 1}])[1][0]
    )
    assert (
        "mock_rules" in ensemble([{"provider": "mock", "rubric": "r", "mock_rules": "nope"}])[1][0]
    )
    assert (
        "response"
        in ensemble([{"provider": "mock", "rubric": "r", "mock_rules": [["n", 1]]}])[1][0]
    )
    listed = ensemble(
        [
            {
                "provider": "mock",
                "model": "same",
                "rubric": "r",
                "mock_default": "1",
                "mock_rules": [["MARKER", ["1", "0"]]],
            },
            {"provider": "mock", "model": "same", "rubric": "r", "mock_default": "0"},
        ]
    )
    assert listed[0] == pytest.approx(0.5)
    assert (
        "timeout"
        in ensemble([{"provider": "openai", "rubric": "r", "api_key": "k", "timeout": True}])[1][0]
    )
    assert (
        "timeout"
        in ensemble([{"provider": "openai", "rubric": "r", "api_key": "k", "timeout": 0}])[1][0]
    )
    assert (
        "timeout"
        in ensemble(
            [{"provider": "openai", "rubric": "r", "api_key": "k", "timeout": float("nan")}]
        )[1][0]
    )
    assert "api_key" in ensemble([{"provider": "openai", "rubric": "r", "api_key": 1}])[1][0]
    assert (
        "base_url"
        in ensemble([{"provider": "openai", "rubric": "r", "api_key": "k", "base_url": 1}])[1][0]
    )
    echoed = ensemble([{"provider": "echo", "model": "e", "rubric": "r"}])
    assert echoed[0] == 0.0

    found = ensemble_judge_scores(
        [
            {"kind": "llm_judge"},
            {"kind": "llm_judge_ensemble", "scores": "nope"},
            {
                "kind": "llm_judge_ensemble",
                "scores": [
                    "x",
                    {"label": 1, "score": 0.2},
                    {"label": "bad", "score": True},
                    {"label": "ok", "score": 0.4},
                ],
            },
        ]
    )
    assert found == (("ok", 0.4),)


def test_openai_judge_config_posts_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Response:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def read(self) -> bytes:
            return self.payload

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    def fake_urlopen(request: object, timeout: float | None = None) -> _Response:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode())  # type: ignore[attr-defined]
        return _Response(b'{"choices":[{"message":{"content":"0.5"}}]}')

    monkeypatch.setattr("promptregress.providers.urllib.request.urlopen", fake_urlopen)
    case, output = _case(
        "llm_judge",
        "answer",
        params={
            "judges": [
                {
                    "provider": "openai",
                    "model": "demo",
                    "rubric": "grade",
                    "api_key": "sek",
                    "base_url": "https://example.test/v1",
                    "timeout": 5,
                }
            ]
        },
    )
    scored = score_case(case, output)
    assert scored[1] == 0.5
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["timeout"] == 5
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "demo"


def test_trace_match_rejects_bad_steps_and_skips_bad_actuals() -> None:
    def trace(value: object, output: str) -> float:
        case, text = _case("trace_match", output, value=value, reference=None)
        return score_case(case, text)[1]

    with pytest.raises(ScoreError, match="list of steps"):
        trace("nope", "[]")
    with pytest.raises(ScoreError, match="must be an object"):
        trace(["nope"], "[]")
    with pytest.raises(ScoreError, match="tool must be a string"):
        trace([{"tool": 1}], "[]")
    with pytest.raises(ScoreError, match="args must be an object"):
        trace([{"tool": "t", "args": []}], "[]")
    with pytest.raises(ScoreError, match="arg keys"):
        trace([{"tool": "t", "args": {1: "a"}}], "[]")
    with pytest.raises(ScoreError, match="not JSON text"):
        trace([{"tool": "t", "args": {"n": float("nan")}}], "[]")
    with pytest.raises(ScoreError, match="not JSON text"):
        trace([{"tool": "t", "args": {"n": object()}}], "[]")

    assert trace([{"tool": "t", "args": None}], '[{"tool": "t"}]') == 1.0
    actual = json.dumps(
        [
            "skip",
            {"tool": 1},
            {"tool": "t", "args": []},
            {"tool": "t", "args": None},
            {"nope": True},
        ]
    )
    assert trace([{"tool": "t"}], actual) == 1.0
    assert trace([{"tool": "t", "args": {"q": "hi"}}], "not json") == 0.0
    assert trace([], "not json") == 1.0


def _paired_row() -> PairedRow:
    return PairedRow(
        case_id="a|b",
        baseline_output="word " * 80,
        candidate_output="",
        baseline_score=1.0,
        candidate_score=0.0,
        diff=-1.0,
        baseline_passed=True,
        candidate_passed=False,
        outcome="loss",
    )


def _sparse_comparison() -> Comparison:
    baseline_row = case_result("a", 1.0, True, output="baseline text")
    baseline = RunResult(
        suite_name="suite",
        config_name="base <b>",
        results=[
            CaseResult(
                case_id=baseline_row.case_id,
                output=baseline_row.output,
                scores=baseline_row.scores,
                score=baseline_row.score,
                passed=True,
                latency_ms=1.0,
                sample_scores=[1.0, 0.0],
                sample_outputs=["baseline text", "other"],
                score_sd=0.5,
                n_samples=2,
            )
        ],
        cost=CostEstimate(model="gpt-demo", usd=None, note="model not in the pricing file"),
    )
    candidate = RunResult(
        suite_name="suite",
        config_name="candidate",
        results=[],
        cost=None,
    )
    return Comparison(
        baseline=baseline,
        candidate=candidate,
        mean_diff=0.0,
        ci_low=0.0,
        ci_high=0.0,
        mcnemar_p=1.0,
        wins=0,
        losses=1,
        ties=0,
        per_case=[_paired_row()],
        wilcoxon_stat=float("inf"),
        ttest_stat=float("-inf"),
        cohens_dz=float("nan"),
        drift=DriftSummary(
            embedder="hash",
            mean_drift=0.0,
            max_drift=0.0,
            per_case=[],
            top=[],
        ),
        judge_agreement=(
            JudgeAgreement(
                side="candidate",
                config_name="candidate",
                n_cases=1,
                n_judges=2,
                mean_pairwise_abs=0.5,
                fleiss_kappa=None,
                pass_threshold=0.5,
                labels=("mock/a", "mock/b"),
                per_judge_mean=(1.0, 0.0),
                notes=("dropped case z: judge labels differ",),
            ),
        ),
    )


def test_reports_cover_sparse_sections() -> None:
    comparison = _sparse_comparison()
    verdict = evaluate_gate(comparison, max_regression=1.0)
    text = format_text(comparison, verdict, failure_clusters=[])
    markdown = render_markdown(comparison, verdict, failure_clusters=[])
    html = render_html(comparison, verdict, failure_clusters=[])
    payload = comparison_to_dict(comparison, verdict, failure_clusters=[])

    assert "cohens_dz=nan" in text
    assert "ttest_p=" in text
    assert "stability:" in text
    assert "base <b>_least_stable:" in text
    assert "- a 0.500000" in text
    assert "candidate_least_stable:" in text
    assert "- none" in text
    assert "cost_note=" in text
    assert "unknown" in text
    assert "top_drift:" in text
    assert "failure_modes:" in text
    assert "fleiss_kappa=undefined" in text
    assert "dropped case z" in text

    assert "nan" in markdown
    assert "-inf" in markdown
    assert "## Stability" in markdown
    assert "| none |  |" in markdown
    assert "No score regressions" in markdown
    assert "| none |  |" in markdown
    assert "Estimated cost" in markdown
    assert "unknown" in markdown
    assert "model not in the pricing file" in markdown
    assert "Semantic drift" in markdown
    assert "undefined" in markdown
    assert "dropped case z" in markdown
    assert ">" in markdown

    assert "Stability" in html
    assert "none" in html
    assert "unknown" in html
    assert "..." in html
    assert "&lt;b&gt;" in html
    assert "undefined" in html
    assert "dropped case z" in html
    assert "No score regressions" in html

    assert payload["wilcoxon_stat"] is None
    assert payload["ttest_stat"] is None
    assert payload["cohens_dz"] is None
    assert math.isnan(comparison.cohens_dz)
    assert payload["cost"]["delta_usd"] is None
    assert payload["cost"]["note"] == "cost was estimated for only one run"


def test_irt_loader_and_fit_reject_bad_inputs(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_run_report(broken)
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_run_report(array)

    with pytest.raises(ValueError, match="config_name and results"):
        run_from_dict({"config_name": "c"}, label="probe")
    with pytest.raises(ValueError, match="results must be a list"):
        run_from_dict({"config_name": "c", "results": {}}, label="probe")
    with pytest.raises(ValueError, match="must be strings"):
        run_from_dict({"config_name": 1, "suite_name": "s", "results": []}, label="probe")
    with pytest.raises(ValueError, match="must be an object"):
        run_from_dict({"config_name": "c", "results": ["x"]}, label="probe")
    with pytest.raises(ValueError, match="missing score"):
        run_from_dict({"config_name": "c", "results": [{"case_id": "a"}]}, label="probe")
    with pytest.raises(ValueError, match="case_id must be a string"):
        run_from_dict(
            {"config_name": "c", "results": [{"case_id": 1, "score": 1, "passed": True}]},
            label="probe",
        )
    with pytest.raises(ValueError, match="passed must be a boolean"):
        run_from_dict(
            {"config_name": "c", "results": [{"case_id": "a", "score": 1, "passed": 1}]},
            label="probe",
        )
    with pytest.raises(ValueError, match="output must be a string"):
        run_from_dict(
            {
                "config_name": "c",
                "results": [{"case_id": "a", "score": 1, "passed": True, "output": 1}],
            },
            label="probe",
        )
    with pytest.raises(ValueError, match="scores must be an object"):
        run_from_dict(
            {
                "config_name": "c",
                "results": [{"case_id": "a", "score": 1, "passed": True, "scores": []}],
            },
            label="probe",
        )
    with pytest.raises(ValueError, match="scores must map strings"):
        run_from_dict(
            {
                "config_name": "c",
                "results": [{"case_id": "a", "score": 1, "passed": True, "scores": {1: 0.5}}],
            },
            label="probe",
        )
    with pytest.raises(ValueError, match="score must be a finite number"):
        run_from_dict(
            {
                "config_name": "c",
                "results": [{"case_id": "a", "score": True, "passed": True}],
            },
            label="probe",
        )
    with pytest.raises(ValueError, match="error must be a string"):
        run_from_dict(
            {
                "config_name": "c",
                "results": [{"case_id": "a", "score": 1, "passed": True, "error": 1}],
            },
            label="probe",
        )
    with pytest.raises(ValueError, match="tags must be a list"):
        run_from_dict(
            {
                "config_name": "c",
                "results": [{"case_id": "a", "score": 1, "passed": True, "tags": "a"}],
            },
            label="probe",
        )
    with pytest.raises(ValueError, match="prompt_tokens"):
        run_from_dict(
            {
                "config_name": "c",
                "results": [{"case_id": "a", "score": 1, "passed": True, "prompt_tokens": -1}],
            },
            label="probe",
        )
    with pytest.raises(ValueError, match="cached must be a boolean"):
        run_from_dict(
            {
                "config_name": "c",
                "results": [{"case_id": "a", "score": 1, "passed": True, "cached": "yes"}],
            },
            label="probe",
        )
    with pytest.raises(ValueError, match="sample_scores must be a list"):
        run_from_dict(
            {
                "config_name": "c",
                "results": [{"case_id": "a", "score": 1, "passed": True, "sample_scores": "no"}],
            },
            label="probe",
        )
    with pytest.raises(ValueError, match="sample_outputs must be a list"):
        run_from_dict(
            {
                "config_name": "c",
                "results": [{"case_id": "a", "score": 1, "passed": True, "sample_outputs": [1]}],
            },
            label="probe",
        )
    with pytest.raises(ValueError, match="judge_scores must be a list"):
        run_from_dict(
            {
                "config_name": "c",
                "results": [{"case_id": "a", "score": 1, "passed": True, "judge_scores": "no"}],
            },
            label="probe",
        )
    with pytest.raises(ValueError, match="judge_scores entries"):
        run_from_dict(
            {
                "config_name": "c",
                "results": [
                    {"case_id": "a", "score": 1, "passed": True, "judge_scores": [["only"]]}
                ],
            },
            label="probe",
        )
    with pytest.raises(ValueError, match="n_samples must be an integer"):
        run_from_dict(
            {
                "config_name": "c",
                "results": [{"case_id": "a", "score": 1, "passed": True, "n_samples": True}],
            },
            label="probe",
        )
    with pytest.raises(ValueError, match="sample_outputs length"):
        run_from_dict(
            {
                "config_name": "c",
                "results": [
                    {
                        "case_id": "a",
                        "score": 1,
                        "passed": True,
                        "n_samples": 2,
                        "sample_scores": [1.0, 0.0],
                        "sample_outputs": ["only"],
                    }
                ],
            },
            label="probe",
        )
    with pytest.raises(ValueError, match="sample_scores length 2"):
        run_from_dict(
            {
                "config_name": "c",
                "results": [
                    {"case_id": "a", "score": 1, "passed": True, "sample_scores": [1.0, 0.0]}
                ],
            },
            label="probe",
        )
    with pytest.raises(ValueError, match="sample_outputs length 2"):
        run_from_dict(
            {
                "config_name": "c",
                "results": [
                    {"case_id": "a", "score": 1, "passed": True, "sample_outputs": ["x", "y"]}
                ],
            },
            label="probe",
        )
    loaded = run_from_dict(
        {
            "config_name": "c",
            "suite_name": "s",
            "results": [{"case_id": "a", "score": 1, "passed": True}],
            "cost": {"model": "m", "usd": None, "note": None},
        },
        label="probe",
    )
    assert loaded.cost is not None
    assert loaded.cost.usd is None
    with pytest.raises(ValueError, match="cost must be an object"):
        run_from_dict(
            {"config_name": "c", "results": [], "cost": []},
            label="probe",
        )
    with pytest.raises(ValueError, match="cost.model"):
        run_from_dict(
            {"config_name": "c", "results": [], "cost": {"model": 1}},
            label="probe",
        )
    with pytest.raises(ValueError, match="cost.note"):
        run_from_dict(
            {"config_name": "c", "results": [], "cost": {"model": "m", "note": 1}},
            label="probe",
        )

    one = RunResult(
        suite_name="s",
        config_name="m",
        results=[case_result("a", 1.0, True)],
    )
    with pytest.raises(ValueError, match="response must be"):
        response_matrix([one, one], response="nope")
    with pytest.raises(ValueError, match="at least two"):
        response_matrix([one])
    empty = RunResult(suite_name="s", config_name="m", results=[])
    with pytest.raises(ValueError, match="at least one case"):
        response_matrix([empty, empty])
    duplicate = RunResult(
        suite_name="s",
        config_name="m",
        results=[case_result("a", 1.0, True), case_result("a", 0.0, False)],
    )
    with pytest.raises(ValueError, match="Duplicate case id"):
        response_matrix([duplicate, one])
    other = RunResult(
        suite_name="s",
        config_name="n",
        results=[case_result("b", 1.0, True)],
    )
    with pytest.raises(ValueError, match="do not match"):
        response_matrix([one, other])
    non_finite = RunResult(
        suite_name="s",
        config_name="m",
        results=[
            CaseResult(
                case_id="a",
                output="o",
                scores={},
                score=float("nan"),
                passed=True,
                latency_ms=0.0,
            )
        ],
    )
    with pytest.raises(ValueError, match="Non-finite"):
        response_matrix([one, non_finite])

    wide = RunResult(
        suite_name="s",
        config_name="wide",
        results=[
            CaseResult(
                case_id="a",
                output="o",
                scores={},
                score=1.5,
                passed=True,
                latency_ms=0.0,
            )
        ],
    )
    low = RunResult(
        suite_name="s",
        config_name="wide",
        results=[
            CaseResult(
                case_id="a",
                output="o",
                scores={},
                score=-0.2,
                passed=False,
                latency_ms=0.0,
            )
        ],
    )
    names, case_ids, matrix, notes = response_matrix([wide, low], response="passed")
    assert names == ["wide", "wide#2"]
    assert case_ids == ["a"]
    assert matrix == [
        [
            1.0,
        ],
        [0.0],
    ]
    assert notes == []
    clipped = fit_runs([wide, low], iterations=1)
    assert clipped.details
    assert any("Clipped" in note for note in clipped.details)
    plain = fit_runs([one, other_same(one)], iterations=1)
    assert plain.details == [] or all("Clipped" not in note for note in plain.details)

    with pytest.raises(ValueError, match="one row per model"):
        fit_2pl([[1.0]], case_ids=["a"], model_names=["m", "n"], iterations=1)
    with pytest.raises(ValueError, match="at least one case"):
        fit_2pl([], case_ids=[], model_names=[], iterations=1)
    with pytest.raises(ValueError, match="unique"):
        fit_2pl([[1.0, 0.0]], case_ids=["a", "a"], model_names=["m"], iterations=1)
    with pytest.raises(ValueError, match="at least one model"):
        fit_2pl([], case_ids=["a"], model_names=[], iterations=1)
    with pytest.raises(ValueError, match="one entry per case"):
        fit_2pl([[1.0]], case_ids=["a", "b"], model_names=["m"], iterations=1)
    with pytest.raises(ValueError, match="finite numbers"):
        fit_2pl([[True]], case_ids=["a"], model_names=["m"], iterations=1)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="finite numbers"):
        fit_2pl([[1.5]], case_ids=["a"], model_names=["m"], iterations=1)
    with pytest.raises(ValueError, match="iterations"):
        fit_2pl([[1.0], [0.0]], case_ids=["a"], model_names=["m", "n"], iterations=0)
    with pytest.raises(ValueError, match="easy_threshold"):
        fit_2pl(
            [[1.0], [0.0]],
            case_ids=["a"],
            model_names=["m", "n"],
            iterations=1,
            easy_threshold=2.0,
            hard_threshold=1.0,
        )
    with pytest.raises(ValueError, match="seed"):
        fit_2pl([[1.0], [0.0]], case_ids=["a"], model_names=["m", "n"], seed=True, iterations=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="learning_rate"):
        fit_2pl(
            [[1.0], [0.0]],
            case_ids=["a"],
            model_names=["m", "n"],
            iterations=1,
            learning_rate=float("nan"),
        )
    with pytest.raises(ValueError, match="positive"):
        fit_2pl(
            [[1.0], [0.0]],
            case_ids=["a"],
            model_names=["m", "n"],
            iterations=1,
            learning_rate=0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        fit_2pl(
            [[1.0], [0.0]],
            case_ids=["a"],
            model_names=["m", "n"],
            iterations=1,
            prior_l2=-0.1,
        )

    hard = fit_2pl(
        [[1.0, 0.0], [0.0, 1.0]],
        case_ids=["easy", "hard"],
        model_names=["m", "n"],
        iterations=1,
        min_discrimination=10,
        hard_threshold=-5,
        easy_threshold=-6,
    )
    hard_text = format_irt(hard)
    assert "low_discrimination" in hard_text
    assert "very_hard" in hard_text
    easy = fit_2pl(
        [[1.0, 0.0], [0.0, 1.0]],
        case_ids=["easy", "hard"],
        model_names=["m", "n"],
        iterations=1,
        min_discrimination=0.01,
        hard_threshold=6,
        easy_threshold=5,
    )
    assert "very_easy" in format_irt(easy)
    assert _sigmoid(100.0) == 1.0
    assert _sigmoid(-100.0) == 0.0
    assert 0.0 < _sigmoid(0.0) < 1.0


def other_same(run: RunResult) -> RunResult:
    """A second run with the same cases and a different config name."""
    return RunResult(
        suite_name=run.suite_name,
        config_name=run.config_name + "-2",
        results=list(run.results),
    )
