"""Suite lint, and the IRT suite audit of saved comparisons."""

import importlib.metadata
import json
import math
from pathlib import Path

import pytest

from promptregress.audit import (
    AUDIT_SEED,
    MAX_REDUNDANT_DISAGREEMENTS,
    MIN_CASES,
    MIN_CONFIGS,
    NEAR_ZERO_DISCRIMINATION,
    REFERENCE_KINDS,
    STRONG_NEGATIVE_POINT_BISERIAL,
    SUITE_AUDIT_LIMIT,
    SUITE_AUDIT_NOTE,
    audit_comparison_files,
    audit_runs,
    format_suite_audit,
    is_comparison_report,
    suite_audit_to_dict,
)
from promptregress.cli import main
from promptregress.models import AssertionSpec, Case, CaseResult, RunResult
from promptregress.report import write_json
from promptregress.scorers import EXACT_KINDS, SCORERS
from promptregress.stats import compare_runs


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_reference_kinds_are_the_hardcoded_target_scorers() -> None:
    # SCORERS / EXACT_KINDS do not record "needs a reference".
    assert REFERENCE_KINDS == frozenset({"exact_match", "contains", "levenshtein", "tfidf_cosine"})
    assert "tfidf_cosine" in SCORERS
    assert "exact_match" in EXACT_KINDS


def test_audit_reports_each_error_without_crashing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    problems = tmp_path / "problems.yaml"
    problems.write_text(
        "\n".join(
            [
                "name: problems",
                "cases:",
                "  - id: bare",
                "    input: hello",
                "  - id: weird",
                "    input: other text",
                "    assertions:",
                "      - kind: not_a_scorer",
                "",
            ]
        ),
        encoding="utf-8",
    )
    dups = _write(
        tmp_path / "dups.json",
        {
            "name": "dups",
            "cases": [
                {"id": "a", "input": "one", "assertions": [{"kind": "json_valid"}]},
                {"id": "a", "input": "two", "assertions": [{"kind": "json_valid"}]},
            ],
        },
    )
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    missing = tmp_path / "missing.json"

    code = main(["audit", str(problems), str(dups), str(bad), str(missing)])
    assert code == 1
    text = capsys.readouterr().out
    assert text.count(f"file: {problems}") == 1
    assert text.count(f"file: {dups}") == 1
    assert (
        "case: bare  severity: error  message: case has no assertions; "
        "score is 0 and passed is false"
    ) in text
    assert "case: weird  severity: error  message: unknown scorer kind 'not_a_scorer'." in text
    assert "case: a  severity: error  message: duplicate case id 'a'" in text
    assert "Malformed JSON" in text
    assert "Suite file not found" in text
    assert "severity: warning" not in text
    assert text.endswith("errors=5 warnings=0\n")


def test_audit_json_shape(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    problems = tmp_path / "problems.yaml"
    problems.write_text(
        "name: problems\ncases:\n  - id: bare\n    input: hello\n",
        encoding="utf-8",
    )
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    code = main(["audit", str(problems), str(bad), "--json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"findings", "errors", "warnings"}
    assert isinstance(payload["findings"], list)
    assert payload["findings"]
    for finding in payload["findings"]:
        assert set(finding) == {"file", "case", "severity", "check", "message"}
        assert finding["severity"] in {"error", "warning"}
        assert all(isinstance(finding[key], str) for key in ("file", "case", "check", "message"))
    assert payload["errors"] == sum(
        1 for finding in payload["findings"] if finding["severity"] == "error"
    )
    assert payload["warnings"] == sum(
        1 for finding in payload["findings"] if finding["severity"] == "warning"
    )
    checks = {finding["check"] for finding in payload["findings"]}
    assert checks == {"no_assertions", "load"}
    bare = next(finding for finding in payload["findings"] if finding["check"] == "no_assertions")
    assert bare["case"] == "bare"
    assert bare["severity"] == "error"
    load = next(finding for finding in payload["findings"] if finding["check"] == "load")
    assert load["case"] == ""
    assert "Malformed JSON" in load["message"]


def test_audit_warnings_do_not_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(
        tmp_path / "warn.json",
        {
            "name": "warn",
            "cases": [
                {
                    "id": "copy-a",
                    "input": "same question",
                    "assertions": [{"kind": "json_valid"}],
                },
                {
                    "id": "copy-b",
                    "input": "same question",
                    "assertions": [{"kind": "json_valid"}],
                },
                {
                    "id": "blank-ref",
                    "input": "question-blank",
                    "reference": "   ",
                    "assertions": [
                        {"kind": "contains"},
                        {"kind": "levenshtein"},
                        {"kind": "tfidf_cosine"},
                    ],
                },
                {
                    "id": "blank-value",
                    "input": "question-value",
                    "reference": "a real reference",
                    "assertions": [{"kind": "exact_match", "value": " "}],
                },
                {
                    "id": "judge-case",
                    "input": "grade this",
                    "assertions": [
                        {"kind": "llm_judge", "params": {"rubric": "Score it."}},
                    ],
                },
                {
                    "id": "json-case",
                    "input": "json please",
                    "assertions": [{"kind": "json_valid"}],
                },
                {
                    "id": "valued",
                    "input": "has a value",
                    "assertions": [{"kind": "exact_match", "value": "ok"}],
                },
            ],
        },
    )
    code = main(["audit", str(path)])
    assert code == 0
    text = capsys.readouterr().out
    assert "severity: error" not in text
    assert "case: copy-b  severity: warning  message: identical input text to case 'copy-a'" in text
    assert "case: copy-a  severity:" not in text
    assert "contains has an empty reference and no value" in text
    assert "levenshtein has an empty reference and no value" in text
    assert "tfidf_cosine has an empty reference and no value" in text
    assert "exact_match value is empty" in text
    assert "judge-case" not in text
    assert "json-case" not in text
    assert "case: valued  severity:" not in text
    assert text.endswith("errors=0 warnings=5\n")


def test_audit_clean_suite_has_zero_findings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(
        tmp_path / "clean.json",
        {
            "name": "clean",
            "cases": [
                {
                    "id": "a",
                    "input": "one",
                    "reference": "one",
                    "assertions": [{"kind": "exact_match"}],
                },
                {
                    "id": "b",
                    "input": "two",
                    "reference": "two",
                    "assertions": [{"kind": "contains", "value": "two"}],
                },
            ],
        },
    )
    assert main(["audit", str(path)]) == 0
    text = capsys.readouterr().out
    assert "no findings" in text
    assert text.endswith("errors=0 warnings=0\n")
    assert main(["audit", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"findings": [], "errors": 0, "warnings": 0}


class _EntryPoint:
    def __init__(self, name: str, loaded: object) -> None:
        self.name = name
        self._loaded = loaded

    def load(self) -> object:
        return self._loaded


def test_audit_accepts_an_entry_point_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def plugin(output: str, case: Case, assertion: AssertionSpec) -> float:
        del output, case, assertion
        return 1.0

    path = _write(
        tmp_path / "plug.json",
        {
            "name": "plug",
            "cases": [
                {
                    "id": "p",
                    "input": "q",
                    "assertions": [{"kind": "plugin_score"}],
                }
            ],
        },
    )
    saved_scorers = dict(SCORERS)
    saved_exact = set(EXACT_KINDS)

    def fake_entry_points(**kwargs: object) -> list[_EntryPoint]:
        if kwargs.get("group") == "promptregress.scorers":
            return [_EntryPoint("plugin_score", plugin)]
        return []

    try:
        assert main(["audit", str(path)]) == 1
        assert "unknown scorer kind 'plugin_score'" in capsys.readouterr().out
        monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
        assert main(["audit", str(path), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["findings"] == []
        assert "plugin_score" in SCORERS
    finally:
        SCORERS.clear()
        SCORERS.update(saved_scorers)
        EXACT_KINDS.clear()
        EXACT_KINDS.update(saved_exact)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "audit"

# Six configs, config_a .. config_f. Score equals the pass bit.
# noise varies and does not track the totals. reverse is the mirror image.
# redundant_b flips one cell of redundant_a. dead fails everywhere.
PLANTED: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("core_a", (1, 1, 1, 0, 0, 0)),
    ("core_b", (1, 1, 0, 1, 0, 0)),
    ("core_c", (1, 0, 1, 0, 0, 1)),
    ("redundant_a", (0, 1, 1, 1, 0, 1)),
    ("redundant_b", (1, 1, 1, 1, 0, 1)),
    ("noise", (0, 1, 1, 0, 1, 0)),
    ("reverse", (0, 0, 0, 0, 1, 1)),
    ("dead", (0, 0, 0, 0, 0, 0)),
)
PLANTED_CONFIGS: tuple[str, ...] = (
    "config_a",
    "config_b",
    "config_c",
    "config_d",
    "config_e",
    "config_f",
)


def _result(case_id: str, passed: bool, score: float | None = None) -> CaseResult:
    value = float(int(passed)) if score is None else score
    return CaseResult(
        case_id=case_id,
        output="pass" if passed else "fail",
        scores={"exact_match": value},
        score=value,
        passed=passed,
        latency_ms=1.0,
    )


def _runs(
    configs: tuple[str, ...] | list[str],
    cases: tuple[tuple[str, tuple[int, ...]], ...] | list[tuple[str, tuple[int, ...]]],
    scores: dict[tuple[int, str], float] | None = None,
) -> list[RunResult]:
    overrides = {} if scores is None else scores
    runs: list[RunResult] = []
    for index, config in enumerate(configs):
        results: list[CaseResult] = []
        for case_id, bits in cases:
            passed = bool(bits[index])
            results.append(_result(case_id, passed, overrides.get((index, case_id))))
        runs.append(RunResult("suite-audit-demo", config, results))
    return runs


def planted_runs() -> list[RunResult]:
    return _runs(PLANTED_CONFIGS, PLANTED)


def write_example(dest: Path) -> None:
    """Write the three comparison files with ``compare_runs`` and ``write_json``."""
    dest.mkdir(parents=True, exist_ok=True)
    runs = planted_runs()
    pairs = ((0, 1), (2, 3), (4, 5))
    for index, (left, right) in enumerate(pairs, start=1):
        comparison = compare_runs(runs[left], runs[right], seed=AUDIT_SEED)
        write_json(comparison, dest / f"cmp_{index:02d}.json")


def _pearson(left: list[float], right: list[float]) -> float | None:
    mean_left = math.fsum(left) / len(left)
    mean_right = math.fsum(right) / len(right)
    delta_left = [value - mean_left for value in left]
    delta_right = [value - mean_right for value in right]
    sum_left = math.fsum(value * value for value in delta_left)
    sum_right = math.fsum(value * value for value in delta_right)
    if sum_left == 0.0 or sum_right == 0.0:
        return None
    product = math.fsum(a * b for a, b in zip(delta_left, delta_right, strict=True))
    return product / math.sqrt(sum_left * sum_right)


def test_suite_audit_constants_and_planted_flags() -> None:
    assert MIN_CONFIGS == 4
    assert MIN_CASES == 5
    assert NEAR_ZERO_DISCRIMINATION == 0.15
    assert STRONG_NEGATIVE_POINT_BISERIAL == -0.2
    assert MAX_REDUNDANT_DISAGREEMENTS == 1
    assert AUDIT_SEED == 13
    assert SUITE_AUDIT_LIMIT in SUITE_AUDIT_NOTE
    assert "They cost runtime and add no signal." in SUITE_AUDIT_NOTE

    report = audit_runs(planted_runs())
    by_id = {case.case_id: case for case in report.cases}
    assert report.n_configs == 6
    assert report.n_cases == 8
    assert report.configs == PLANTED_CONFIGS
    assert report.seed == 13
    assert [case.case_id for case in report.cases] == [case_id for case_id, _ in PLANTED]
    for case_id, bits in PLANTED:
        assert by_id[case_id].passes == "".join(str(bit) for bit in bits)

    assert by_id["noise"].flags == ("non_discriminating",)
    assert by_id["noise"].discrimination <= NEAR_ZERO_DISCRIMINATION
    assert by_id["noise"].point_biserial is not None
    assert by_id["noise"].point_biserial > STRONG_NEGATIVE_POINT_BISERIAL

    assert by_id["reverse"].flags == ("non_discriminating", "misfit")
    assert by_id["reverse"].discrimination <= NEAR_ZERO_DISCRIMINATION
    assert by_id["reverse"].point_biserial is not None
    assert by_id["reverse"].point_biserial <= STRONG_NEGATIVE_POINT_BISERIAL

    assert by_id["dead"].flags == ("dead",)
    assert by_id["dead"].n_pass == 0
    assert by_id["dead"].point_biserial is None
    assert by_id["dead"].discrimination > NEAR_ZERO_DISCRIMINATION

    assert [(pair.case_a, pair.case_b, pair.disagreements) for pair in report.redundant_pairs] == [
        ("redundant_a", "redundant_b", 1)
    ]
    assert by_id["redundant_a"].flags == ("redundant",)
    assert by_id["redundant_b"].flags == ("redundant",)
    for case_id in ("core_a", "core_b", "core_c"):
        assert by_id[case_id].flags == ()
        assert by_id[case_id].discrimination > NEAR_ZERO_DISCRIMINATION
        assert by_id[case_id].point_biserial is not None
        assert by_id[case_id].point_biserial > STRONG_NEGATIVE_POINT_BISERIAL

    text = format_suite_audit(report)
    assert text.startswith("suite audit\n")
    assert "case=noise discrimination=" in text
    assert "case=reverse point_biserial=" in text
    assert "case_a=redundant_a case_b=redundant_b disagreements=1" in text
    assert "case=dead passed=false" in text
    assert SUITE_AUDIT_LIMIT in text
    payload = suite_audit_to_dict(report)
    assert payload["dead"] == [{"case_id": "dead", "passed": False}]
    assert payload["response"] == "passed"
    assert payload["near_zero_discrimination"] == 0.15
    assert payload["strong_negative_point_biserial"] == -0.2


def test_suite_audit_threshold_is_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = planted_runs()
    base = audit_runs(runs)
    noise = next(case for case in base.cases if case.case_id == "noise")
    reverse = next(case for case in base.cases if case.case_id == "reverse")
    assert reverse.point_biserial is not None

    monkeypatch.setattr(
        "promptregress.audit.NEAR_ZERO_DISCRIMINATION",
        noise.discrimination,
    )
    assert "noise" in {case.case_id for case in audit_runs(runs).cases_with("non_discriminating")}
    monkeypatch.setattr(
        "promptregress.audit.NEAR_ZERO_DISCRIMINATION",
        noise.discrimination - 1e-9,
    )
    assert "noise" not in {
        case.case_id for case in audit_runs(runs).cases_with("non_discriminating")
    }

    monkeypatch.setattr(
        "promptregress.audit.STRONG_NEGATIVE_POINT_BISERIAL",
        reverse.point_biserial,
    )
    assert "reverse" in {case.case_id for case in audit_runs(runs).cases_with("misfit")}
    monkeypatch.setattr(
        "promptregress.audit.STRONG_NEGATIVE_POINT_BISERIAL",
        reverse.point_biserial - 1e-9,
    )
    assert "reverse" not in {case.case_id for case in audit_runs(runs).cases_with("misfit")}


def test_suite_audit_refuses_small_unpairable_and_nonfinite_inputs() -> None:
    wide = _runs(("c0", "c1", "c2", "c3"), (("a", (1, 0, 1, 0)),))
    with pytest.raises(ValueError, match="got 4 configs and 1 cases"):
        audit_runs(wide)
    short = _runs(
        ("c0", "c1", "c2"),
        (
            ("a", (1, 0, 1)),
            ("b", (1, 1, 0)),
            ("c", (0, 1, 0)),
            ("d", (0, 0, 1)),
            ("e", (1, 0, 0)),
        ),
    )
    with pytest.raises(ValueError, match="got 3 configs and 5 cases"):
        audit_runs(short)

    mismatched = planted_runs()
    dropped = mismatched[1].results[:-1]
    mismatched[1] = RunResult("suite-audit-demo", "config_b", dropped)
    with pytest.raises(ValueError, match="does not impute"):
        audit_runs(mismatched)

    nan = planted_runs()
    broken = nan[0].results[0]
    nan[0] = RunResult(
        "suite-audit-demo",
        "config_a",
        [CaseResult(broken.case_id, broken.output, broken.scores, float("nan"), True, 1.0)]
        + nan[0].results[1:],
    )
    with pytest.raises(ValueError, match="finite"):
        audit_runs(nan)

    flagged = planted_runs()
    first = flagged[0].results[0]
    flagged[0] = RunResult(
        "suite-audit-demo",
        "config_a",
        [CaseResult(first.case_id, first.output, first.scores, 1.0, 1, 1.0)]  # type: ignore[arg-type]
        + flagged[0].results[1:],
    )
    with pytest.raises(ValueError, match="boolean"):
        audit_runs(flagged)

    repeated = planted_runs()
    repeated[0] = RunResult(
        "suite-audit-demo",
        "config_a",
        [repeated[0].results[0], *repeated[0].results],
    )
    with pytest.raises(ValueError, match="repeats case id"):
        audit_runs(repeated)

    with pytest.raises(ValueError, match="at least one comparison"):
        audit_comparison_files([])
    with pytest.raises(ValueError, match="sequence of comparison"):
        audit_comparison_files("one.json")  # type: ignore[arg-type]


def test_suite_audit_duplicate_configs_and_case_alignment() -> None:
    runs = planted_runs()
    duplicate = RunResult("suite-audit-demo", "config_a", list(runs[0].results))
    report = audit_runs([*runs, duplicate])
    assert report.n_configs == 6
    assert len(report.duplicates_ignored) == 1
    assert report.duplicates_ignored[0].config == "config_a"
    assert report.duplicates_ignored[0].kept_path == "run 0"
    original = audit_runs(runs)
    assert [case.passes for case in report.cases] == [case.passes for case in original.cases]

    conflict = RunResult(
        "suite-audit-demo",
        "config_a",
        [
            _result(runs[0].results[0].case_id, not runs[0].results[0].passed),
            *runs[0].results[1:],
        ],
    )
    with pytest.raises(ValueError, match="different case scores or pass bits"):
        audit_runs([*runs, conflict])

    reordered = planted_runs()
    reordered[2] = RunResult(
        "suite-audit-demo",
        "config_c",
        list(reversed(reordered[2].results)),
    )
    aligned = audit_runs(reordered)
    original = audit_runs(runs)
    assert [case.passes for case in aligned.cases] == [case.passes for case in original.cases]

    identical = _runs(
        ("c0", "c1", "c2", "c3"),
        (
            ("same", (1, 1, 0, 0)),
            ("copy", (1, 1, 0, 0)),
            ("other", (0, 0, 1, 1)),
            ("left", (1, 0, 1, 0)),
            ("right", (0, 1, 0, 1)),
        ),
    )
    pairs = audit_runs(identical)
    assert [(pair.case_a, pair.case_b, pair.disagreements) for pair in pairs.redundant_pairs] == [
        ("same", "copy", 0)
    ]


def test_point_biserial_uses_score_totals() -> None:
    cases = (
        ("a", (1, 1, 0, 0)),
        ("b", (1, 0, 1, 0)),
        ("c", (1, 0, 0, 1)),
        ("d", (0, 1, 1, 0)),
        ("special", (1, 0, 1, 0)),
    )
    configs = ("c0", "c1", "c2", "c3")
    plain = audit_runs(_runs(configs, cases))
    inverted_scores = {(index, "special"): float(1 - bit) for index, bit in enumerate((1, 0, 1, 0))}
    inverted = audit_runs(_runs(configs, cases, scores=inverted_scores))
    plain_special = next(case for case in plain.cases if case.case_id == "special")
    inverted_special = next(case for case in inverted.cases if case.case_id == "special")
    assert plain_special.point_biserial != inverted_special.point_biserial

    passes = [1.0, 0.0, 1.0, 0.0]
    totals = []
    for index in range(4):
        scores = [float(bits[index]) for _, bits in cases]
        scores[-1] = float(1 - cases[-1][1][index])
        totals.append(math.fsum(scores))
    expected = _pearson(passes, totals)
    assert inverted_special.point_biserial == pytest.approx(expected)

    flat = audit_runs(
        _runs(
            configs,
            cases,
            scores={(index, case_id): 0.5 for index in range(4) for case_id, _ in cases},
        )
    )
    assert all(case.point_biserial is None for case in flat.cases)
    assert flat.cases_with("misfit") == ()


def test_zero_correlation_is_not_a_misfit() -> None:
    report = audit_runs(
        _runs(
            ("c0", "c1", "c2", "c3"),
            (
                ("a", (1, 1, 0, 0)),
                ("b", (1, 0, 1, 0)),
                ("c", (1, 0, 0, 1)),
                ("d", (0, 1, 1, 0)),
                ("e", (0, 1, 0, 1)),
            ),
        )
    )
    assert report.redundant_pairs == ()
    assert report.cases_with("dead") == ()
    orthogonal = [case for case in report.cases if case.point_biserial == 0.0]
    assert orthogonal
    assert all("misfit" not in case.flags for case in orthogonal)
    assert any("non_discriminating" in case.flags for case in orthogonal)


def test_example_files_round_trip_and_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)
    regenerated = tmp_path / "regen"
    write_example(regenerated)
    for name in ("cmp_01.json", "cmp_02.json", "cmp_03.json"):
        assert (regenerated / name).read_bytes() == (EXAMPLE / name).read_bytes()
        assert is_comparison_report(EXAMPLE / name)
        payload = json.loads((EXAMPLE / name).read_text(encoding="utf-8"))
        assert "gate" in payload
        assert "baseline" in payload and "candidate" in payload

    direct = audit_runs(planted_runs())
    loaded = audit_comparison_files(
        [EXAMPLE / "cmp_01.json", EXAMPLE / "cmp_02.json", EXAMPLE / "cmp_03.json"]
    )
    assert [case.passes for case in loaded.cases] == [case.passes for case in direct.cases]
    assert [case.flags for case in loaded.cases] == [case.flags for case in direct.cases]
    assert loaded.n_comparisons == 3
    assert loaded.comparison_paths[0].endswith("cmp_01.json")

    code = main(
        [
            "audit",
            "examples/audit/cmp_01.json",
            "examples/audit/cmp_02.json",
            "examples/audit/cmp_03.json",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.startswith("suite audit\n")
    assert "errors=" not in captured.out
    assert SUITE_AUDIT_LIMIT in captured.out

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    spec_path = ROOT / "SPEC.md"
    spec = spec_path.read_text(encoding="utf-8") if spec_path.exists() else None
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = changelog.split("## 0.13.3", 1)[0]
    for line in captured.out.splitlines():
        assert line in readme
    assert "promptregress audit examples/audit/cmp_01.json" in readme
    assert SUITE_AUDIT_LIMIT in readme
    assert SUITE_AUDIT_LIMIT in unreleased
    assert "NEAR_ZERO_DISCRIMINATION" in unreleased or "0.15" in unreleased
    if spec is not None:
        assert SUITE_AUDIT_LIMIT in spec
        assert "NEAR_ZERO_DISCRIMINATION" in spec
        assert "STRONG_NEGATIVE_POINT_BISERIAL" in spec
        assert "MIN_CONFIGS" in spec
        assert "MIN_CASES" in spec
        assert "MAX_REDUNDANT_DISAGREEMENTS" in spec
        assert "0.15" in spec
        assert "-0.2" in spec


def test_audit_cli_json_out_mix_and_suite_lint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "reports"
    write_example(dest)
    out = tmp_path / "audit.json"
    code = main(
        [
            "audit",
            str(dest / "cmp_01.json"),
            str(dest / "cmp_02.json"),
            str(dest / "cmp_03.json"),
            "--json",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert payload["n_configs"] == 6
    assert payload["non_discriminating"][0]["case_id"] == "noise"
    assert {item["case_id"] for item in payload["misfit"]} == {"reverse"}
    assert payload["redundant_pairs"] == [
        {"case_a": "redundant_a", "case_b": "redundant_b", "disagreements": 1}
    ]
    assert payload["dead"] == [{"case_id": "dead", "passed": False}]
    assert "refused" not in payload

    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [
                    {"id": "a", "input": "a", "assertions": [{"kind": "json_valid"}]},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert is_comparison_report(suite) is False
    assert is_comparison_report(dest / "cmp_01.json") is True
    mixed = main(["audit", str(suite), str(dest / "cmp_01.json")])
    assert mixed == 2
    assert "not a mix" in capsys.readouterr().err

    rejected = main(["audit", str(suite), "--out", str(tmp_path / "nope.json")])
    assert rejected == 2
    assert "--out writes the suite-audit JSON" in capsys.readouterr().err

    lint = main(["audit", str(suite)])
    assert lint == 0
    assert "errors=0 warnings=0" in capsys.readouterr().out

    too_small = main(["audit", str(dest / "cmp_01.json")])
    assert too_small == 2
    assert "at least 4 configs" in capsys.readouterr().err
