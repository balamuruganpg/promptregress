"""Suite audit: each check, exit codes, JSON, and a clean file."""

import importlib.metadata
import json
from pathlib import Path

import pytest

from promptregress.audit import REFERENCE_KINDS
from promptregress.cli import main
from promptregress.models import AssertionSpec, Case
from promptregress.scorers import EXACT_KINDS, SCORERS


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
