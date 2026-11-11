"""The composite action, the dogfood workflow, and the demo configs they run."""

import json
from pathlib import Path

import pytest
import yaml

from promptregress.cli import (
    DEMO_BASELINE_TEMPLATE,
    DEMO_CANDIDATE_TEMPLATE,
    demo_rules,
    main,
)
from promptregress.loading import load_suite

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> object:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_action_and_selftest_yaml_parse() -> None:
    action = _load_yaml(ROOT / "action.yml")
    assert isinstance(action, dict)
    assert action["name"] == "PromptRegress gate"
    assert action["runs"]["using"] == "composite"
    inputs = action["inputs"]
    for name in (
        "suite",
        "baseline-config",
        "candidate-config",
        "max-regression",
        "python-version",
    ):
        assert name in inputs
    assert inputs["suite"]["required"] is True
    assert inputs["baseline-config"]["required"] is True
    assert inputs["candidate-config"]["required"] is True
    assert inputs["max-regression"]["default"] == "0"
    steps = action["runs"]["steps"]
    uses = [step.get("uses", "") for step in steps]
    assert all("checkout" not in item for item in uses)
    assert any(item.startswith("actions/setup-python@") for item in uses)
    assert any(item.startswith("actions/upload-artifact@") for item in uses)
    script = "\n".join(step.get("run", "") for step in steps)
    assert "pip install" in script
    assert "promptregress" in script
    assert "pyproject.toml" in script
    assert "GITHUB_ACTION_PATH" in script
    text = (ROOT / "action.yml").read_text(encoding="utf-8")
    assert "Checkout is the caller's job" in text

    # PyYAML 1.1 treats the bare key `on` as boolean true. GitHub Actions uses
    # YAML 1.2, where `on` is a string. Accept either so safe_load still proves
    # the workflow file parses.
    workflow = _load_yaml(ROOT / ".github" / "workflows" / "pr-selftest.yml")
    assert isinstance(workflow, dict)
    trigger = workflow.get("on", workflow.get(True))
    assert isinstance(trigger, dict)
    assert "push" in trigger
    job = workflow["jobs"]["demo-gate"]
    gate = next(step for step in job["steps"] if step.get("uses") == "./")
    assert gate["continue-on-error"] is True
    assert gate["with"]["suite"] == "examples/support_qa.json"
    assert gate["with"]["baseline-config"] == "examples/ci/baseline.yaml"
    assert gate["with"]["candidate-config"] == "examples/ci/candidate.yaml"
    assert gate["with"]["max-regression"] == "0"
    workflow_text = (ROOT / ".github" / "workflows" / "pr-selftest.yml").read_text(encoding="utf-8")
    assert 'test "$code" = "1"' in workflow_text
    assert "fails by design" in workflow_text.lower()


def test_demo_ci_configs_fail_the_gate(capsys: pytest.CaptureFixture[str]) -> None:
    suite = load_suite(ROOT / "examples" / "support_qa.json")
    baseline = _load_yaml(ROOT / "examples" / "ci" / "baseline.yaml")
    candidate = _load_yaml(ROOT / "examples" / "ci" / "candidate.yaml")
    assert isinstance(baseline, dict)
    assert isinstance(candidate, dict)
    assert baseline["provider"] == "mock"
    assert baseline["template"] == DEMO_BASELINE_TEMPLATE
    assert baseline["mock_default"] == "I do not know."
    assert [tuple(pair) for pair in baseline["mock_rules"]] == demo_rules(suite)
    assert candidate["template"] == DEMO_CANDIDATE_TEMPLATE

    code = main(
        [
            "compare",
            str(ROOT / "examples" / "support_qa.json"),
            "--provider",
            str(baseline["provider"]),
            "--baseline-template",
            str(baseline["template"]),
            "--candidate-template",
            str(candidate["template"]),
            "--mock-rules",
            json.dumps(baseline["mock_rules"]),
            "--mock-default",
            str(baseline["mock_default"]),
            "--max-regression",
            "0",
            "--no-history",
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "mean_diff=-0.214286" in captured.out
    assert "gate_passed=false" in captured.out
    assert "FAIL regression" in captured.out
