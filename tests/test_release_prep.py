"""Release workflow, contributor docs, and the pre-commit hooks."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> object:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_release_workflow_builds_on_version_tags() -> None:
    path = ROOT / ".github" / "workflows" / "release.yml"
    workflow = _load_yaml(path)
    assert isinstance(workflow, dict)
    # PyYAML 1.1 treats the bare key `on` as boolean true. GitHub Actions
    # uses YAML 1.2, where `on` is a string.
    trigger = workflow.get("on", workflow.get(True))
    assert isinstance(trigger, dict)
    assert list(trigger) == ["push"]
    assert trigger["push"]["tags"] == ["v*"]
    text = path.read_text(encoding="utf-8").lower()
    assert "pypi" not in text
    assert "twine" not in text

    job = workflow["jobs"]["build"]
    assert job["runs-on"] == "ubuntu-latest"
    steps = job["steps"]
    runs = "\n".join(step.get("run", "") for step in steps)
    assert "pip install build" in runs
    assert "python -m build" in runs
    uses = [step.get("uses", "") for step in steps]
    assert any(item.startswith("actions/checkout@") for item in uses)
    assert any(item.startswith("actions/setup-python@") for item in uses)
    upload = next(
        step for step in steps if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    )
    assert upload["with"]["path"] == "dist/*"
    assert upload["with"]["if-no-files-found"] == "error"


def test_contributing_documents_the_gate_and_scorers() -> None:
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert text.strip()
    for command in ("pytest", "ruff check", "ruff format --check", "mypy src/promptregress"):
        assert command in text
    assert "register_scorer" in text
    assert "promptregress.scorers" in text


def test_security_policy_covers_reporting_and_supported_versions() -> None:
    text = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert text.strip()
    assert "vulnerability" in text.lower()
    assert "security/advisories" in text
    assert "0.2.x" in text
    assert "0.1.x" in text


def test_pre_commit_config_pins_ruff_and_mypy() -> None:
    config = _load_yaml(ROOT / ".pre-commit-config.yaml")
    assert isinstance(config, dict)
    repos = config["repos"]
    assert isinstance(repos, list)
    by_repo = {item["repo"]: item for item in repos}
    ruff = by_repo["https://github.com/astral-sh/ruff-pre-commit"]
    assert ruff["rev"] == "v0.17.4"
    assert [hook["id"] for hook in ruff["hooks"]] == ["ruff-check", "ruff-format"]
    mypy_repo = by_repo["https://github.com/pre-commit/mirrors-mypy"]
    assert mypy_repo["rev"] == "v2.4.1"
    assert [hook["id"] for hook in mypy_repo["hooks"]] == ["mypy"]
    assert mypy_repo["hooks"][0]["args"] == ["src/promptregress"]
    assert mypy_repo["hooks"][0]["pass_filenames"] is False


def test_dist_directory_is_gitignored() -> None:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "dist/" in lines
