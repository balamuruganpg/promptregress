"""promptregress doctor: pass path and one forced failure."""

import importlib.metadata
import sys
from pathlib import Path

import pytest

import promptregress
from promptregress.cli import main
from promptregress.doctor import (
    Check,
    cache_check,
    package_check,
    plugin_check,
    provider_check,
    python_check,
    smoke_check,
)
from promptregress.models import CaseResult, RunResult


def _quiet_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-promptregress-doctor-secret")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PROMPTREGRESS_API_KEY", raising=False)
    monkeypatch.delenv("PROMPTREGRESS_BASE_URL", raising=False)
    monkeypatch.delenv("PROMPTREGRESS_MODEL", raising=False)


def test_doctor_passes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _quiet_env(monkeypatch)
    with pytest.raises(SystemExit) as help_exit:
        main(["doctor", "--help"])
    assert help_exit.value.code == 0
    assert "smoke" in capsys.readouterr().out

    assert main(["doctor"]) == 0
    report = capsys.readouterr().out
    assert "PromptRegress doctor" in report
    assert "python: ok " in report
    assert f"package: ok promptregress {promptregress.__version__}" in report
    assert "OPENAI_API_KEY=present" in report
    assert "XAI_API_KEY=absent" in report
    assert "ANTHROPIC_API_KEY=absent" in report
    assert "PROMPTREGRESS_API_KEY=absent" in report
    assert "PROMPTREGRESS_BASE_URL=absent" in report
    assert "PROMPTREGRESS_MODEL=absent" in report
    assert "sk-promptregress-doctor-secret" not in report
    assert "cache: ok " in report
    assert "writable" in report
    assert "scorer_plugins: ok " in report
    assert "smoke: ok mock score=1.000000 passed=true" in report
    assert "result: pass (6 passed, 0 failed)" in report
    assert not (tmp_path / ".promptregress" / "cache" / ".doctor-write-probe").exists()


def test_doctor_forced_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _quiet_env(monkeypatch)

    def forced() -> Check:
        return Check("smoke", False, "forced failure")

    monkeypatch.setattr("promptregress.doctor.smoke_check", forced)
    assert main(["doctor"]) == 1
    report = capsys.readouterr().out
    assert "python: ok " in report
    assert "smoke: FAIL forced failure" in report
    assert "result: fail (5 passed, 1 failed)" in report
    assert "sk-promptregress-doctor-secret" not in report


def test_python_check_rejects_an_old_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 9, 1, "final", 0))
    check = python_check()
    assert check == Check("python", False, "3.9.1 requires Python >= 3.10")


def test_package_check_rejects_a_missing_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("promptregress.doctor.__version__", "")
    check = package_check()
    assert check == Check("package", False, "promptregress version is missing")


def test_provider_check_fails_on_a_blank_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PROMPTREGRESS_API_KEY", raising=False)
    monkeypatch.delenv("PROMPTREGRESS_BASE_URL", raising=False)
    monkeypatch.delenv("PROMPTREGRESS_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", " \t")
    check = provider_check()
    assert check.ok is False
    assert "OPENAI_API_KEY=empty" in check.detail
    assert "XAI_API_KEY=absent" in check.detail
    assert "ANTHROPIC_API_KEY=absent" in check.detail
    assert "PROMPTREGRESS_MODEL=absent" in check.detail


def test_provider_check_reports_anthropic_without_the_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "PROMPTREGRESS_API_KEY",
        "PROMPTREGRESS_BASE_URL",
        "PROMPTREGRESS_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-doctor-secret")
    check = provider_check()
    assert check.ok is True
    assert "ANTHROPIC_API_KEY=present" in check.detail
    assert "sk-ant-doctor-secret" not in check.detail
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    absent = provider_check()
    assert absent.ok is True
    assert "ANTHROPIC_API_KEY=absent" in absent.detail


def test_cache_check_reports_oserror(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError("read-only")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    target = tmp_path / "blocked"
    check = cache_check(target)
    assert check.ok is False
    assert check.name == "cache"
    assert str(target) in check.detail
    assert "not writable (OSError)" in check.detail


class _Entry:
    def __init__(self, name: str, *, error: BaseException | None = None) -> None:
        self.name = name
        self._error = error

    def load(self) -> object:
        if self._error is not None:
            raise self._error
        return object()


def _patch_entries(monkeypatch: pytest.MonkeyPatch, entries: list[_Entry] | None) -> None:
    def fake_entry_points(**kwargs: object) -> list[_Entry]:
        if entries is None:
            raise OSError("catalog down")
        assert kwargs.get("group") == "promptregress.scorers"
        return list(entries)

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)


def test_plugin_check_lists_loaded_names(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entries(monkeypatch, [_Entry("alpha"), _Entry("beta")])
    check = plugin_check()
    assert check == Check("scorer_plugins", True, "alpha, beta")


def test_plugin_check_reports_a_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entries(monkeypatch, [_Entry("good"), _Entry("bad", error=RuntimeError("x"))])
    check = plugin_check()
    assert check.ok is False
    assert check.detail == "found good; failed bad (RuntimeError)"


def test_plugin_check_when_every_load_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entries(monkeypatch, [_Entry("bad", error=ValueError("x"))])
    check = plugin_check()
    assert check.ok is False
    assert check.detail == "found none; failed bad (ValueError)"


def test_plugin_check_when_lookup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entries(monkeypatch, None)
    check = plugin_check()
    assert check == Check("scorer_plugins", False, "lookup failed (OSError)")


def test_smoke_check_reports_a_raised_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> RunResult:
        raise RuntimeError("down")

    monkeypatch.setattr("promptregress.doctor.run_suite", boom)
    check = smoke_check()
    assert check == Check("smoke", False, "RuntimeError: down")


def test_smoke_check_expects_one_case(monkeypatch: pytest.MonkeyPatch) -> None:
    def empty(*args: object, **kwargs: object) -> RunResult:
        return RunResult(suite_name="doctor", config_name="smoke", results=[])

    monkeypatch.setattr("promptregress.doctor.run_suite", empty)
    check = smoke_check()
    assert check == Check("smoke", False, "expected one case")


def test_smoke_check_reports_a_wrong_score(monkeypatch: pytest.MonkeyPatch) -> None:
    def wrong(*args: object, **kwargs: object) -> RunResult:
        return RunResult(
            suite_name="doctor",
            config_name="smoke",
            results=[
                CaseResult(
                    case_id="doctor-smoke",
                    output="nope",
                    scores={"exact_match": 0.0},
                    score=0.0,
                    passed=False,
                    latency_ms=0.0,
                    error="bad",
                )
            ],
        )

    monkeypatch.setattr("promptregress.doctor.run_suite", wrong)
    check = smoke_check()
    assert check.ok is False
    assert check.detail == "score=0.000000 passed=false error=bad"
