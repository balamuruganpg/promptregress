"""Offline diagnostics for a PromptRegress install."""

import importlib.metadata
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from promptregress import __version__
from promptregress.models import AssertionSpec, Case, Suite
from promptregress.providers import MockProvider
from promptregress.runner import run_suite

# Names only. Doctor never prints the values.
PROVIDER_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "XAI_API_KEY",
    "PROMPTREGRESS_API_KEY",
    "PROMPTREGRESS_BASE_URL",
    "PROMPTREGRESS_MODEL",
)


@dataclass(frozen=True)
class Check:
    """One diagnostic line. ``ok`` is false only when the check failed."""

    name: str
    ok: bool
    detail: str


def default_cache_dir() -> Path:
    """Cache directory doctor probes, next to the history file."""
    return Path.cwd() / ".promptregress" / "cache"


# Kept as a value so the diagnostic can fail on an older interpreter.
_MIN_PYTHON: tuple[int, int] = (3, 10)


def python_check() -> Check:
    """Pass when the interpreter meets ``_MIN_PYTHON``."""
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] >= _MIN_PYTHON:
        return Check("python", True, version)
    needed = ".".join(str(part) for part in _MIN_PYTHON)
    return Check("python", False, f"{version} requires Python >= {needed}")


def package_check() -> Check:
    """Pass when the installed package exposes a version string."""
    if isinstance(__version__, str) and __version__ != "":
        return Check("package", True, f"promptregress {__version__}")
    return Check("package", False, "promptregress version is missing")


def provider_check() -> Check:
    """Report whether provider env vars are set, without their values.

    Absent is fine: the offline demo does not need a key. A variable that is
    set to an empty or all-whitespace string is a failed sanity check.
    """
    parts: list[str] = []
    ok = True
    for name in PROVIDER_ENV_VARS:
        if name not in os.environ:
            parts.append(f"{name}=absent")
            continue
        if os.environ[name].strip() == "":
            parts.append(f"{name}=empty")
            ok = False
            continue
        parts.append(f"{name}=present")
    return Check("providers", ok, " ".join(parts))


def cache_check(directory: Path | None = None) -> Check:
    """Pass when ``directory`` can be created and a probe file written.

    The default is :func:`default_cache_dir`. The probe file is removed.
    """
    target = default_cache_dir() if directory is None else Path(directory)
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".doctor-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check("cache", False, f"{target} not writable ({exc.__class__.__name__})")
    return Check("cache", True, f"{target} writable")


def plugin_check() -> Check:
    """List ``promptregress.scorers`` entry points and try to load each one.

    An empty group is a pass. A load error fails the check. Names are
    reported; nothing else about the callable is printed.
    """
    try:
        discovered = importlib.metadata.entry_points(group="promptregress.scorers")
    except Exception as exc:
        return Check("scorer_plugins", False, f"lookup failed ({exc.__class__.__name__})")
    names: list[str] = []
    failures: list[str] = []
    for entry in discovered:
        try:
            entry.load()
        except Exception as exc:
            failures.append(f"{entry.name} ({exc.__class__.__name__})")
            continue
        names.append(entry.name)
    if failures:
        found = ", ".join(names) if names else "none"
        bad = ", ".join(failures)
        return Check("scorer_plugins", False, f"found {found}; failed {bad}")
    if not names:
        return Check("scorer_plugins", True, "none")
    return Check("scorer_plugins", True, ", ".join(names))


def smoke_check() -> Check:
    """Score one case through :class:`MockProvider`.

    The rule returns the reference for the input. The check passes only when
    that case scores 1 and passes, with no error string.
    """
    case = Case(
        id="doctor-smoke",
        input="ping",
        reference="pong",
        assertions=[AssertionSpec(kind="exact_match", value="pong")],
        tags=["doctor"],
    )
    suite = Suite(name="doctor", cases=[case], description="doctor smoke")
    provider = MockProvider(rules=[("ping", "pong")], default="fail", name="mock", model="doctor")
    try:
        result = run_suite(suite, provider, "smoke", prompt_template="{input}")
    except Exception as exc:
        return Check("smoke", False, f"{exc.__class__.__name__}: {exc}")
    if len(result.results) != 1:
        return Check("smoke", False, "expected one case")
    row = result.results[0]
    if row.error or not row.passed or row.score != 1.0 or row.output != "pong":
        passed = str(row.passed).lower()
        return Check("smoke", False, f"score={row.score:.6f} passed={passed} error={row.error}")
    return Check("smoke", True, "mock score=1.000000 passed=true")


def collect_checks(cache_dir: Path | None = None) -> list[Check]:
    """Run every doctor check. ``cache_dir`` overrides the default cache path."""
    return [
        python_check(),
        package_check(),
        provider_check(),
        cache_check(cache_dir),
        plugin_check(),
        smoke_check(),
    ]


def format_report(checks: list[Check]) -> str:
    """Plain-text report. Ends with a newline. No secret values."""
    lines = ["PromptRegress doctor"]
    for check in checks:
        status = "ok" if check.ok else "FAIL"
        lines.append(f"{check.name}: {status} {check.detail}")
    failed = sum(1 for check in checks if not check.ok)
    passed = len(checks) - failed
    outcome = "pass" if failed == 0 else "fail"
    lines.append(f"result: {outcome} ({passed} passed, {failed} failed)")
    return "\n".join(lines) + "\n"


def run_doctor(cache_dir: Path | None = None) -> tuple[str, bool]:
    """Return ``(report, all_checks_passed)``."""
    checks = collect_checks(cache_dir)
    return format_report(checks), all(check.ok for check in checks)
