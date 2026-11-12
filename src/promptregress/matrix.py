"""Rank two or more named configs on one suite.

The best config is the highest mean score. Equal means are broken by config
name, ascending, so the reference row is stable. Every other row reports
``mean(this - best)`` on the paired case scores, with a bootstrap confidence
interval from :func:`promptregress.stats.compare_runs`. The best row's
difference is 0 with a point interval. This is a ranking, not a gate.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from promptregress.cache import ResponseCache
from promptregress.loading import strip_utf8_bom_text
from promptregress.models import RunResult, Suite
from promptregress.providers import Provider, provider_model_name
from promptregress.runner import run_suite
from promptregress.stats import compare_runs

_PROVIDER_KINDS = {"mock", "echo", "openai"}


class MatrixError(ValueError):
    """The matrix file or the requested comparison is not usable."""


@dataclass(frozen=True)
class MatrixConfig:
    """One named config inside a matrix file.

    ``mock_rules`` and ``mock_default`` are ``None`` when the file does not
    set them, so the CLI can fall back to its own flags. An explicit empty
    rule list is an empty tuple, not ``None``.
    """

    name: str
    provider: str
    template: str
    model: str | None = None
    mock_rules: tuple[tuple[str, str], ...] | None = None
    mock_default: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = None


@dataclass(frozen=True)
class MatrixRow:
    """One config after ranking, including its difference from the best."""

    rank: int
    name: str
    provider: str
    model: str
    template: str
    mean_score: float
    pass_rate: float
    diff_vs_best: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class MatrixResult:
    """Ranking of a matrix. ``rows`` is best-first."""

    suite_name: str
    best_name: str
    rows: list[MatrixRow]
    n_boot: int
    seed: int
    alpha: float
    ci_method: str


def load_matrix(path: str | Path) -> list[MatrixConfig]:
    """Load two or more named configs from a YAML list or mapping.

    A list uses each object's ``name``. A mapping uses the keys as names.
    Provider, model, and template are the config. ``provider`` is ``mock``,
    ``echo``, or ``openai``. ``model`` may be omitted.
    """
    matrix_path = Path(path)
    if not matrix_path.is_file():
        raise MatrixError(f"Matrix file not found: {matrix_path}")
    try:
        # utf-8-sig (+ residual / per-line BOM strip) matches suite/pricing
        # YAML: a double BOM or a mid-file Windows-chunk BOM must not glue
        # onto a config name as U+FEFF.
        text = strip_utf8_bom_text(matrix_path.read_text(encoding="utf-8-sig"))
    except UnicodeError as exc:
        raise MatrixError(f"Matrix file {matrix_path} is not valid UTF-8: {exc}") from exc
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise MatrixError(f"Matrix file is not valid YAML: {exc}") from exc
    raw_items = _raw_items(loaded, matrix_path)
    if len(raw_items) < 2:
        raise MatrixError("A matrix needs at least two named configs")
    configs: list[MatrixConfig] = []
    seen: set[str] = set()
    for index, (name, body) in enumerate(raw_items):
        configs.append(_config_from_body(name, body, index))
        if name in seen:
            raise MatrixError(f"Duplicate matrix config name {name!r}")
        seen.add(name)
    return configs


def run_matrix(
    suite: Suite,
    entries: Sequence[tuple[MatrixConfig, Provider]],
    *,
    cache: ResponseCache | None = None,
    max_workers: int = 1,
    n_boot: int = 2000,
    seed: int = 13,
    alpha: float = 0.05,
    ci_method: str = "percentile",
) -> MatrixResult:
    """Run every config and rank the runs by mean score."""
    if len(entries) < 2:
        raise MatrixError("A matrix needs at least two named configs")
    finished: list[tuple[MatrixConfig, RunResult, str]] = []
    for config, provider in entries:
        result = run_suite(
            suite,
            provider,
            config.name,
            prompt_template=config.template,
            cache=cache,
            max_workers=max_workers,
        )
        finished.append((config, result, provider_model_name(provider)))
    return rank_runs(
        finished,
        suite_name=suite.name,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        ci_method=ci_method,
    )


def rank_runs(
    finished: Sequence[tuple[MatrixConfig, RunResult, str]],
    *,
    suite_name: str,
    n_boot: int = 2000,
    seed: int = 13,
    alpha: float = 0.05,
    ci_method: str = "percentile",
) -> MatrixResult:
    """Rank runs that have already been scored.

    ``finished`` is ``(config, run, model name)``. The best run is the
    reference for every paired difference, including its own.
    """
    if len(finished) < 2:
        raise MatrixError("A matrix needs at least two named configs")
    ordered = sorted(finished, key=lambda item: (-item[1].mean_score, item[0].name))
    best_config, best_run, _best_model = ordered[0]
    rows: list[MatrixRow] = []
    for rank, (config, run, model) in enumerate(ordered, start=1):
        comparison = compare_runs(
            best_run,
            run,
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
            ci_method=ci_method,
        )
        rows.append(
            MatrixRow(
                rank=rank,
                name=config.name,
                provider=config.provider,
                model=model,
                template=config.template,
                mean_score=run.mean_score,
                pass_rate=run.pass_rate,
                diff_vs_best=comparison.mean_diff,
                ci_low=comparison.ci_low,
                ci_high=comparison.ci_high,
            )
        )
    return MatrixResult(
        suite_name=suite_name,
        best_name=best_config.name,
        rows=rows,
        n_boot=n_boot,
        seed=seed,
        alpha=float(alpha),
        ci_method=ci_method,
    )


def matrix_to_dict(result: MatrixResult) -> dict[str, Any]:
    """JSON-ready ranking."""
    return {
        "suite_name": result.suite_name,
        "best": result.best_name,
        "n_boot": result.n_boot,
        "seed": result.seed,
        "alpha": result.alpha,
        "ci_method": result.ci_method,
        "rows": [
            {
                "rank": row.rank,
                "name": row.name,
                "provider": row.provider,
                "model": row.model,
                "template": row.template,
                "mean_score": row.mean_score,
                "pass_rate": row.pass_rate,
                "diff_vs_best": row.diff_vs_best,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
            }
            for row in result.rows
        ],
    }


def render_matrix_markdown(result: MatrixResult) -> str:
    """Markdown ranking table, including a trailing newline."""
    level = int(round((1.0 - result.alpha) * 100))
    lines = [
        "# PromptRegress matrix",
        "",
        f"Suite: `{_cell(result.suite_name)}`",
        "",
        f"Best: `{_cell(result.best_name)}`",
        "",
        "Diff vs best is the mean of this config's case score minus the best",
        "config's case score. The interval is a bootstrap CI for that mean.",
        "Ties on the mean score are broken by config name.",
        "",
        f"| Rank | Config | Mean score | Pass rate | Diff vs best | {level}% CI low | "
        f"{level}% CI high |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.rows:
        lines.append(
            f"| {row.rank} | {_cell(row.name)} | {row.mean_score:.6f} | "
            f"{row.pass_rate:.6f} | {row.diff_vs_best:.6f} | {row.ci_low:.6f} | "
            f"{row.ci_high:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_matrix_json(result: MatrixResult, path: str | Path) -> None:
    """Write the ranking as indented JSON."""
    _write(path, json.dumps(matrix_to_dict(result), indent=2) + "\n")


def write_matrix_markdown(result: MatrixResult, path: str | Path) -> None:
    """Write the Markdown ranking table."""
    _write(path, render_matrix_markdown(result))


def _raw_items(loaded: Any, path: Path) -> list[tuple[str, Any]]:
    if isinstance(loaded, list):
        items: list[tuple[str, Any]] = []
        for index, body in enumerate(loaded):
            if not isinstance(body, dict):
                raise MatrixError(f"{path}: config {index} must be a mapping")
            name = body.get("name")
            if not isinstance(name, str) or name.strip() == "":
                raise MatrixError(f"{path}: config {index} needs a non-empty name")
            items.append((name, body))
        return items
    if isinstance(loaded, dict):
        items = []
        for name, body in loaded.items():
            if not isinstance(name, str) or name.strip() == "":
                raise MatrixError(f"{path}: config names must be non-empty strings")
            if not isinstance(body, dict):
                raise MatrixError(f"{path}: config {name!r} must be a mapping")
            items.append((name, body))
        return items
    raise MatrixError(f"{path}: matrix must be a list or a mapping of configs")


def _config_from_body(name: str, body: dict[str, Any], index: int) -> MatrixConfig:
    provider = body.get("provider")
    if not isinstance(provider, str) or provider not in _PROVIDER_KINDS:
        raise MatrixError(f"Config {name!r} provider must be one of: mock, echo, openai")
    template = body.get("template")
    if not isinstance(template, str):
        raise MatrixError(f"Config {name!r} needs a string template")
    model = _optional_str(body, "model", name)
    api_key = _optional_str(body, "api_key", name)
    base_url = _optional_str(body, "base_url", name)
    mock_default = _optional_str(body, "mock_default", name)
    timeout = _optional_timeout(body, name)
    mock_rules: tuple[tuple[str, str], ...] | None
    if "mock_rules" not in body or body["mock_rules"] is None:
        mock_rules = None
    else:
        mock_rules = tuple(_rule_list(body["mock_rules"], name))
    listed_name = body.get("name", name)
    if listed_name != name:
        raise MatrixError(f"Config {index} name {listed_name!r} does not match {name!r}")
    return MatrixConfig(
        name=name,
        provider=provider,
        template=template,
        model=model,
        mock_rules=mock_rules,
        mock_default=mock_default,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )


def _optional_str(body: dict[str, Any], key: str, name: str) -> str | None:
    if key not in body or body[key] is None:
        return None
    value = body[key]
    if not isinstance(value, str):
        raise MatrixError(f"Config {name!r} field {key} must be a string")
    return value


def _optional_timeout(body: dict[str, Any], name: str) -> float | None:
    if "timeout" not in body or body["timeout"] is None:
        return None
    value = body["timeout"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise MatrixError(f"Config {name!r} timeout must be a positive number")
    return float(value)


def _rule_list(raw: Any, name: str) -> list[tuple[str, str]]:
    if not isinstance(raw, list):
        raise MatrixError(f"Config {name!r} mock_rules must be a list of pairs")
    rules: list[tuple[str, str]] = []
    for index, item in enumerate(raw):
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
        ):
            raise MatrixError(f"Config {name!r} mock_rules item {index} must be a pair of strings")
        rules.append((item[0], item[1]))
    return rules


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _write(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
