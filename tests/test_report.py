"""JSON, Markdown, and HTML reports contain the comparison and the gate."""

import json
from pathlib import Path

from helpers import run_with
from promptregress.gate import evaluate_gate
from promptregress.report import format_text, render_html, write_html, write_json, write_markdown
from promptregress.stats import compare_runs


def _comparison():
    baseline = run_with(
        [("password_reset", 1.0, True), ("hours", 1.0, True)],
        "baseline",
    )
    candidate = run_with(
        [("password_reset", 0.0, False), ("hours", 1.0, True)],
        "candidate",
    )
    return compare_runs(baseline, candidate, n_boot=200, seed=13)


def test_reports_include_gate_and_cases(tmp_path: Path) -> None:
    comparison = _comparison()
    verdict = evaluate_gate(comparison, max_regression=0.0)
    assert verdict.passed is False
    json_path = tmp_path / "nested" / "report.json"
    md_path = tmp_path / "nested" / "report.md"
    write_json(comparison, json_path, verdict)
    write_markdown(comparison, md_path, verdict)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["mean_diff"] == comparison.mean_diff
    assert payload["ci_low"] == comparison.ci_low
    assert payload["ci_high"] == comparison.ci_high
    assert payload["gate"]["passed"] is False
    assert payload["losses"] == 1
    assert payload["wins"] == 0
    assert payload["per_case"][0]["case_id"] == "password_reset"
    assert payload["per_case"][0]["outcome"] == "loss"
    assert "FAIL regression" in payload["gate"]["reasons"][0]
    text = md_path.read_text(encoding="utf-8")
    assert "password_reset" in text
    assert "loss" in text
    assert "## Gate" in text
    assert "**FAIL**" in text
    assert "Mean diff" in text
    assert "McNemar" in text
    plain = format_text(comparison, verdict)
    assert "mean_diff=" in plain
    assert "gate_passed=false" in plain
    assert "password_reset" in plain


def test_omitted_verdict_uses_the_default_gate(tmp_path: Path) -> None:
    comparison = _comparison()
    path = tmp_path / "report.md"
    write_markdown(comparison, path)
    text = path.read_text(encoding="utf-8")
    assert "**FAIL**" in text
    assert "max_regression 0.000000" in text
    html_path = tmp_path / "report.html"
    write_html(comparison, html_path)
    html = html_path.read_text(encoding="utf-8")
    assert 'class="verdict fail"' in html
    assert "max_regression 0.000000" in html


def test_html_report_is_self_contained(tmp_path: Path) -> None:
    comparison = _comparison()
    verdict = evaluate_gate(comparison, max_regression=0.0)
    path = tmp_path / "nested" / "report.html"
    write_html(comparison, path, verdict)
    text = path.read_text(encoding="utf-8")
    assert "<style>" in text
    assert "</style>" in text
    assert "<link" not in text.lower()
    assert "<script" not in text.lower()
    assert "http://" not in text
    assert "https://" not in text
    assert "suite" in text
    assert "baseline" in text
    assert "candidate" in text
    assert "password_reset" in text
    assert "loss" in text
    assert "Paired statistics" in text
    assert "Mean diff (candidate - baseline)" in text
    assert "McNemar" in text
    assert f"{comparison.mean_diff:.6f}" in text
    assert f"{comparison.ci_low:.6f}" in text
    assert f"{comparison.ci_high:.6f}" in text
    assert "Gate" in text
    assert 'class="verdict fail"' in text
    assert ">FAIL<" in text
    assert "FAIL regression" in text
    assert "Per-case results" in text
    assert "Wins (candidate score higher)" in text
    assert "Losses (candidate score lower)" in text
    passed = evaluate_gate(comparison, max_regression=1.0)
    passing = render_html(comparison, passed)
    assert 'class="verdict pass"' in passing
    assert ">PASS<" in passing
    assert "PASS regression" in passing


def test_reports_include_advanced_stats_and_slices(tmp_path: Path) -> None:
    tags = {"a": ["billing"], "b": ["support"]}
    baseline = run_with([("a", 1.0, True), ("b", 0.0, False)], "baseline", tags=tags)
    candidate = run_with([("a", 0.0, False), ("b", 1.0, True)], "candidate", tags=tags)
    comparison = compare_runs(baseline, candidate, n_boot=80, seed=5)
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    html_path = tmp_path / "report.html"
    write_json(comparison, json_path)
    write_markdown(comparison, md_path)
    write_html(comparison, html_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "wilcoxon_p" in payload
    assert "ttest_p" in payload
    assert payload["effect_size_label"] in {"negligible", "small", "medium", "large"}
    assert payload["ci_method"] == "percentile"
    slice_tags = {item["tag"] for item in payload["slices"]}
    assert slice_tags == {"billing", "support"}
    markdown = md_path.read_text(encoding="utf-8")
    assert "Wilcoxon p-value" in markdown
    assert "Paired t p-value" in markdown
    assert "Cohen's dz" in markdown
    assert "## Slices" in markdown
    assert "billing" in markdown
    html = html_path.read_text(encoding="utf-8")
    assert "Wilcoxon p-value" in html
    assert "Paired t p-value" in html
    assert "Cohen&#x27;s dz" in html or "Cohen's dz" in html
    assert "Slices" in html
    assert "billing" in html
    plain = format_text(comparison)
    assert "wilcoxon_p=" in plain
    assert "ttest_p=" in plain
    assert "effect_size=" in plain
    assert "slices:" in plain


def test_html_escapes_case_text() -> None:
    baseline = run_with([("<script>", 1.0, True)], "base <b>")
    candidate = run_with([("<script>", 0.0, False)], "cand")
    comparison = compare_runs(baseline, candidate, n_boot=20, seed=1)
    text = render_html(comparison, evaluate_gate(comparison, max_regression=1.0))
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert "base &lt;b&gt;" in text
