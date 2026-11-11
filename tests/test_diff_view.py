"""Side-by-side outputs for cases whose score or text changed."""

from helpers import case_result
from promptregress.cluster import make_embedder
from promptregress.drift import measure_drift, with_drift
from promptregress.gate import evaluate_gate
from promptregress.models import RunResult
from promptregress.report import render_html, render_markdown
from promptregress.stats import compare_runs


def test_diff_view_shows_changed_outputs_and_escapes_html() -> None:
    baseline = RunResult(
        suite_name="suite",
        config_name="baseline",
        results=[
            case_result("losing_case", 1.0, True, output="SAFE baseline\nline two"),
            case_result("tie_diff", 1.0, True, output="AAA"),
            case_result("unchanged_quiet", 1.0, True, output="SAME"),
        ],
    )
    candidate = RunResult(
        suite_name="suite",
        config_name="candidate",
        results=[
            case_result("losing_case", 0.0, False, output="<script>alert(1)</script>"),
            case_result("tie_diff", 1.0, True, output="BBB"),
            case_result("unchanged_quiet", 1.0, True, output="SAME"),
        ],
    )
    comparison = compare_runs(baseline, candidate, n_boot=40, seed=1)
    comparison = with_drift(
        comparison,
        measure_drift(comparison, make_embedder("hash"), embedder_name="hash"),
    )
    verdict = evaluate_gate(comparison, max_regression=1.0)
    html = render_html(comparison, verdict)
    head, marker, diff = html.partition("<h2>Output diff</h2>")
    assert marker
    assert "unchanged_quiet" in head
    assert "unchanged_quiet" not in diff
    assert "losing_case" in diff
    assert "tie_diff" in diff
    assert "SAFE baseline" in diff
    assert "line two" in diff
    assert "AAA" in diff
    assert "BBB" in diff
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in diff
    assert "<script>" not in html
    assert "<script" not in html.lower()
    assert 'class="diff-columns"' in diff
    assert "<pre>" in diff
    assert "monospace" in html
    assert comparison.drift is not None
    drifts = {item.case_id: item.drift for item in comparison.drift.per_case}
    losing = next(row for row in comparison.per_case if row.case_id == "losing_case")
    assert f"delta {losing.diff:.6f}" in diff
    assert f"drift {drifts['losing_case']:.6f}" in diff
    assert f"drift {drifts['tie_diff']:.6f}" in diff

    markdown = render_markdown(comparison, verdict)
    assert "> SAFE baseline" in markdown
    assert "> line two" in markdown
    assert "> <script>alert(1)</script>" in markdown
    assert "AAA" not in markdown
    assert "BBB" not in markdown
    assert "unchanged_quiet" in markdown
