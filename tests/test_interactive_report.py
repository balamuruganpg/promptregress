"""Interactive comparison HTML: inline script, filters, and escaped output."""

from dataclasses import replace
from html.parser import HTMLParser

from helpers import case_result, run_with
from promptregress.cli import run_demo
from promptregress.gate import evaluate_gate
from promptregress.models import RunResult
from promptregress.report import render_html
from promptregress.stats import compare_runs


class _ScriptTags(HTMLParser):
    """Count real script elements. Escaped text must not add one."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.srcs: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "script":
            return
        self.count += 1
        src = None
        for key, value in attrs:
            if key.lower() == "src":
                src = value
        self.srcs.append(src)


def _scripts(html: str) -> _ScriptTags:
    parser = _ScriptTags()
    parser.feed(html)
    parser.close()
    return parser


def _inline_script(html: str) -> str:
    start = html.lower().find("<script")
    assert start != -1
    body = html.find(">", start) + 1
    end = html.lower().find("</script>", body)
    assert end != -1
    return html[body:end]


def _comparison():
    tags = {"password_reset": ["account", "security"], "hours": ["support"]}
    baseline = run_with(
        [("password_reset", 1.0, True), ("hours", 0.4, False), ("beta", 0.2, False)],
        "baseline",
        tags=tags,
    )
    # The loss carries the hostile output. The tie fails on both sides.
    # beta is a win. case_input is what the runner records for the search box.
    candidate = RunResult(
        suite_name="suite",
        config_name="candidate",
        results=[
            replace(
                case_result("password_reset", 0.0, False, output="<script>alert(1)</script>"),
                case_input="How do I reset my password?",
            ),
            case_result("hours", 0.4, False, output="same"),
            replace(
                case_result("beta", 0.9, True, output="better", tags=["billing"]),
                case_input="<script>typed</script>",
            ),
        ],
    )
    return compare_runs(baseline, candidate, n_boot=40, seed=1)


def test_report_has_inline_controls_and_sort_attributes() -> None:
    comparison = _comparison()
    html = render_html(comparison, evaluate_gate(comparison, max_regression=1.0))
    scripts = _scripts(html)
    assert scripts.count == 1
    assert scripts.srcs == [None]
    assert 'id="case-search"' in html
    assert 'id="case-outcome"' in html
    assert 'id="case-tag"' in html
    assert 'placeholder="Case id or input"' in html
    for value, label in (
        ("all", "all"),
        ("win", "wins"),
        ("loss", "losses"),
        ("tie", "ties"),
        ("failure", "failures"),
    ):
        assert f'value="{value}">{label}</option>' in html or (
            f'value="{value}" selected>{label}</option>' in html
        )
    for key in ("case-id", "baseline-score", "candidate-score", "delta", "drift", "latency"):
        assert f'data-sort="{key}"' in html
        assert f'data-{key}="' in html
    assert 'type="button"' in html
    assert 'aria-sort="none"' in html
    assert "<button" in html
    assert 'class="donut"' in html
    assert 'class="means"' in html
    assert "<svg" in html
    script = _inline_script(html)
    for forbidden in (
        "innerHTML",
        "outerHTML",
        "document.write",
        "insertAdjacentHTML",
        "http://",
        "https://",
    ):
        assert forbidden not in script
    assert "http://" not in html
    assert "https://" not in html
    assert "<link" not in html.lower()
    assert html.count('<tbody class="case"') == len(comparison.per_case)
    assert html.count("<details") == len(comparison.per_case)


def test_script_output_is_escaped_and_does_not_add_a_script_element() -> None:
    html = render_html(_comparison(), evaluate_gate(_comparison(), max_regression=1.0))
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;script&gt;typed&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "<script>typed</script>" not in html
    scripts = _scripts(html)
    assert scripts.count == 1
    assert scripts.srcs == [None]
    assert 'data-search="password_reset How do I reset my password?"' in html
    assert 'value="billing"' in html
    assert 'value="account"' in html
    assert 'value="security"' in html
    assert 'value="support"' in html
    assert 'data-failure="true"' in html
    assert 'data-failure="false"' in html
    assert 'class="diff-columns"' in html
    assert "<h2>Output diff</h2>" not in html


def test_demo_counter_matches_the_case_count() -> None:
    comparison, verdict = run_demo()
    html = render_html(comparison, verdict)
    count = len(comparison.per_case)
    assert count == comparison.wins + comparison.losses + comparison.ties
    assert f'data-shown="{count}"' in html
    assert f'data-total="{count}"' in html
    assert (
        f'<span id="case-shown">{count}</span> of <span id="case-total">{count}</span> cases'
        in html
    )
    assert html.count('<tbody class="case"') == count
    assert "How do I reset my password?" in html
    assert "http://" not in html
    assert "https://" not in html
    scripts = _scripts(html)
    assert scripts.count == 1
    assert scripts.srcs == [None]
