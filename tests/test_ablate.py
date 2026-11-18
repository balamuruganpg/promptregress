"""Prompt-section ablation: parser, mock end-to-end, and the CLI."""

import json
from pathlib import Path

import pytest

import promptregress.ablate as ablate_mod
from promptregress.ablate import ablate, format_ablation, parse_sections
from promptregress.cache import ResponseCache
from promptregress.cli import main
from promptregress.loading import load_suite
from promptregress.models import AssertionSpec, Case, Suite
from promptregress.providers import MockProvider
from promptregress.runner import format_prompt, run_suite
from promptregress.stats import paired_bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "ablate"


def _case(case_id: str, text: str, reference: str) -> Case:
    return Case(
        id=case_id,
        input=text,
        reference=reference,
        assertions=[AssertionSpec(kind="exact_match")],
    )


def _rules() -> list[tuple[str, str]]:
    data = json.loads((EXAMPLE / "rules.json").read_text(encoding="utf-8"))
    return [(str(item[0]), str(item[1])) for item in data]


def _example_template() -> str:
    return (EXAMPLE / "template.txt").read_text(encoding="utf-8")


def test_removal_rebuilds_exactly() -> None:
    original = "Intro line.\n\n## Alpha\nbody a\n\n## Beta\nbody b\n"
    parsed = parse_sections(original)
    assert parsed.preamble == "Intro line.\n\n"
    assert parsed.names == ("Alpha", "Beta")
    assert parsed.sections[0].text == "## Alpha\nbody a\n\n"
    assert parsed.sections[1].text == "## Beta\nbody b\n"
    assert parsed.render() == original
    assert parsed.render(drop=["Alpha"]) == "Intro line.\n\n## Beta\nbody b\n"
    assert parsed.render(drop=["Beta"]) == "Intro line.\n\n## Alpha\nbody a\n\n"
    assert parsed.render(drop=["Alpha", "Beta"]) == "Intro line.\n\n"
    assert parsed.render(drop=[]) == original


def test_strict_headings_leave_non_headings_in_place() -> None:
    text = "##NoSpace\n# Title\n### Three\n  ## indented\n##   \nsee ## not\n## Real\nbody\n"
    parsed = parse_sections(text)
    assert parsed.names == ("Real",)
    assert parsed.preamble.startswith("##NoSpace\n")
    assert "### Three\n" in parsed.preamble
    assert "  ## indented\n" in parsed.preamble
    assert "##   \n" in parsed.preamble
    assert "see ## not\n" in parsed.preamble
    assert parsed.render(drop=["Real"]) == parsed.preamble

    nested = "## Alpha\n### Nested\nstill alpha\n## Beta\nend\n"
    parsed_nested = parse_sections(nested)
    assert parsed_nested.names == ("Alpha", "Beta")
    assert "### Nested\n" in parsed_nested.sections[0].text

    assert parse_sections("## Name ##\nbody\n").names == ("Name ##",)
    assert parse_sections("## Policy\na\n## policy\nb\n").names == ("Policy", "policy")


def test_duplicate_section_names() -> None:
    with pytest.raises(ValueError, match="duplicate section name: 'Alpha'"):
        parse_sections("## Alpha\n## Alpha\n")
    with pytest.raises(ValueError, match="duplicate section name: 'Alpha'"):
        parse_sections("## Alpha\n## Alpha  \n")
    with pytest.raises(ValueError, match="duplicate section name: 'Alpha'"):
        parse_sections("## Alpha \n## \tAlpha\t\n")


def test_no_sections() -> None:
    for text in ("", "just text\n", "# Title\n", "### Nope\n", "##NoSpace\n", "##   \n"):
        with pytest.raises(ValueError, match="no sections found"):
            parse_sections(text)
    with pytest.raises(ValueError, match="no sections found"):
        parse_sections("  ## indented\n")


def test_crlf_and_eof_round_trip() -> None:
    original = "Intro\r\n\r\n## Alpha\r\nbody\r\n\r\n## Beta\r\nmore\r\n"
    parsed = parse_sections(original)
    assert parsed.names == ("Alpha", "Beta")
    assert parsed.render() == original
    assert parsed.render(drop=["Alpha"]) == "Intro\r\n\r\n## Beta\r\nmore\r\n"

    at_eof = "## Only\nbody"
    parsed_eof = parse_sections(at_eof)
    assert parsed_eof.preamble == ""
    assert parsed_eof.render() == at_eof
    assert parsed_eof.sections[0].text == at_eof


def test_unknown_drop_and_string_drop() -> None:
    parsed = parse_sections("## Alpha\nbody\n## Beta\nmore\n")
    with pytest.raises(ValueError, match="unknown section name: 'Nope'"):
        parsed.render(drop=["Nope"])
    with pytest.raises(ValueError, match="unknown section names: "):
        parsed.render(drop=["Zed", "Nope"])
    with pytest.raises(TypeError, match="not a string"):
        parsed.render(drop="Alpha")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="section names must be strings"):
        parsed.render(drop=["Alpha", 1])  # type: ignore[list-item]


def test_placeholders_are_filled_after_removal() -> None:
    template = "Intro\n\n{input}\n## Tone\nBe warm.\n"
    parsed = parse_sections(template)
    assert parsed.names == ("Tone",)
    case = Case(
        id="hours",
        input="## Policy\nPOLICY::hours",
        reference=None,
        assertions=[],
    )
    # Headings that arrive inside {input} are not sections. Tone is removed
    # first, then the case input is substituted into what remains.
    prompt = format_prompt(parsed.render(drop=["Tone"]), case)
    assert prompt == "Intro\n\n## Policy\nPOLICY::hours\n"
    assert "## Tone" not in prompt

    both = "Q: {input}\n\n## Policy\nPOLICY::{input}\n## Tone\nBe warm.\n"
    filled = format_prompt(parse_sections(both).render(drop=["Tone"]), _case("h", "hours", "x"))
    assert filled == "Q: hours\n\n## Policy\nPOLICY::hours\n"


def test_example_ablation_load_bearing_and_noop() -> None:
    source = Path(ablate_mod.__file__).read_text(encoding="utf-8")
    assert "paired_bootstrap_ci" in source
    assert "random.Random" not in source
    assert "randrange" not in source

    suite = load_suite(EXAMPLE / "suite.json")
    template = _example_template()
    assert parse_sections(template).render() == template
    provider = MockProvider(rules=_rules(), default="")
    report = ablate(suite, provider, template)
    assert report.suite_name == "ablation-demo"
    assert report.n_cases == 4
    assert report.full_mean == 1.0
    assert report.preamble_chars == len(parse_sections(template).preamble)
    by_name = {item.name: item for item in report.sections}
    assert tuple(by_name) == ("Policy", "Tone")

    policy = by_name["Policy"]
    tone = by_name["Tone"]
    assert policy.mean == 0.0
    assert policy.delta == -1.0
    assert policy.cases_affected == tuple(case.id for case in suite.cases)
    assert (policy.ci_low, policy.ci_high) == paired_bootstrap_ci([-1.0, -1.0, -1.0, -1.0])
    assert policy.ci_high < 0.0

    assert tone.mean == 1.0
    assert tone.delta == 0.0
    assert tone.cases_affected == ()
    assert tone.ci_low <= 0.0 <= tone.ci_high
    assert (tone.ci_low, tone.ci_high) == paired_bootstrap_ci([0.0, 0.0, 0.0, 0.0])

    text = format_ablation(report)
    assert text.startswith("ablation\n")
    assert text.endswith("\n")
    assert "delta=-1.000000" in text
    assert "delta=0.000000" in text
    assert "cases_affected=4" in text
    assert "cases_affected=0" in text
    assert "cases: hours, refund, shipping, warranty" in text


def test_partial_section_ci_matches_bootstrap_helper() -> None:
    suite = Suite(
        name="partial",
        cases=[
            _case("a", "keep", "yes-keep"),
            _case("b", "other", "yes-other"),
        ],
    )
    template = "## Policy\nPOLICY::{input}\n\n## Hint\nHINT::{input}\n"
    rules = [("POLICY::keep", "yes-keep"), ("HINT::other", "yes-other")]
    report = ablate(suite, MockProvider(rules=rules, default=""), template)
    by_name = {item.name: item for item in report.sections}
    policy = by_name["Policy"]
    hint = by_name["Hint"]
    assert report.full_mean == 1.0
    assert policy.mean == 0.5
    assert policy.delta == -0.5
    assert policy.cases_affected == ("a",)
    assert hint.delta == -0.5
    assert hint.cases_affected == ("b",)
    assert (policy.ci_low, policy.ci_high) == paired_bootstrap_ci([-1.0, 0.0])
    assert (hint.ci_low, hint.ci_high) == paired_bootstrap_ci([0.0, -1.0])

    bca = ablate(suite, MockProvider(rules=rules, default=""), template, ci_method="bca")
    bca_policy = {item.name: item for item in bca.sections}["Policy"]
    assert (bca_policy.ci_low, bca_policy.ci_high) == paired_bootstrap_ci([-1.0, 0.0], method="bca")


def test_cache_reuses_full_prompt_and_repeated_cases(tmp_path: Path) -> None:
    suite = load_suite(EXAMPLE / "suite.json")
    template = _example_template()
    provider = MockProvider(rules=_rules(), default="")
    cache = ResponseCache(tmp_path / "cache")
    run_suite(suite, provider, "seed", prompt_template=template, cache=cache)
    assert provider.call_count == 4
    report = ablate(suite, provider, template, cache=cache, n_boot=30)
    # The seeded full prompt is reused (4 hits). Removing Policy also removes
    # {input}, so all four cases share one prompt: one miss and three hits.
    # Removing Tone leaves {input}, so those four prompts are new.
    assert provider.call_count == 9
    assert cache.hits == 7
    by_name = {item.name: item for item in report.sections}
    assert by_name["Policy"].delta == -1.0
    assert by_name["Tone"].delta == 0.0
    assert by_name["Tone"].ci_low <= 0.0 <= by_name["Tone"].ci_high

    repeated = Suite(
        name="repeated",
        cases=[
            _case("a", "hours", "We are open 9 to 5."),
            _case("b", "hours", "We are open 9 to 5."),
        ],
    )
    provider_repeated = MockProvider(rules=_rules(), default="")
    cache_repeated = ResponseCache(tmp_path / "cache-repeated")
    repeated_template = "## Policy\nPOLICY::{input}\n## Tone\nBe warm.\n"
    repeated_report = ablate(
        repeated,
        provider_repeated,
        repeated_template,
        cache=cache_repeated,
        n_boot=20,
    )
    # Three distinct prompts, each shared by the two cases.
    assert provider_repeated.call_count == 3
    assert cache_repeated.hits == 3
    repeated_by_name = {item.name: item for item in repeated_report.sections}
    assert repeated_by_name["Policy"].cases_affected == ("a", "b")
    assert repeated_by_name["Tone"].cases_affected == ()
    assert repeated_by_name["Tone"].delta == 0.0


def test_workers_and_samples_keep_the_same_deltas() -> None:
    suite = load_suite(EXAMPLE / "suite.json")
    template = _example_template()
    rules = _rules()

    def run(workers: int, samples: int) -> object:
        provider = MockProvider(rules=rules, default="")
        return ablate(
            suite,
            provider,
            template,
            max_workers=workers,
            samples=samples,
            n_boot=40,
        ).sections

    base = run(1, 1)
    assert run(2, 1) == base
    assert run(1, 2) == base


def test_validation_does_not_call_the_provider() -> None:
    suite = load_suite(EXAMPLE / "suite.json")
    template = _example_template()
    provider = MockProvider(rules=_rules(), default="")
    with pytest.raises(ValueError, match="n_boot"):
        ablate(suite, provider, template, n_boot=0)
    with pytest.raises(ValueError, match="no sections found"):
        ablate(suite, provider, "plain")
    with pytest.raises(ValueError, match="duplicate section name"):
        ablate(suite, provider, "## A\n## A\n")
    with pytest.raises(ValueError, match="no cases"):
        ablate(Suite(name="empty", cases=[]), provider, "## A\nbody\n")
    duplicate_ids = Suite(
        name="dup",
        cases=[
            _case("a", "hours", "We are open 9 to 5."),
            _case("a", "refund", "Refunds take 5 days."),
        ],
    )
    with pytest.raises(ValueError, match="duplicate case"):
        ablate(duplicate_ids, provider, template)
    with pytest.raises(TypeError, match="template must be a string"):
        ablate(suite, provider, None)  # type: ignore[arg-type]
    assert provider.call_count == 0


def _ablate_argv(*extra: str) -> list[str]:
    return [
        "ablate",
        "--suite",
        str(EXAMPLE / "suite.json"),
        "--provider",
        "mock",
        "--mock-rules",
        (EXAMPLE / "rules.json").read_text(encoding="utf-8"),
        "--template",
        _example_template(),
        *extra,
    ]


def test_cli_text_json_out_and_cache(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(_ablate_argv())
    assert code == 0
    text = capsys.readouterr().out
    assert text.startswith("ablation\n")
    assert "suite=ablation-demo" in text
    assert "full_mean=1.000000" in text
    assert "delta=-1.000000" in text
    assert "ci=[-1.000000, -1.000000]" in text
    assert "cases_affected=4" in text
    assert "cases: hours, refund, shipping, warranty" in text
    assert "delta=0.000000" in text
    assert "ci=[0.000000, 0.000000]" in text
    assert "cases_affected=0" in text
    assert "not causal importance" in text

    out = tmp_path / "nested" / "ablation.json"
    code = main(_ablate_argv("--json", "--out", str(out)))
    assert code == 0
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert payload["full_mean"] == 1.0
    assert payload["ci_method"] == "percentile"
    assert payload["n_boot"] == 2000
    assert payload["seed"] == 13
    sections = {row["name"]: row for row in payload["sections"]}
    assert sections["Policy"]["mean"] == 0.0
    assert sections["Policy"]["delta"] == -1.0
    assert sections["Policy"]["ci_low"] == -1.0
    assert sections["Policy"]["ci_high"] == -1.0
    assert sections["Policy"]["cases_affected"] == ["hours", "refund", "shipping", "warranty"]
    assert sections["Tone"]["mean"] == 1.0
    assert sections["Tone"]["delta"] == 0.0
    assert sections["Tone"]["ci_low"] <= 0.0 <= sections["Tone"]["ci_high"]
    assert sections["Tone"]["cases_affected"] == []

    cache_dir = tmp_path / "cache"
    code = main(_ablate_argv("--cache", str(cache_dir), "--n-boot", "20"))
    assert code == 0
    capsys.readouterr()
    # 4 full prompts, 1 shared prompt with Policy removed, 4 with Tone removed.
    assert len(list(cache_dir.glob("*.json"))) == 9


def test_cli_validation_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite = str(EXAMPLE / "suite.json")
    code = main(["ablate", "--suite", suite, "--provider", "mock", "--template", "no headings"])
    assert code == 2
    assert "no sections found" in capsys.readouterr().err

    code = main(["ablate", "--suite", suite, "--provider", "mock", "--template", "## A\n## A\n"])
    assert code == 2
    assert "duplicate section name" in capsys.readouterr().err

    code = main(
        [
            "ablate",
            "--suite",
            str(tmp_path / "missing.json"),
            "--provider",
            "mock",
            "--template",
            "## A\nx\n",
        ]
    )
    assert code == 2
    assert "error:" in capsys.readouterr().err

    code = main(
        [
            "ablate",
            "--suite",
            suite,
            "--provider",
            "mock",
            "--template",
            "## A\nx\n",
            "--n-boot",
            "0",
        ]
    )
    assert code == 2
    assert "n_boot" in capsys.readouterr().err


def test_cli_help_and_required_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as help_exit:
        main(["ablate", "--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "--suite" in help_text
    assert "--template" in help_text
    assert "--provider" in help_text
    assert "95%" in help_text
    assert "Bootstrap" in help_text

    with pytest.raises(SystemExit) as missing_suite:
        main(["ablate", "--provider", "mock", "--template", "## A\n"])
    assert missing_suite.value.code == 2
    with pytest.raises(SystemExit) as missing_template:
        main(["ablate", "--suite", "suite.json", "--provider", "mock"])
    assert missing_template.value.code == 2
