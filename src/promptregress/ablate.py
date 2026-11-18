"""Leave-one-out ablation of named sections in a prompt template.

A section heading is one line, as ``str.splitlines`` defines a line. The
line must begin with ``##`` (no indentation). Those two characters must be
followed by one or more ASCII spaces or horizontal tabs, then a non-empty
name. Trailing ASCII spaces and tabs are not part of the name. ``#``,
``###``, a heading with no space (``##Name``), and an indented ``##`` are
body text, not headings. Closing hashes are part of the name: ``## Name ##``
is the section ``Name ##``. Names are case-sensitive. Only the ASCII space
and the horizontal tab are separators; other unicode spaces stay in the name.

Text before the first heading is the preamble. It is not a section, it is
not removed by ablation, and it is kept byte for byte in every rebuilt
template. A template that starts with a heading has an empty preamble. A
heading cannot start mid-line. A ``##`` line inside a fenced code block is
still a heading: fences are not special.

:func:`parse_sections` raises ``ValueError`` when the template has no
section, and when two headings share a name after the trim above.

Sections are parsed on the template. ``{input}`` and ``{reference}`` are
filled in later, by :func:`promptregress.runner.format_prompt`, on the
template that remains after a section is removed. A heading that arrives
only inside a case input is not a section.

:func:`ablate` scores the full template, then scores it again with each
section removed, in template order. The delta for a section is the mean of
``score(without) - score(full)`` on the suite cases. A per-case difference
within ``1e-12`` is stored as ``0``, the same tie band ``compare`` uses.
The confidence interval is :func:`promptregress.stats.paired_bootstrap_ci`
on those paired diffs. It is a marginal contribution under this suite and
provider, not a causal effect in general.

The full run and every leave-one-out run share one :class:`ResponseCache`.
A case whose rendered prompt was already stored is not sent to the provider
again. Pass ``cache`` to keep that directory. When ``cache`` is omitted, a
temporary directory is used for the call and then removed.
"""

import json
import re
import tempfile
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from promptregress.cache import ResponseCache
from promptregress.junit_xml import strip_unpaired_surrogates
from promptregress.models import RunResult, Suite
from promptregress.providers import Provider
from promptregress.runner import run_suite
from promptregress.stats import paired_bootstrap_ci

# Same tie band as compare_runs. A smaller per-case difference is stored as 0.
_TIE_EPS = 1e-12

_HEADING_RE = re.compile(r"##[ \t]+(.+?)[ \t]*")

ABLATION_NOTE = (
    "Delta is the mean of per-case score without the section, minus the "
    "full-prompt score. The interval is a bootstrap confidence interval for "
    "that paired mean on this suite and provider. It is a marginal "
    "contribution under this measurement, not causal importance in general. "
    "Sections can interact, and a section that matters on other cases can "
    "look unused here. A per-case difference within 1e-12 counts as no change, "
    "the same tie band compare uses."
)


@dataclass(frozen=True)
class PromptSection:
    """One ``##`` section and the exact template slice that holds it.

    ``text`` starts at the first character of the heading line and runs
    through the character before the next heading, or through the end of
    the template. The heading line, the body, and the original line endings
    are all in that slice.
    """

    name: str
    text: str


@dataclass(frozen=True)
class ParsedPrompt:
    """A template split into a preamble and ordered sections.

    ``render`` with nothing removed returns the original template, including
    line endings. Removing a name drops that section's slice and leaves every
    other byte unchanged.
    """

    preamble: str
    sections: tuple[PromptSection, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """Section names in template order."""
        return tuple(section.name for section in self.sections)

    def render(self, drop: Collection[str] = ()) -> str:
        """Rebuild the template, omitting any section whose name is in ``drop``.

        ``drop`` is a collection of names, not a single string. An unknown
        name raises ``ValueError``. The relative order of the kept sections
        does not change.
        """
        if isinstance(drop, str):
            raise TypeError("drop must be a collection of section names, not a string")
        known = {section.name for section in self.sections}
        requested: list[str] = []
        for name in drop:
            if not isinstance(name, str):
                raise TypeError("section names must be strings")
            requested.append(name)
        missing = sorted({name for name in requested if name not in known})
        if len(missing) == 1:
            raise ValueError(f"unknown section name: {missing[0]!r}")
        if missing:
            listed = ", ".join(repr(name) for name in missing)
            raise ValueError(f"unknown section names: {listed}")
        drop_set = set(requested)
        parts = [self.preamble]
        for section in self.sections:
            if section.name not in drop_set:
                parts.append(section.text)
        return "".join(parts)


@dataclass(frozen=True)
class SectionImpact:
    """One section after leave-one-out scoring.

    ``mean`` is the suite mean with this section removed. ``delta`` is the
    mean paired difference ``score(without) - score(full)``. ``ci_low`` and
    ``ci_high`` bound that mean. ``cases_affected`` lists case ids, in suite
    order, whose score changed by more than ``1e-12``.
    """

    name: str
    mean: float
    delta: float
    ci_low: float
    ci_high: float
    cases_affected: tuple[str, ...]


@dataclass(frozen=True)
class AblationReport:
    """Leave-one-out result for every section of one template.

    ``sections`` follows the template order. ``full_mean`` is the suite mean
    with every section kept. ``preamble_chars`` is the length of the text
    before the first heading; that text was kept in every run.
    """

    suite_name: str
    n_cases: int
    full_mean: float
    preamble_chars: int
    sections: tuple[SectionImpact, ...]
    n_boot: int
    seed: int
    alpha: float
    ci_method: str


def parse_sections(template: str) -> ParsedPrompt:
    """Split ``template`` into a preamble and named ``##`` sections.

    Raises ``TypeError`` when ``template`` is not a string. Raises
    ``ValueError`` with ``no sections found`` when no heading matches, and
    ``duplicate section name: ...`` when two headings trim to the same name.
    See the module docstring for what counts as a heading.
    """
    if not isinstance(template, str):
        raise TypeError("template must be a string")
    starts: list[tuple[int, str]] = []
    seen: set[str] = set()
    offset = 0
    for line in template.splitlines(keepends=True):
        name = _heading_name(line)
        if name is not None:
            if name in seen:
                raise ValueError(f"duplicate section name: {name!r}")
            seen.add(name)
            starts.append((offset, name))
        offset += len(line)
    if not starts:
        raise ValueError("no sections found")
    preamble = template[: starts[0][0]]
    sections: list[PromptSection] = []
    for index, (start, name) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(template)
        sections.append(PromptSection(name=name, text=template[start:end]))
    return ParsedPrompt(preamble=preamble, sections=tuple(sections))


def ablate(
    suite: Suite,
    provider: Provider,
    template: str,
    *,
    cache: ResponseCache | None = None,
    max_workers: int = 1,
    samples: int = 1,
    n_boot: int = 2000,
    seed: int = 13,
    alpha: float = 0.05,
    ci_method: str = "percentile",
) -> AblationReport:
    """Score ``template`` and each leave-one-out variant on ``suite``.

    ``n_boot``, ``seed``, ``alpha``, and ``ci_method`` are passed to
    :func:`promptregress.stats.paired_bootstrap_ci`. They are checked before
    any provider call. ``samples`` and ``max_workers`` are the runner's.
    The same cache covers the full prompt and every ablated prompt, so a
    case whose prompt text does not change is not called again.
    """
    parsed = parse_sections(template)
    if not suite.cases:
        raise ValueError("suite has no cases")
    ids = [case.id for case in suite.cases]
    if len(set(ids)) != len(ids):
        raise ValueError("suite has duplicate case ids")
    _check_bootstrap(n_boot=n_boot, seed=seed, alpha=alpha, ci_method=ci_method)
    if cache is None:
        with tempfile.TemporaryDirectory(prefix="promptregress-ablate-") as directory:
            return _ablate_parsed(
                suite,
                provider,
                parsed,
                cache=ResponseCache(directory),
                max_workers=max_workers,
                samples=samples,
                n_boot=n_boot,
                seed=seed,
                alpha=alpha,
                ci_method=ci_method,
            )
    return _ablate_parsed(
        suite,
        provider,
        parsed,
        cache=cache,
        max_workers=max_workers,
        samples=samples,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        ci_method=ci_method,
    )


def format_ablation(report: AblationReport) -> str:
    """Text report. Ends with a newline.

    Each section line has the name, the ablated mean, the delta versus the
    full prompt, the bootstrap interval, and how many cases changed. Case
    ids follow on the next line when at least one changed.
    """
    lines = [
        "ablation",
        f"suite={_clean(report.suite_name)}",
        f"n_cases={report.n_cases}",
        f"full_mean={_fmt(report.full_mean)}",
        f"preamble_chars={report.preamble_chars}",
        f"n_boot={report.n_boot}",
        f"seed={report.seed}",
        f"alpha={_fmt(report.alpha)}",
        f"ci_method={report.ci_method}",
        f"note={ABLATION_NOTE}",
    ]
    for impact in report.sections:
        lines.append(
            f"- {_clean(impact.name)} mean={_fmt(impact.mean)} "
            f"delta={_fmt(impact.delta)} "
            f"ci=[{_fmt(impact.ci_low)}, {_fmt(impact.ci_high)}] "
            f"cases_affected={len(impact.cases_affected)}"
        )
        if impact.cases_affected:
            ids = ", ".join(_clean(case_id) for case_id in impact.cases_affected)
            lines.append(f"  cases: {ids}")
    return strip_unpaired_surrogates("\n".join(lines) + "\n")


def ablation_to_dict(report: AblationReport) -> dict[str, object]:
    """JSON-ready report. Floats that are zero are positive zero."""
    return {
        "suite": _clean(report.suite_name),
        "n_cases": report.n_cases,
        "full_mean": _num(report.full_mean),
        "preamble_chars": report.preamble_chars,
        "n_boot": report.n_boot,
        "seed": report.seed,
        "alpha": _num(report.alpha),
        "ci_method": report.ci_method,
        "note": ABLATION_NOTE,
        "sections": [_section_to_dict(impact) for impact in report.sections],
    }


def write_ablation_json(payload: dict[str, object], path: str | Path) -> None:
    """Write ``payload`` as indented JSON. Rejects NaN and Infinity."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _ablate_parsed(
    suite: Suite,
    provider: Provider,
    parsed: ParsedPrompt,
    *,
    cache: ResponseCache,
    max_workers: int,
    samples: int,
    n_boot: int,
    seed: int,
    alpha: float,
    ci_method: str,
) -> AblationReport:
    full = run_suite(
        suite,
        provider,
        "full",
        prompt_template=parsed.render(),
        cache=cache,
        max_workers=max_workers,
        samples=samples,
    )
    impacts: list[SectionImpact] = []
    for section in parsed.sections:
        ablated = run_suite(
            suite,
            provider,
            f"without:{section.name}",
            prompt_template=parsed.render(drop=[section.name]),
            cache=cache,
            max_workers=max_workers,
            samples=samples,
        )
        diffs, affected = _paired_diffs(full, ablated)
        ci_low, ci_high = paired_bootstrap_ci(
            diffs,
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
            method=ci_method,
        )
        delta = sum(diffs) / len(diffs)
        impacts.append(
            SectionImpact(
                name=section.name,
                mean=_num(ablated.mean_score),
                delta=_num(delta),
                ci_low=_num(ci_low),
                ci_high=_num(ci_high),
                cases_affected=tuple(affected),
            )
        )
    return AblationReport(
        suite_name=suite.name,
        n_cases=len(suite.cases),
        full_mean=_num(full.mean_score),
        preamble_chars=len(parsed.preamble),
        sections=tuple(impacts),
        n_boot=n_boot,
        seed=seed,
        alpha=_num(float(alpha)),
        ci_method=ci_method,
    )


def _paired_diffs(full: RunResult, ablated: RunResult) -> tuple[list[float], list[str]]:
    """Paired ``ablated - full`` diffs in suite order, plus ids that moved."""
    full_ids = [result.case_id for result in full.results]
    ablated_ids = [result.case_id for result in ablated.results]
    if full_ids != ablated_ids:
        raise ValueError("ablated run cases do not match the full run")
    diffs: list[float] = []
    affected: list[str] = []
    for base, other in zip(full.results, ablated.results, strict=True):
        raw = other.score - base.score
        if abs(raw) <= _TIE_EPS:
            diffs.append(0.0)
        else:
            diffs.append(raw)
            affected.append(other.case_id)
    return diffs, affected


def _check_bootstrap(*, n_boot: int, seed: int, alpha: float, ci_method: str) -> None:
    """Reject bad bootstrap arguments before any provider call.

    The check is one call of the shared helper on a constant, so the rules
    stay the helper's rules.
    """
    paired_bootstrap_ci([0.0], n_boot=n_boot, seed=seed, alpha=alpha, method=ci_method)


def _heading_name(line: str) -> str | None:
    """Return the section name when ``line`` is a heading, else ``None``."""
    body = line[:-1] if line.endswith("\n") else line
    if body.endswith("\r"):
        body = body[:-1]
    match = _HEADING_RE.fullmatch(body)
    if match is None:
        return None
    name = match.group(1).strip(" \t")
    if name == "":
        return None
    return name


def _section_to_dict(impact: SectionImpact) -> dict[str, object]:
    return {
        "name": _clean(impact.name),
        "mean": _num(impact.mean),
        "delta": _num(impact.delta),
        "ci_low": _num(impact.ci_low),
        "ci_high": _num(impact.ci_high),
        "cases_affected": [_clean(case_id) for case_id in impact.cases_affected],
    }


def _clean(text: str) -> str:
    return strip_unpaired_surrogates(text)


def _num(value: float) -> float:
    """Return ``value`` as a float, with signed zero folded to ``0.0``."""
    number = float(value)
    if number == 0.0:
        return 0.0
    return number


def _fmt(value: float) -> str:
    """Six decimal places, and ``0.000000`` rather than ``-0.000000``."""
    return f"{_num(value):.6f}"
