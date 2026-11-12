"""Adversarial surface variants of an eval suite.

Each operator is a pure function of the case input and an integer seed. It
returns the input unchanged when it does not apply. Nothing reads the
process-global random stream. Choices use ``seed % n`` (and ``seed // n``
for the distractor side). There is no shuffle, so there is no RNG.

The synthesizer mixes the caller's seed with the case id, the operator
name, and the repetition index through SHA-256, then passes that integer
in. The same seed always rebuilds the same suite.

These edits test surface form. They do not add facts or new labels.
Keeping the correct answer is assumed, not proven: a wrapper, a synonym,
or a homoglyph can change the task for some prompts and scorers.
"""

import csv
import hashlib
import io
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from promptregress.loading import load_suite
from promptregress.models import AssertionSpec, Case, ScoringConfig, Suite

# One-way whole-word map. Keys and values are lowercase; replacement keeps
# the source word's case (upper, lower, or capitalised). Not a thesaurus:
# each pair is a conservative near-paraphrase, and the reverse is not applied.
SYNONYM_MAP: dict[str, str] = {
    "quickly": "rapidly",
    "large": "big",
    "little": "small",
    "begin": "start",
    "purchase": "buy",
    "difficult": "hard",
    "simple": "easy",
    "glad": "happy",
    "assist": "help",
    "additional": "extra",
}

# Sentence punctuation only. A mark between two alphanumeric characters
# (decimals, ``e.g``) is left alone. Commas are left for number_format.
PUNCTUATION_SWAPS: dict[str, str] = {
    ".": "?",
    "?": ".",
    "!": ".",
    ";": ",",
    ":": ";",
}

INSTRUCTION_WRAPPERS: tuple[str, ...] = (
    "Please answer: {text}",
    "Q: {text} A:",
)

# Irrelevant filler. Inserting one of these is not a new question.
DISTRACTORS: tuple[str, ...] = (
    "The note about the weather is not part of the question.",
    "An unrelated sentence about office plants sits here.",
    "This filler line does not add a fact you should use.",
    "A spare remark about Tuesdays is beside the point.",
)

# Latin letter to a single Cyrillic lookalike. Not a full confusable table.
HOMOGLYPHS: dict[str, str] = {
    "a": "\u0430",
    "e": "\u0435",
    "o": "\u043e",
    "p": "\u0440",
    "c": "\u0441",
    "x": "\u0445",
    "A": "\u0410",
    "E": "\u0415",
    "O": "\u041e",
    "P": "\u0420",
    "C": "\u0421",
    "X": "\u0425",
}

_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9.,])"
    r"(?P<int>\d{1,3}(?:,\d{3})+|\d{4,})"
    r"(?P<frac>\.\d+)?"
    r"(?![A-Za-z0-9,])"
)

_SYNONYM_RE = re.compile(
    r"(?<![A-Za-z])("
    + "|".join(re.escape(word) for word in sorted(SYNONYM_MAP, key=len, reverse=True))
    + r")(?![A-Za-z])",
    re.IGNORECASE,
)

Operator = Callable[[str, int], str]
OPERATORS: dict[str, Operator] = {}

DEFAULT_SEED = 13
DEFAULT_PER_CASE = 1


def _register(fn: Operator) -> Operator:
    OPERATORS[fn.__name__] = fn
    return fn


def _require_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")


def _require_per_case(per_case: int) -> None:
    if isinstance(per_case, bool) or not isinstance(per_case, int) or per_case < 1:
        raise ValueError("per-case must be an integer >= 1")


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source.islower():
        return replacement.lower()
    if len(source) > 1 and source[0].isupper() and source[1:].islower():
        return replacement[0].upper() + replacement[1:].lower()
    return replacement


def _group_thousands(digits: str) -> str:
    parts: list[str] = []
    rest = digits
    while rest:
        parts.append(rest[-3:])
        rest = rest[:-3]
    return ",".join(reversed(parts))


def _rewrite_number(match: re.Match[str]) -> str:
    integer = match.group("int")
    fraction = match.group("frac") or ""
    if integer[0] == "0":
        return match.group(0)
    if "," in integer:
        return integer.replace(",", "") + fraction
    return _group_thousands(integer) + fraction


@_register
def whitespace_casing(text: str, seed: int = 0) -> str:
    """Flip one ASCII letter and double one whitespace character.

    ``seed`` selects the letter (``seed % n``) and the whitespace
    character. Skips when the text has no ASCII letter and no whitespace.
    """
    _require_seed(seed)
    letters = [i for i, char in enumerate(text) if ("A" <= char <= "Z") or ("a" <= char <= "z")]
    spaces = [i for i, char in enumerate(text) if char.isspace()]
    if not letters and not spaces:
        return text
    chars = list(text)
    if letters:
        index = letters[seed % len(letters)]
        chars[index] = chars[index].swapcase()
    if spaces:
        index = spaces[seed % len(spaces)]
        chars.insert(index, chars[index])
    return "".join(chars)


@_register
def punctuation(text: str, seed: int = 0) -> str:
    """Swap ``. ? ! ; :`` with the fixed map in :data:`PUNCTUATION_SWAPS`.

    A mark with an alphanumeric character on both sides is left alone, so
    ``1.5`` stays. ``seed`` does not change the result. Skips when no mark
    qualifies.
    """
    _require_seed(seed)
    chars = list(text)
    changed = False
    for index, char in enumerate(chars):
        replacement = PUNCTUATION_SWAPS.get(char)
        if replacement is None:
            continue
        before = index > 0 and chars[index - 1].isalnum()
        after = index + 1 < len(chars) and chars[index + 1].isalnum()
        if before and after:
            continue
        chars[index] = replacement
        changed = True
    if not changed:
        return text
    return "".join(chars)


@_register
def synonym(text: str, seed: int = 0) -> str:
    """Replace whole words from :data:`SYNONYM_MAP`.

    Matching is case-insensitive and the replacement keeps the source
    word's case. A longer word that merely contains a key (``beginning``)
    is not a match. ``seed`` does not change the result. Skips when no
    key matches.
    """
    _require_seed(seed)

    def repl(match: re.Match[str]) -> str:
        word = match.group(0)
        mapped = SYNONYM_MAP[word.casefold()]
        return _match_case(word, mapped)

    updated, count = _SYNONYM_RE.subn(repl, text)
    if count == 0:
        return text
    return updated


@_register
def instruction_rephrase(text: str, seed: int = 0) -> str:
    """Wrap the text as ``Please answer: …`` or ``Q: … A:``.

    ``seed % 2`` selects the wrapper (even: please, odd: Q/A). Skips when
    the text is empty or only whitespace.
    """
    _require_seed(seed)
    if text.strip() == "":
        return text
    template = INSTRUCTION_WRAPPERS[seed % len(INSTRUCTION_WRAPPERS)]
    return template.format(text=text)


@_register
def distractor(text: str, seed: int = 0) -> str:
    """Insert one sentence from :data:`DISTRACTORS`.

    ``seed % 4`` selects the sentence. ``(seed // 4) % 2`` selects the
    side: even prepends, odd appends. Skips when the text is empty or
    only whitespace.
    """
    _require_seed(seed)
    if text.strip() == "":
        return text
    sentence = DISTRACTORS[seed % len(DISTRACTORS)]
    if (seed // len(DISTRACTORS)) % 2 == 0:
        return f"{sentence} {text}"
    return f"{text} {sentence}"


@_register
def homoglyph(text: str, seed: int = 0) -> str:
    """Replace one Latin letter with a Cyrillic lookalike.

    The letters are the keys of :data:`HOMOGLYPHS`. ``seed`` selects which
    occurrence. Skips when none of those letters appear.
    """
    _require_seed(seed)
    indexes = [i for i, char in enumerate(text) if char in HOMOGLYPHS]
    if not indexes:
        return text
    index = indexes[seed % len(indexes)]
    char = text[index]
    return text[:index] + HOMOGLYPHS[char] + text[index + 1 :]


@_register
def number_format(text: str, seed: int = 0) -> str:
    """Swap a safe thousands form: ``1,000`` with ``1000``.

    A decimal fraction glued to that number is kept (``1,000.50`` becomes
    ``1000.50``, and the reverse). Other numbers are left alone: fewer
    than four digits with no thousands commas, a leading zero, a number
    next to a letter, and European dot-grouping such as ``1.000``.
    ``seed`` does not change the result. Skips when nothing qualifies.
    """
    _require_seed(seed)
    return _NUMBER_RE.sub(_rewrite_number, text)


OPERATOR_NAMES: tuple[str, ...] = tuple(OPERATORS)


def apply_operator(name: str, text: str, seed: int = 0) -> str:
    """Run one named operator. Raises ``ValueError`` if the name is unknown."""
    if name not in OPERATORS:
        raise _unknown_operator(name)
    return OPERATORS[name](text, seed)


def parse_operator_list(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated operator list.

    ``None`` selects every operator, in :data:`OPERATOR_NAMES` order.
    """
    if raw is None:
        return OPERATOR_NAMES
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    return _check_operator_names(names)


def variant_seed(seed: int, case_id: str, operator: str, index: int) -> int:
    """Stable per-variant seed.

    Mixes ``seed``, the case id, the operator name, and the repetition
    index with SHA-256. The process hash salt is not used. The result is
    a non-negative integer below ``2**63``.
    """
    _require_seed(seed)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("variant index must be an integer >= 0")
    payload = f"{seed}\0{case_id}\0{operator}\0{index}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def synthesize_suite(
    suite: Suite,
    *,
    operators: Sequence[str] | None = None,
    per_case: int = DEFAULT_PER_CASE,
    seed: int = DEFAULT_SEED,
) -> Suite:
    """Build a suite of surface variants. The input suite is not modified.

    Case order is the source order, then operator order, then repetition
    index ``0 .. per_case-1``. Each variant id is ``{id}__{operator}`` when
    ``per_case`` is 1, and ``{id}__{operator}__{index}`` otherwise.
    Metadata gains ``source_case_id``, ``operator``, ``variant_index``, and
    ``changed`` (false when the operator skipped). References, assertions,
    tags, and scoring weights are copied.
    """
    _require_per_case(per_case)
    _require_seed(seed)
    names = _check_operator_names(None if operators is None else operators)
    if not suite.cases:
        raise ValueError(f"Suite {suite.name!r} is empty")
    cases: list[Case] = []
    seen: set[str] = set()
    for case in suite.cases:
        for name in names:
            for index in range(per_case):
                mixed = variant_seed(seed, case.id, name, index)
                updated = apply_operator(name, case.input, mixed)
                new_id = _variant_id(case.id, name, index, per_case)
                if new_id in seen:
                    raise ValueError(f"duplicate synthesized case id: {new_id}")
                seen.add(new_id)
                metadata = dict(case.metadata)
                metadata["source_case_id"] = case.id
                metadata["operator"] = name
                metadata["variant_index"] = index
                metadata["changed"] = updated != case.input
                cases.append(
                    replace(
                        case,
                        id=new_id,
                        input=updated,
                        tags=list(case.tags),
                        metadata=metadata,
                        assertions=list(case.assertions),
                        source_line=None,
                    )
                )
    description = _synth_description(suite)
    return Suite(
        name=f"{suite.name}-synth",
        description=description,
        cases=cases,
        scoring=ScoringConfig(weights=dict(suite.scoring.weights)),
    )


def write_suite(suite: Suite, path: str | Path) -> None:
    """Write ``suite`` in the format selected by the path suffix.

    ``.json`` and ``.yaml`` / ``.yml`` write the suite document the loader
    expects, including scoring weights when they are set. ``.jsonl`` writes
    one case object per line (the loader names that suite from the file
    stem and does not store scoring weights). ``.csv`` writes columns
    ``id``, ``input``, and ``reference``; the loader then attaches
    ``exact_match`` and drops tags, metadata, and other assertions.

    Raises ``ValueError`` for an unsupported suffix, an empty suite, or a
    path that cannot be written. Parent directories are not created.
    """
    if not suite.cases:
        raise ValueError(f"Suite {suite.name!r} is empty")
    target = Path(path)
    if target.exists() and target.is_dir():
        raise ValueError(f"cannot write suite to {target}: path is a directory")
    suffix = target.suffix.lower()
    text = _render_suite(suite, suffix)
    try:
        target.write_text(text, encoding="utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"cannot write suite to {target}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"cannot write suite to {target}: {exc}") from exc


def synth_to_path(
    suite_path: str | Path,
    out_path: str | Path,
    *,
    operators: str | None = None,
    per_case: int = DEFAULT_PER_CASE,
    seed: int = DEFAULT_SEED,
) -> str:
    """Load a suite, write variants to a new file, and return a text summary.

    The source file is only read. ``--out`` must be a different path.
    """
    source = Path(suite_path)
    target = Path(out_path)
    if _same_path(source, target):
        raise ValueError("--out must be a different path from --suite")
    names = parse_operator_list(operators)
    _require_per_case(per_case)
    _require_seed(seed)
    suite = load_suite(source)
    synthesized = synthesize_suite(suite, operators=names, per_case=per_case, seed=seed)
    write_suite(synthesized, target)
    changed = sum(1 for case in synthesized.cases if case.metadata.get("changed") is True)
    return format_synth(
        suite_path=str(suite_path),
        out_path=str(out_path),
        in_cases=len(suite.cases),
        out_cases=len(synthesized.cases),
        changed=changed,
        operators=names,
        per_case=per_case,
        seed=seed,
    )


def format_synth(
    *,
    suite_path: str,
    out_path: str,
    in_cases: int,
    out_cases: int,
    changed: int,
    operators: Sequence[str],
    per_case: int,
    seed: int,
) -> str:
    """Plain-text summary of one synth run. Ends with a newline."""
    lines = [
        "synth",
        f"suite={suite_path}",
        f"out={out_path}",
        f"in_cases={in_cases}",
        f"out_cases={out_cases}",
        f"changed={changed}",
        f"operators={','.join(operators)}",
        f"per_case={per_case}",
        f"seed={seed}",
    ]
    return "\n".join(lines) + "\n"


def case_to_dict(case: Case) -> dict[str, Any]:
    """One case in the JSON/YAML shape :func:`load_suite` reads."""
    return {
        "id": case.id,
        "input": case.input,
        "reference": case.reference,
        "tags": list(case.tags),
        "metadata": dict(case.metadata),
        "assertions": [_assertion_to_dict(item) for item in case.assertions],
    }


def suite_to_dict(suite: Suite) -> dict[str, Any]:
    """Suite document. Omits ``scoring`` when every weight is the default."""
    payload: dict[str, Any] = {
        "name": suite.name,
        "description": suite.description,
    }
    if suite.scoring.weights:
        payload["scoring"] = {"weights": dict(suite.scoring.weights)}
    payload["cases"] = [case_to_dict(case) for case in suite.cases]
    return payload


def _check_operator_names(operators: Sequence[str] | None) -> tuple[str, ...]:
    if operators is None:
        return OPERATOR_NAMES
    if isinstance(operators, str):
        raise ValueError("operators must be a list of names, not a string")
    names: list[str] = []
    seen: set[str] = set()
    for raw in operators:
        if not isinstance(raw, str):
            raise ValueError("operator names must be strings")
        name = raw.strip()
        if not name:
            raise ValueError("operators list contains an empty name")
        if name not in OPERATORS:
            raise _unknown_operator(name)
        if name in seen:
            raise ValueError(f"operator {name!r} is repeated")
        seen.add(name)
        names.append(name)
    if not names:
        raise ValueError("operators list is empty")
    return tuple(names)


def _unknown_operator(name: str) -> ValueError:
    known = ", ".join(OPERATOR_NAMES)
    return ValueError(f"unknown operator {name!r}. Known operators: {known}")


def _variant_id(case_id: str, operator: str, index: int, per_case: int) -> str:
    if per_case == 1:
        return f"{case_id}__{operator}"
    return f"{case_id}__{operator}__{index}"


def _synth_description(suite: Suite) -> str:
    note = (
        "Surface variants from promptregress synth. "
        "The correct answer is assumed to be unchanged; that is not proven."
    )
    base = suite.description.strip()
    if not base:
        return note
    return f"{base} {note}"


def _assertion_to_dict(assertion: AssertionSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": assertion.kind}
    if assertion.value is not None:
        payload["value"] = assertion.value
    if assertion.weight != 1.0:
        payload["weight"] = assertion.weight
    if assertion.params:
        payload["params"] = dict(assertion.params)
    return payload


def _render_suite(suite: Suite, suffix: str) -> str:
    if suffix == ".json":
        body = json.dumps(suite_to_dict(suite), indent=2, ensure_ascii=False, allow_nan=False)
        return body + "\n"
    if suffix in {".yaml", ".yml"}:
        return _render_yaml(suite_to_dict(suite))
    if suffix == ".jsonl":
        lines = [
            json.dumps(case_to_dict(case), ensure_ascii=False, allow_nan=False)
            for case in suite.cases
        ]
        return "\n".join(lines) + "\n"
    if suffix == ".csv":
        return _render_csv(suite)
    raise ValueError(
        f"Unsupported suite format {suffix!r}. Use .json, .yaml, .yml, .jsonl, or .csv."
    )


class _QuotedDumper(yaml.SafeDumper):
    """Quote every string so YAML 1.1 does not reread ``yes`` as a bool."""


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


_QuotedDumper.add_representer(str, _represent_str)


def _render_yaml(payload: dict[str, Any]) -> str:
    dumped = yaml.dump(
        payload,
        Dumper=_QuotedDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    if not isinstance(dumped, str):
        raise ValueError("cannot serialize suite as YAML")
    if not dumped.endswith("\n"):
        return dumped + "\n"
    return dumped


def _render_csv(suite: Suite) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=["id", "input", "reference"],
        lineterminator="\n",
    )
    writer.writeheader()
    for case in suite.cases:
        reference = "" if case.reference is None else case.reference
        writer.writerow({"id": case.id, "input": case.input, "reference": reference})
    return buffer.getvalue()


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.absolute() == right.absolute()
