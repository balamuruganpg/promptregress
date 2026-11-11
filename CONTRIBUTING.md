# Contributing

## Setup

```bash
python -m pip install -e ".[dev]"
```

The runtime dependency is PyYAML only. pytest, ruff, and mypy are the `dev` extra. The suite and the demo run offline: no network and no API keys.

This checkout's dev environment is `/tmp/pr-venv`. Use `/tmp/pr-venv/bin/python`, and the same prefix for `pytest`, `ruff`, and `mypy`, when that virtualenv is the one you installed into.

## Quality gate

From the repository root, all four of these must pass:

```bash
pytest
ruff check
ruff format --check
mypy src/promptregress
```

`mypy` uses the settings in `pyproject.toml` and checks `src/promptregress` only. CI (`.github/workflows/ci.yml`) runs the same commands on Python 3.10 and 3.12.

Optional: `pre-commit install` runs the hooks in `.pre-commit-config.yaml`. Local hooks call `/tmp/pr-venv/bin/ruff` (`ruff check`, `ruff format --check`) and `/tmp/pr-venv/bin/mypy` (`mypy src/promptregress`). Pinned mirror hooks for the same three checks are listed after them. CI still checks formatting with `ruff format --check`.

## Adding a scorer

A scorer is a callable `(output: str, case: Case, assertion: AssertionSpec) -> float`. Return a value in `[0.0, 1.0]`. `score_case` clamps anything outside that range. The assertion `kind` is the registered name. Read the expected target from `assertion.value`, and extra options from `assertion.params`.

Register it in-tree:

```python
from promptregress.models import AssertionSpec, Case
from promptregress.scorers import register_scorer


@register_scorer("my_kind")  # graded unless exact=True
def my_kind(output: str, case: Case, assertion: AssertionSpec) -> float:
    return 1.0 if assertion.value in output else 0.0


@register_scorer("my_exact", exact=True)
def my_exact(output: str, case: Case, assertion: AssertionSpec) -> float:
    return 1.0 if output == assertion.value else 0.0
```

`@register_scorer` with no name uses the function name.

Pass rule in `score_case`:

- Every exact-type score must be at least 0.999.
- If any graded scorer is present, the weighted mean must also be at least 0.5.
- With only exact-type scorers, the weighted-mean check is not applied.

Ship a third-party scorer with an entry point. The group is `promptregress.scorers`. The entry point name is the kind.

```toml
[project.entry-points."promptregress.scorers"]
my_kind = "my_pkg.scorers:my_kind"
```

An entry point that does not register itself is loaded as a graded scorer. It does not replace a built-in of the same name. To opt into the exact-type rule, call `@register_scorer("my_kind", exact=True)` when the module is imported. A name that is already registered, including that self-registration, is left as it is. A non-callable raises `ScoreError`.

Add tests for the pass and fail edges. Keep them deterministic and offline.

## Pull requests

- Say what changed and how you ran the gate.
- Do not add a runtime dependency. Do not weaken existing tests or `SPEC.md` to make a change pass.
- Type every public function.

## Version bump

Bump first, then run the full gate, and never hardcode the version string in tests.

1. Set the same version in `pyproject.toml` (`version`) and `src/promptregress/__init__.py` (`__version__`).
2. Move the `Unreleased` notes in `CHANGELOG.md` under that version and the date.
3. Never hardcode the version string in tests. Read `promptregress.__version__`.
4. Bump first, then run the full gate: `pytest -q`, `ruff check`, `ruff format --check`, and `mypy src/promptregress`.
5. Do not publish from the tag workflow. Publishing is manual until a token is configured. `.github/workflows/release.yml` only builds an sdist and a wheel and uploads them as a workflow artifact.
