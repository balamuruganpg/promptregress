"""Estimate run cost from a YAML or JSON pricing table.

The file maps a model name to ``prompt_per_1k`` and ``completion_per_1k``,
both USD per 1,000 tokens. Cost is

``prompt_tokens / 1000 * prompt_per_1k + completion_tokens / 1000 * completion_per_1k``.

Token totals are the run's summed usage fields. A model that is not in the
table and that recorded tokens estimates as ``None`` with a note. A run
with zero prompt tokens and zero completion tokens costs ``0``, including
an echo provider whose model is absent from the table. A bad file raises
:class:`PricingError` and does not guess a price.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from promptregress.loading import strip_utf8_bom_text
from promptregress.models import CostEstimate, RunResult


class PricingError(ValueError):
    """The pricing file is missing or malformed."""


@dataclass(frozen=True)
class ModelPrice:
    """USD charged per 1,000 prompt tokens and per 1,000 completion tokens."""

    prompt_per_1k: float
    completion_per_1k: float


def load_pricing(path: str | Path) -> dict[str, ModelPrice]:
    """Load a ``.json``, ``.yaml``, or ``.yml`` pricing map.

    The document must be a mapping of model name to an object with
    ``prompt_per_1k`` and ``completion_per_1k``. Prices are non-negative
    finite numbers. Zero is allowed. Booleans are rejected. Extra keys on
    a model are ignored. An empty mapping is valid: every model with
    tokens is then unknown.
    """
    pricing_path = Path(path)
    if not pricing_path.is_file():
        raise PricingError(f"Pricing file not found: {pricing_path}")
    suffix = pricing_path.suffix.lower()
    try:
        # utf-8-sig strips a leading BOM so Windows-saved JSON pricing
        # tables load; json.loads rejects a BOM the same way suite JSON does.
        # strip_utf8_bom_text covers a residual U+FEFF after a double-BOM
        # re-save and mid-file line-leading BOMs from concatenated chunks.
        text = strip_utf8_bom_text(pricing_path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise PricingError(f"Pricing file not found: {pricing_path}") from exc
    except UnicodeError as exc:
        # Decode errors are not OSError; surface them as PricingError so
        # callers that catch only that type (and the CLI's config path)
        # see a pricing failure, not a raw codec exception.
        raise PricingError(f"Pricing file {pricing_path} is not valid UTF-8: {exc}") from exc
    if suffix == ".json":
        data = _load_json(pricing_path, text)
    elif suffix in {".yaml", ".yml"}:
        data = _load_yaml(pricing_path, text)
    else:
        raise PricingError(
            f"Malformed pricing file {pricing_path}: use a .json, .yaml, or .yml file"
        )
    return _prices_from_mapping(pricing_path, data)


def estimate_cost(
    result: RunResult,
    model: str,
    prices: Mapping[str, ModelPrice],
) -> CostEstimate:
    """Price ``result`` for ``model``.

    Zero prompt and completion tokens return ``0.0`` and no note, whether
    or not ``model`` is in ``prices``. Any other missing model returns
    ``usd=None`` and ``note`` ``unknown model '<name>'``.
    """
    if not isinstance(model, str) or model == "":
        raise ValueError("model must be a non-empty string")
    prompt_tokens = result.total_prompt_tokens
    completion_tokens = result.total_completion_tokens
    if prompt_tokens == 0 and completion_tokens == 0:
        return CostEstimate(model=model, usd=0.0, note=None)
    price = prices.get(model)
    if price is None:
        return CostEstimate(model=model, usd=None, note=f"unknown model {model!r}")
    try:
        usd = (prompt_tokens / 1000.0) * price.prompt_per_1k + (
            completion_tokens / 1000.0
        ) * price.completion_per_1k
    except OverflowError:
        # Token totals past the float range (custom provider / edited report)
        # must not crash ``--pricing``; treat as unknown so a cost budget fails.
        return CostEstimate(
            model=model,
            usd=None,
            note=f"cost overflow for model {model!r}",
        )
    if not math.isfinite(usd):
        return CostEstimate(
            model=model,
            usd=None,
            note=f"cost overflow for model {model!r}",
        )
    return CostEstimate(model=model, usd=usd, note=None)


def cost_delta(baseline: CostEstimate, candidate: CostEstimate) -> tuple[float | None, str | None]:
    """Candidate minus baseline, in USD.

    ``None`` when either side has no numeric estimate. The note joins the
    notes that are set. Both numeric and note-free returns ``(delta, None)``.
    """
    notes = [item.note for item in (baseline, candidate) if item.note]
    note = "; ".join(notes) if notes else None
    if baseline.usd is None or candidate.usd is None:
        return None, note
    return candidate.usd - baseline.usd, note


def _load_json(path: Path, text: str) -> Any:
    try:
        return json.loads(text)
    except RecursionError as exc:
        raise PricingError(f"Malformed pricing file {path}: document is too deeply nested") from exc
    except json.JSONDecodeError as exc:
        raise PricingError(f"Malformed pricing file {path}: {exc}") from exc


def _load_yaml(path: Path, text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except RecursionError as exc:
        raise PricingError(f"Malformed pricing file {path}: document is too deeply nested") from exc
    except yaml.YAMLError as exc:
        raise PricingError(f"Malformed pricing file {path}: {exc}") from exc
    except ValueError as exc:
        raise PricingError(f"Malformed pricing file {path}: {exc}") from exc


def _prices_from_mapping(path: Path, data: Any) -> dict[str, ModelPrice]:
    if not isinstance(data, dict):
        raise PricingError(
            f"Malformed pricing file {path}: expected a mapping of model name to prices"
        )
    prices: dict[str, ModelPrice] = {}
    for key, value in data.items():
        if not isinstance(key, str) or key == "":
            raise PricingError(
                f"Malformed pricing file {path}: model names must be non-empty strings"
            )
        if not isinstance(value, dict):
            raise PricingError(
                f"Malformed pricing file {path}: model {key!r} must be a mapping "
                "with prompt_per_1k and completion_per_1k"
            )
        prompt = _price(path, key, value, "prompt_per_1k")
        completion = _price(path, key, value, "completion_per_1k")
        prices[key] = ModelPrice(prompt_per_1k=prompt, completion_per_1k=completion)
    return prices


def _price(path: Path, model: str, value: dict[str, Any], key: str) -> float:
    if key not in value:
        raise PricingError(f"Malformed pricing file {path}: model {model!r} is missing {key}")
    raw = value[key]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise PricingError(
            f"Malformed pricing file {path}: model {model!r} {key} "
            "must be a non-negative finite number"
        )
    number = float(raw)
    if not math.isfinite(number) or number < 0:
        raise PricingError(
            f"Malformed pricing file {path}: model {model!r} {key} "
            "must be a non-negative finite number"
        )
    return number
