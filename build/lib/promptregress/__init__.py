"""Regression testing for LLM prompts and models."""

__version__ = "0.3.1"

from promptregress.gate import GateRule, GateVerdict
from promptregress.models import (
    AssertionSpec,
    Case,
    CaseResult,
    Comparison,
    PairedRow,
    RunResult,
    ScoringConfig,
    SliceSummary,
    Suite,
)

__all__ = [
    "__version__",
    "AssertionSpec",
    "Case",
    "CaseResult",
    "Comparison",
    "GateRule",
    "GateVerdict",
    "PairedRow",
    "RunResult",
    "ScoringConfig",
    "SliceSummary",
    "Suite",
]
