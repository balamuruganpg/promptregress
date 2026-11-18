"""Semantic drift between paired baseline and candidate outputs.

Drift of one case is ``1 - cosine_similarity`` of the two embedded outputs.
Cosine is clamped to ``[-1, 1]``, so drift is in ``[0, 2]``. Reports keep
that raw value. They do not rescale it into ``[0, 1]``.

Identical output strings have drift 0 and are not embedded. A pair of zero
vectors (no tokens on either side, when the embedder is actually called)
has cosine 1. A zero vector next to a non-zero vector has cosine 0, so
drift is 1. The embedder is the same family used by failure clustering:
``hash`` (default), ``tfidf``, or ``sentence-transformer``.
"""

import math
from collections.abc import Sequence
from dataclasses import replace

from promptregress.cluster import Embedder
from promptregress.models import Comparison, DriftCase, DriftSummary

DRIFT_DEFINITION = (
    "drift = 1 - cosine_similarity; cosine is clamped to [-1, 1], "
    "so drift is in [0, 2] and is reported raw"
)

_TOP = 5


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine of two vectors, clamped to ``[-1, 1]``.

    Two zero vectors return ``1`` (the same direction for drift: no
    difference). One zero vector returns ``0``. A non-finite cosine returns
    ``0``. Unequal dimensions raise ``ValueError``.
    """
    if len(left) != len(right):
        raise ValueError("embedding vectors must share a dimension")
    if not left:
        raise ValueError("embedding vectors must be non-empty")
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        dot += left_value * right_value
        left_sq += left_value * left_value
        right_sq += right_value * right_value
    left_norm = math.sqrt(left_sq)
    right_norm = math.sqrt(right_sq)
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    cosine = dot / (left_norm * right_norm)
    if not math.isfinite(cosine):
        return 0.0
    if cosine > 1.0:
        return 1.0
    if cosine < -1.0:
        return -1.0
    return cosine


def measure_drift(
    comparison: Comparison,
    embedder: Embedder,
    *,
    embedder_name: str,
) -> DriftSummary:
    """Embed each paired output and summarize drift.

    Each case is embedded as the pair ``[baseline_output, candidate_output]``
    so a corpus embedder such as TF-IDF sees both texts together. Identical
    strings skip the embedder and contribute drift ``0``. The case output is
    the text stored on the paired row, which is the first sample when the
    run used more than one sample.

    ``top`` holds at most five cases. Ties keep the lexicographically smaller
    case id first. An empty comparison has mean ``0``, max ``0``, and no rows.
    """
    if not isinstance(embedder_name, str) or embedder_name.strip() == "":
        raise ValueError("embedder_name must be a non-empty string")
    rows: list[DriftCase] = []
    for row in comparison.per_case:
        rows.append(
            DriftCase(
                case_id=row.case_id,
                drift=_pair_drift(embedder, row.baseline_output, row.candidate_output),
            )
        )
    if rows:
        mean_drift = sum(item.drift for item in rows) / len(rows)
        max_drift = max(item.drift for item in rows)
    else:
        mean_drift = 0.0
        max_drift = 0.0
    top = sorted(rows, key=lambda item: (-item.drift, item.case_id))[:_TOP]
    return DriftSummary(
        embedder=embedder_name,
        mean_drift=mean_drift,
        max_drift=max_drift,
        per_case=rows,
        top=top,
    )


def with_drift(comparison: Comparison, summary: DriftSummary) -> Comparison:
    """Return a copy of ``comparison`` carrying the drift summary."""
    return replace(comparison, drift=summary)


def _pair_drift(embedder: Embedder, baseline: str, candidate: str) -> float:
    if baseline == candidate:
        return 0.0
    vectors = embedder.embed([baseline, candidate])
    if len(vectors) != 2:
        raise ValueError("embedder returned a different number of vectors than texts")
    return 1.0 - cosine_similarity(vectors[0], vectors[1])
