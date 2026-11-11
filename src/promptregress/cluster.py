"""Cluster score regressions and label each group with TF-IDF terms.

``HashEmbedder`` and ``TfidfEmbedder`` are stdlib. ``SentenceTransformerEmbedder``
imports ``sentence_transformers`` inside its constructor only, and only when
the caller asks for that embedder. The import is not performed when this
module loads. Install it with the optional ``ml`` extra.

k-means uses cosine distance, a seeded ``random.Random``, and spherical
updates (the mean vector, then L2-normalised). Empty clusters are not
reported. Clusters of size 1 are reported.
"""

import hashlib
import importlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from promptregress.models import CaseResult, PairedRow
from promptregress.scorers import _tfidf_vectors, _tokens

_HASH_DIM = 256
_DEFAULT_TERMS = 5
_DEFAULT_MAX_ITER = 50
_ML_EXTRA = (
    "SentenceTransformerEmbedder requires the optional dependency "
    "sentence-transformers. Install it with: pip install 'promptregress[ml]'"
)


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to vectors. Non-zero vectors are L2-normalised."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text, in the same order."""


@dataclass(frozen=True)
class FailureCluster:
    """One group of failing cases.

    ``label_terms`` are the top TF-IDF terms in the cluster, highest weight
    first. ``case_ids`` keeps the order of the input. ``size`` is
    ``len(case_ids)``. ``mean_score_drop`` is the mean of the per-case drops
    (baseline minus candidate for a paired row, or ``1 - score`` for a
    single :class:`CaseResult`).
    """

    label_terms: list[str]
    case_ids: list[str]
    size: int
    mean_score_drop: float


class HashEmbedder:
    """Deterministic hashed bag-of-words embedder.

    Tokens are the scorer's casefolded words. Each token increments one of
    ``dim`` buckets (default 256) chosen by SHA-256, which does not depend
    on ``PYTHONHASHSEED``. The vector is L2-normalised. A text with no
    tokens is the zero vector.
    """

    def __init__(self) -> None:
        self.dim = _HASH_DIM

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hash_vector(text, self.dim) for text in texts]


class TfidfEmbedder:
    """Smoothed TF-IDF over the corpus passed to :meth:`embed`.

    Uses the same word tokenizer and ``log((1 + N) / (1 + df)) + 1`` IDF as
    ``tfidf_cosine``. The vocabulary is sorted so the dense layout is
    deterministic. Vectors are L2-normalised. A text with no tokens, or a
    corpus with no tokens at all, yields a zero vector of length 1 so
    downstream k-means still has a dimension.
    """

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        docs = [_tokens(text, "word", 1) for text in texts]
        sparse = _tfidf_vectors(docs)
        vocabulary = sorted({token for vector in sparse for token in vector})
        if not vocabulary:
            return [[0.0] for _ in texts]
        rows: list[list[float]] = []
        for vector in sparse:
            dense = [float(vector.get(token, 0.0)) for token in vocabulary]
            rows.append(_l2_normalize(dense))
        return rows


class SentenceTransformerEmbedder:
    """Optional sentence-transformers embedder.

    The package is imported in this constructor, not at module import.
    ``model_name`` is passed to ``SentenceTransformer``. A missing install
    raises :class:`ImportError` naming the ``ml`` extra.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if not isinstance(model_name, str) or model_name.strip() == "":
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name
        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError as exc:
            raise ImportError(_ML_EXTRA) from exc
        factory = getattr(module, "SentenceTransformer", None)
        if not callable(factory):
            raise ImportError(_ML_EXTRA)
        self._model: Any = factory(model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        raw = self._model.encode(list(texts), normalize_embeddings=True)
        return _float_matrix(raw)


def make_embedder(name: str, *, model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Build ``hash``, ``tfidf``, or ``sentence-transformer``."""
    if name == "hash":
        return HashEmbedder()
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "sentence-transformer":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown embedder {name!r}. Choose hash, tfidf, or sentence-transformer.")


def default_k(n: int) -> int:
    """Elbow-free cluster count: ``max(1, min(4, n // 3))`` for ``n >= 1``.

    ``n <= 0`` returns 0. The cap of 4 keeps a small failure set from
    splitting into a cluster per case unless the caller passes ``k``.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("n must be an integer")
    if n <= 0:
        return 0
    return max(1, min(4, n // 3))


def kmeans(
    vectors: Sequence[Sequence[float]],
    k: int,
    *,
    seed: int = 13,
    max_iter: int = _DEFAULT_MAX_ITER,
) -> list[int]:
    """Spherical k-means. Returns a cluster index per row.

    Distance is cosine distance ``1 - cos``. Centroids are the L2-normalised
    mean of their members. Initial centroids are k-means++ with
    ``random.Random(seed)``. Ties in distance keep the lower index. ``k`` is
    clamped to ``[1, n]``. An empty vector list returns ``[]``.
    """
    rows = [list(vector) for vector in vectors]
    n = len(rows)
    if n == 0:
        return []
    _check_seed(seed)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be an integer >= 1")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be an integer >= 1")
    dim = len(rows[0])
    if dim < 1 or any(len(vector) != dim for vector in rows):
        raise ValueError("vectors must be non-empty and share a dimension")
    k_used = min(k, n)
    points = [_l2_normalize(vector) for vector in rows]
    rng = random.Random(seed)
    centroids = _init_centroids(points, k_used, rng)
    labels = [0] * n
    for _ in range(max_iter):
        labels = _assign(points, centroids)
        centroids, counts = _update_centroids(points, labels, k_used, dim)
        if any(count == 0 for count in counts):
            centroids = _reseed_empty(points, labels, centroids, counts)
            labels = _assign(points, centroids)
            centroids, _counts = _update_centroids(points, labels, k_used, dim)
        refreshed = _assign(points, centroids)
        if refreshed == labels:
            labels = refreshed
            break
        labels = refreshed
    return labels


def cluster_failures(
    case_results: Sequence[PairedRow | CaseResult],
    *,
    embedder: Embedder | None = None,
    k: int | None = None,
    seed: int = 13,
    n_terms: int = _DEFAULT_TERMS,
    max_iter: int = _DEFAULT_MAX_ITER,
) -> list[FailureCluster]:
    """Cluster failure rows and label each cluster with top TF-IDF terms.

    Pass the rows to cluster. A :class:`PairedRow` contributes its candidate
    output and a score drop of ``baseline_score - candidate_score``. A
    :class:`CaseResult` contributes its output and a drop of ``1 - score``.
    This function does not drop rows: the caller decides which failures to
    include. ``compare --cluster-failures`` passes score losses only.

    ``k`` defaults to :func:`default_k`. Clusters are ordered by size
    (largest first), then by the first case id. Size-1 clusters are kept.
    A cluster that k-means leaves empty is omitted because it has no cases.
    """
    if isinstance(n_terms, bool) or not isinstance(n_terms, int) or n_terms < 0:
        raise ValueError("n_terms must be an integer >= 0")
    items = [_failure_fields(item) for item in case_results]
    if not items:
        return []
    chosen = embedder if embedder is not None else HashEmbedder()
    texts = [text for _case_id, text, _drop in items]
    vectors = chosen.embed(texts)
    if len(vectors) != len(texts):
        raise ValueError("embedder returned a different number of vectors than texts")
    k_used = default_k(len(items)) if k is None else k
    labels = kmeans(vectors, k_used, seed=seed, max_iter=max_iter)
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(label, []).append(index)
    clusters: list[FailureCluster] = []
    for indices in grouped.values():
        case_ids = [items[index][0] for index in indices]
        drops = [items[index][2] for index in indices]
        clusters.append(
            FailureCluster(
                label_terms=_top_terms(texts, indices, n_terms),
                case_ids=case_ids,
                size=len(case_ids),
                mean_score_drop=sum(drops) / len(drops),
            )
        )
    clusters.sort(
        key=lambda cluster: (-cluster.size, cluster.case_ids[0] if cluster.case_ids else "")
    )
    return clusters


def _failure_fields(item: PairedRow | CaseResult) -> tuple[str, str, float]:
    if isinstance(item, PairedRow):
        return (item.case_id, item.candidate_output, item.baseline_score - item.candidate_score)
    if isinstance(item, CaseResult):
        return (item.case_id, item.output, 1.0 - item.score)
    raise TypeError("case_results must contain PairedRow or CaseResult items")


def _top_terms(texts: Sequence[str], indices: Sequence[int], n_terms: int) -> list[str]:
    if n_terms == 0:
        return []
    docs = [_tokens(text, "word", 1) for text in texts]
    vectors = _tfidf_vectors(docs)
    totals: dict[str, float] = {}
    for index in indices:
        for term, weight in vectors[index].items():
            totals[term] = totals.get(term, 0.0) + weight
    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, weight in ranked[:n_terms] if weight > 0.0]


def _hash_vector(text: str, dim: int) -> list[float]:
    counts = [0.0] * dim
    for token in _tokens(text, "word", 1):
        counts[_bucket(token, dim)] += 1.0
    return _l2_normalize(counts)


def _bucket(token: str, dim: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dim


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    cosine = _dot(left, right) / (left_norm * right_norm)
    cosine = min(1.0, max(-1.0, cosine))
    return 1.0 - cosine


def _check_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")


def _init_centroids(
    points: Sequence[Sequence[float]],
    k: int,
    rng: random.Random,
) -> list[list[float]]:
    n = len(points)
    first = rng.randrange(n)
    centroids = [list(points[first])]
    chosen = {first}
    while len(centroids) < k:
        distances = [
            min(_cosine_distance(point, centroid) for centroid in centroids) for point in points
        ]
        weights = [distance * distance for distance in distances]
        total = sum(weights)
        if total <= 0.0:
            index = next(i for i in range(n) if i not in chosen)
        else:
            pick = rng.random() * total
            running = 0.0
            index = n - 1
            for candidate, weight in enumerate(weights):
                running += weight
                if running >= pick:
                    index = candidate
                    break
        centroids.append(list(points[index]))
        chosen.add(index)
    return centroids


def _assign(points: Sequence[Sequence[float]], centroids: Sequence[Sequence[float]]) -> list[int]:
    labels: list[int] = []
    for point in points:
        best_index = 0
        best_distance = _cosine_distance(point, centroids[0])
        for index in range(1, len(centroids)):
            distance = _cosine_distance(point, centroids[index])
            if distance < best_distance:
                best_distance = distance
                best_index = index
        labels.append(best_index)
    return labels


def _update_centroids(
    points: Sequence[Sequence[float]],
    labels: Sequence[int],
    k: int,
    dim: int,
) -> tuple[list[list[float]], list[int]]:
    sums = [[0.0] * dim for _ in range(k)]
    counts = [0] * k
    for point, label in zip(points, labels, strict=True):
        counts[label] += 1
        bucket = sums[label]
        for index, value in enumerate(point):
            bucket[index] += value
    centroids: list[list[float]] = []
    for index in range(k):
        if counts[index] == 0:
            centroids.append([0.0] * dim)
            continue
        mean = [value / counts[index] for value in sums[index]]
        centroids.append(_l2_normalize(mean))
    return centroids, counts


def _reseed_empty(
    points: Sequence[Sequence[float]],
    labels: Sequence[int],
    centroids: list[list[float]],
    counts: Sequence[int],
) -> list[list[float]]:
    """Point the empty centroids at the rows farthest from their own centroid."""
    distances = [
        _cosine_distance(point, centroids[label])
        for point, label in zip(points, labels, strict=True)
    ]
    order = sorted(range(len(points)), key=lambda index: (-distances[index], index))
    used: set[int] = set()
    updated = [list(centroid) for centroid in centroids]
    for cluster_index, count in enumerate(counts):
        if count > 0:
            continue
        for point_index in order:
            if point_index in used:
                continue
            updated[cluster_index] = list(points[point_index])
            used.add(point_index)
            break
    return updated


def _float_matrix(raw: Any) -> list[list[float]]:
    vectors: list[list[float]] = []
    for row in cast(Sequence[Sequence[object]], raw):
        vectors.append([float(cast(float, value)) for value in row])
    return vectors
