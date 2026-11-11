"""Failure clustering: hashed and TF-IDF embeddings, k-means, reports."""

import json
import math
import sys
import types
from pathlib import Path

import pytest

from helpers import run_with
from promptregress.cli import main
from promptregress.cluster import (
    FailureCluster,
    HashEmbedder,
    SentenceTransformerEmbedder,
    TfidfEmbedder,
    cluster_failures,
    default_k,
    kmeans,
    make_embedder,
)
from promptregress.models import CaseResult, PairedRow
from promptregress.report import write_json
from promptregress.stats import compare_runs


def _loss(case_id: str, output: str, *, baseline: float = 1.0, candidate: float = 0.0) -> PairedRow:
    return PairedRow(
        case_id=case_id,
        baseline_output="ok",
        candidate_output=output,
        baseline_score=baseline,
        candidate_score=candidate,
        diff=candidate - baseline,
        baseline_passed=True,
        candidate_passed=candidate >= 0.5,
        outcome="loss",
    )


def _support(vector: list[float]) -> set[int]:
    return {index for index, value in enumerate(vector) if value != 0.0}


def test_hash_embedder_is_deterministic_and_l2_normalised() -> None:
    embedder = HashEmbedder()
    assert embedder.dim == 256
    first = embedder.embed(["alpha bravo", "alpha bravo", ""])
    second = embedder.embed(["alpha bravo"])
    assert len(first[0]) == 256
    assert first[0] == second[0]
    assert first[0] == first[1]
    norm = math.sqrt(sum(value * value for value in first[0]))
    assert norm == pytest.approx(1.0)
    assert all(value == 0.0 for value in first[2])


def test_kmeans_recovers_two_disjoint_vocabularies() -> None:
    group_a = [
        "alpha bravo charlie",
        "bravo charlie alpha",
        "alpha charlie bravo echo",
        "echo alpha bravo",
    ]
    group_b = [
        "xray yankee zulu",
        "yankee zulu xray",
        "xray zulu yankee hotel",
        "hotel yankee xray",
    ]
    vectors = HashEmbedder().embed(group_a + group_b)
    support_a: set[int] = set()
    support_b: set[int] = set()
    for vector in vectors[:4]:
        support_a.update(_support(vector))
    for vector in vectors[4:]:
        support_b.update(_support(vector))
    assert support_a.isdisjoint(support_b)
    labels = kmeans(vectors, k=2, seed=13)
    assert len(set(labels[:4])) == 1
    assert len(set(labels[4:])) == 1
    assert labels[0] != labels[4]
    again = kmeans(vectors, k=2, seed=13)
    assert again == labels


def test_cluster_failures_labels_and_keeps_singletons() -> None:
    rows = [
        _loss("a1", "alpha bravo charlie", candidate=0.0),
        _loss("a2", "bravo charlie alpha", candidate=0.2),
        _loss("b1", "xray yankee zulu", candidate=0.5),
        _loss("b2", "yankee zulu xray", candidate=0.0),
        _loss("c1", "mango papaya quince", candidate=0.25),
    ]
    clusters = cluster_failures(rows, embedder=HashEmbedder(), k=3, seed=13)
    assert sum(cluster.size for cluster in clusters) == 5
    found = {tuple(cluster.case_ids) for cluster in clusters}
    assert ("a1", "a2") in found or ("a2", "a1") in found
    by_ids = {tuple(cluster.case_ids): cluster for cluster in clusters}
    alpha = next(cluster for cluster in clusters if "a1" in cluster.case_ids)
    assert set(alpha.label_terms) <= {"alpha", "bravo", "charlie"}
    assert alpha.label_terms
    assert alpha.size == len(alpha.case_ids)
    assert alpha.mean_score_drop == pytest.approx((1.0 + 0.8) / 2.0)
    singleton = next(cluster for cluster in clusters if cluster.case_ids == ["c1"])
    assert singleton.size == 1
    assert set(singleton.label_terms) <= {"mango", "papaya", "quince"}
    assert isinstance(singleton, FailureCluster)
    assert by_ids


def test_default_k_caps_at_four() -> None:
    assert default_k(0) == 0
    assert default_k(1) == 1
    assert default_k(2) == 1
    assert default_k(5) == 1
    assert default_k(6) == 2
    assert default_k(9) == 3
    assert default_k(12) == 4
    assert default_k(100) == 4


def test_tfidf_embedder_uses_the_corpus_and_separates_vocabularies() -> None:
    texts = ["alpha bravo", "bravo alpha", "xray yankee", "yankee xray"]
    vectors = TfidfEmbedder().embed(texts)
    assert len(vectors) == 4
    assert len(vectors[0]) == len(vectors[1])
    norm = math.sqrt(sum(value * value for value in vectors[0]))
    assert norm == pytest.approx(1.0)
    labels = kmeans(vectors, k=2, seed=13)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_case_result_drop_is_one_minus_score() -> None:
    item = CaseResult(
        case_id="only",
        output="alpha bravo",
        scores={"exact_match": 0.25},
        score=0.25,
        passed=False,
        latency_ms=0.0,
    )
    clusters = cluster_failures([item], k=1, seed=1)
    assert clusters[0].case_ids == ["only"]
    assert clusters[0].mean_score_drop == pytest.approx(0.75)
    assert cluster_failures([], k=2) == []


def test_sentence_transformer_is_a_lazy_optional_import() -> None:
    assert "sentence_transformers" not in sys.modules
    with pytest.raises(ImportError, match="sentence-transformers"):
        SentenceTransformerEmbedder()
    assert "sentence_transformers" not in sys.modules
    with pytest.raises(ValueError, match="sentence-transformer"):
        make_embedder("nope")
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert "sentence-transformers" in text


def test_tfidf_embedder_uses_a_unit_vector_when_nothing_tokenizes() -> None:
    assert TfidfEmbedder().embed([]) == []
    assert TfidfEmbedder().embed(["...", "??"]) == [[0.0], [0.0]]


def test_make_embedder_names_and_a_stubbed_sentence_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(make_embedder("tfidf"), TfidfEmbedder)
    with pytest.raises(ValueError, match="model_name"):
        SentenceTransformerEmbedder("  ")

    class _Model:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def encode(self, texts: list[str], normalize_embeddings: bool = True) -> list[list[float]]:
            assert normalize_embeddings is True
            return [[1.0, 0.0] for _ in texts]

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _Model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    embedder = make_embedder("sentence-transformer", model_name="tiny-stub")
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert embedder.model_name == "tiny-stub"
    assert embedder.embed([]) == []
    assert embedder.embed(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]

    broken = types.ModuleType("sentence_transformers")
    broken.SentenceTransformer = "not-callable"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", broken)
    with pytest.raises(ImportError, match="sentence-transformers"):
        SentenceTransformerEmbedder("tiny-stub")


def test_default_k_rejects_a_non_integer() -> None:
    with pytest.raises(ValueError, match="integer"):
        default_k(True)  # type: ignore[arg-type]


def test_kmeans_rejects_bad_arguments_and_empty_input() -> None:
    assert kmeans([], k=1) == []
    point = [[1.0, 0.0]]
    with pytest.raises(ValueError, match="k must"):
        kmeans(point, k=0)
    with pytest.raises(ValueError, match="k must"):
        kmeans(point, k=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_iter"):
        kmeans(point, k=1, max_iter=0)
    with pytest.raises(ValueError, match="max_iter"):
        kmeans(point, k=1, max_iter=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dimension"):
        kmeans([[], []], k=1)
    with pytest.raises(ValueError, match="dimension"):
        kmeans([[1.0], [1.0, 0.0]], k=1)
    with pytest.raises(ValueError, match="seed"):
        kmeans(point, k=1, seed=True)  # type: ignore[arg-type]


def test_kmeans_reseeds_empty_clusters_and_moves_a_point() -> None:
    # Identical rows collapse onto one centroid, so the empty ones are reseeded.
    identical = kmeans([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], k=3, seed=13)
    assert identical == [0, 0, 0]
    # A zero vector has cosine distance 1 from every vector, including itself.
    zeros = kmeans([[0.0, 0.0], [0.0, 0.0]], k=2, seed=1)
    assert zeros == [0, 0]
    # Seed 2 puts one point on the wrong side of the first centroids; the
    # updated means move it, which is the iteration that does not stop early.
    points = [
        [1.0, 0.0],
        [0.95, 0.2],
        [0.2, 0.95],
        [0.0, 1.0],
        [-1.0, 0.1],
        [0.5, 0.5],
    ]
    labels = kmeans(points, k=2, seed=2)
    assert labels == kmeans(points, k=2, seed=2)
    assert len(labels) == 6
    assert set(labels) == {0, 1}


def test_cluster_failures_rejects_bad_inputs_and_zero_terms() -> None:
    row = _loss("only", "alpha bravo")
    with pytest.raises(ValueError, match="n_terms"):
        cluster_failures([row], n_terms=-1)
    with pytest.raises(TypeError, match="PairedRow or CaseResult"):
        cluster_failures(["nope"])  # type: ignore[list-item]

    class _Short:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return []

    with pytest.raises(ValueError, match="different number of vectors"):
        cluster_failures([row], embedder=_Short())
    clusters = cluster_failures([row], n_terms=0, k=1, seed=1)
    assert clusters[0].label_terms == []
    assert clusters[0].case_ids == ["only"]


def test_compare_cluster_failures_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [
                    {
                        "id": "alpha_case",
                        "input": "alpha bravo",
                        "reference": "alpha bravo",
                        "assertions": [{"kind": "exact_match"}],
                    },
                    {
                        "id": "xray_case",
                        "input": "xray yankee",
                        "reference": "xray yankee",
                        "assertions": [{"kind": "exact_match"}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    report_json = tmp_path / "cmp.json"
    report_md = tmp_path / "cmp.md"
    report_html = tmp_path / "cmp.html"
    code = main(
        [
            "compare",
            str(suite),
            "--provider",
            "echo",
            "--baseline-template",
            "{input}",
            "--candidate-template",
            "nope",
            "--cluster-failures",
            "--embedder",
            "hash",
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
            "--report-html",
            str(report_html),
        ]
    )
    assert code == 1
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["failure_clustering"]["embedder"] == "hash"
    clustered = {
        case_id
        for cluster in payload["failure_clustering"]["clusters"]
        for case_id in cluster["case_ids"]
    }
    assert clustered == {"alpha_case", "xray_case"}
    markdown = report_md.read_text(encoding="utf-8")
    html = report_html.read_text(encoding="utf-8")
    assert "## Failure modes" in markdown
    assert "Failure modes" in html
    assert html.lower().count("<script") == 1
    assert "<script src" not in html.lower()
    assert "http://" not in html
    assert "failure_modes:" in capsys.readouterr().out

    quiet = tmp_path / "quiet.json"
    comparison = compare_runs(
        run_with([("alpha_case", 1.0, True)], "baseline"),
        run_with([("alpha_case", 1.0, True)], "candidate"),
        n_boot=20,
    )
    write_json(comparison, quiet)
    assert "failure_clustering" not in json.loads(quiet.read_text(encoding="utf-8"))
