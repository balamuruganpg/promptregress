"""Regression tests for bugs found in the independent v0.5.0 review cycle.

Covers the 0.3.0–0.5.0 surface: untrusted run-JSON restore must reject NaN
scores, duplicate case ids, and absurd / inconsistent sample counts with a
clean ValueError — never silent NaN means or fake multi-sample stability.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from promptregress.calibration import fleiss_kappa
from promptregress.irt import load_run_report, run_from_dict
from promptregress.report import run_to_dict


def test_run_json_rejects_nan_and_infinite_scores(tmp_path: Path) -> None:
    """Non-finite case scores must not load as silent NaN / Inf means.

    ``run_from_dict`` only checked ``isinstance(..., (int, float))``, so a
    stored ``NaN`` or ``Infinity`` became ``CaseResult.score`` and poisoned
    ``RunResult.mean_score``, baseline list means, and flaky score stats.
    """
    for raw in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            run_from_dict(
                {
                    "config_name": "c",
                    "suite_name": "s",
                    "results": [{"case_id": "a", "score": raw, "passed": True, "output": "x"}],
                },
                label="probe",
            )

    for name, token in (("nan", "NaN"), ("inf", "Infinity"), ("ninf", "-Infinity")):
        path = tmp_path / f"score_{name}.json"
        path.write_text(
            '{"config_name":"c","suite_name":"s","results":'
            f'[{{"case_id":"a","score":{token},"passed":true,"output":"x"}}]}}',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="finite"):
            load_run_report(path)


def test_run_json_rejects_nan_in_related_numeric_fields() -> None:
    """score_sd, sample_scores, latency, cost.usd, and judge scores are finite."""

    def case(**fields: object) -> dict[str, object]:
        row: dict[str, object] = {
            "case_id": "a",
            "score": 0.5,
            "passed": True,
            "output": "x",
            "sample_scores": [0.5, 0.5],
            "sample_outputs": ["x", "x"],
            "n_samples": 2,
        }
        row.update(fields)
        return {"config_name": "c", "suite_name": "s", "results": [row]}

    with pytest.raises(ValueError, match="score_sd"):
        run_from_dict(case(score_sd=float("nan")), label="probe")
    with pytest.raises(ValueError, match="latency_ms"):
        run_from_dict(case(latency_ms=float("nan")), label="probe")
    with pytest.raises(ValueError, match="sample_scores"):
        run_from_dict(case(sample_scores=[0.5, float("nan")]), label="probe")
    with pytest.raises(ValueError, match="scores"):
        run_from_dict(case(scores={"exact_match": float("inf")}), label="probe")
    with pytest.raises(ValueError, match="judge_scores"):
        run_from_dict(case(judge_scores=[["mock/m", float("nan")]]), label="probe")
    with pytest.raises(ValueError, match="cost.usd"):
        run_from_dict(
            {
                "config_name": "c",
                "suite_name": "s",
                "results": [],
                "cost": {"model": "m", "usd": float("nan")},
            },
            label="probe",
        )


def test_run_json_rejects_duplicate_case_ids() -> None:
    """Duplicate case ids in a stored run are an error, not last-wins averaging.

    ``mean_score`` averaged every row while ``by_id`` kept only the last, so a
    tampered baseline could list one mean and compare/flaky another.
    """
    payload: dict[str, object] = {
        "config_name": "c",
        "suite_name": "s",
        "results": [
            {"case_id": "a", "score": 1.0, "passed": True, "output": "first"},
            {"case_id": "a", "score": 0.0, "passed": False, "output": "second"},
        ],
    }
    with pytest.raises(ValueError, match="[Dd]uplicate case id"):
        run_from_dict(payload, label="probe")


def test_run_json_rejects_absurd_or_inconsistent_sample_counts() -> None:
    """n_samples must match sample lists; a huge count without samples is rejected.

    A stored ``n_samples`` of 10**12 (or any n>1 without matching
    ``sample_scores``) previously loaded and flipped on multi-sample stability
    reporting from a single fabricated ``score_sd``.
    """
    with pytest.raises(ValueError, match="n_samples"):
        run_from_dict(
            {
                "config_name": "c",
                "suite_name": "s",
                "results": [
                    {
                        "case_id": "a",
                        "score": 1.0,
                        "passed": True,
                        "output": "x",
                        "n_samples": 10**12,
                        "score_sd": 0.25,
                    }
                ],
            },
            label="probe",
        )

    with pytest.raises(ValueError, match="sample_scores"):
        run_from_dict(
            {
                "config_name": "c",
                "suite_name": "s",
                "results": [
                    {
                        "case_id": "a",
                        "score": 1.0,
                        "passed": True,
                        "output": "x",
                        "n_samples": 5,
                        "sample_scores": [1.0, 0.0],
                        "sample_outputs": ["x", "y"],
                    }
                ],
            },
            label="probe",
        )

    # Legitimate multi-sample round-trip still loads.
    ok = run_from_dict(
        {
            "config_name": "c",
            "suite_name": "s",
            "results": [
                {
                    "case_id": "a",
                    "score": 0.5,
                    "passed": True,
                    "output": "x",
                    "n_samples": 2,
                    "sample_scores": [1.0, 0.0],
                    "sample_outputs": ["x", "y"],
                    "score_sd": 0.5,
                }
            ],
        },
        label="probe",
    )
    assert ok.results[0].n_samples == 2
    assert run_to_dict(ok)["stability"]["mean_within_case_sd"] == pytest.approx(0.5)


def test_fleiss_kappa_matches_independent_hand_table() -> None:
    """Fleiss' kappa on a 4×3 table distinct from the suite's 11+3 demo.

    Subjects (raters): (1,1,1), (0,0,1), (1,0,0), (1,1,0).
    P_bar = 1/2, p_0 = 5/12, p_1 = 7/12, P_e = 37/72, kappa = -1/35.
    """
    ratings = [(1, 1, 1), (0, 0, 1), (1, 0, 0), (1, 1, 0)]
    assert fleiss_kappa(ratings) == pytest.approx(-1 / 35)
