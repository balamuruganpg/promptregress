"""Stored baselines: save, compare without the baseline provider, list, delete."""

import json
import re
from pathlib import Path

import pytest

from promptregress.cli import (
    DEMO_BASELINE_TEMPLATE,
    DEMO_CANDIDATE_TEMPLATE,
    demo_rules,
    main,
    run_demo,
)
from promptregress.cluster import make_embedder
from promptregress.drift import measure_drift, with_drift
from promptregress.irt import load_run_report
from promptregress.loading import load_suite
from promptregress.models import CaseResult, CostEstimate, RunResult
from promptregress.report import write_run_json

ROOT = Path(__file__).resolve().parents[1]


def _tiny(path: Path, name: str = "tiny", case_ids: tuple[str, ...] = ("a", "b")) -> Path:
    path.write_text(
        json.dumps(
            {
                "name": name,
                "cases": [
                    {
                        "id": case_id,
                        "input": case_id,
                        "reference": case_id,
                        "assertions": [{"kind": "exact_match"}],
                    }
                    for case_id in case_ids
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_run_report_round_trip_keeps_output_usage_and_cost(tmp_path: Path) -> None:
    result = RunResult(
        suite_name="s",
        config_name="c",
        cost=CostEstimate(model="mock", usd=0.5, note=None),
        results=[
            CaseResult(
                case_id="a",
                output="hello <x>",
                scores={"exact_match": 1.0},
                score=1.0,
                passed=True,
                latency_ms=4.0,
                prompt_tokens=3,
                completion_tokens=5,
                total_tokens=9,
                cached=True,
                sample_scores=[1.0, 0.0],
                sample_outputs=["hello <x>", "other"],
                score_sd=0.5,
                n_samples=2,
                tags=["t"],
            )
        ],
    )
    path = tmp_path / "run.json"
    write_run_json(result, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["results"][0]["output"] == "hello <x>"
    loaded = load_run_report(path)
    row = loaded.results[0]
    assert row.output == "hello <x>"
    assert row.prompt_tokens == 3
    assert row.completion_tokens == 5
    assert row.total_tokens == 9
    assert row.cached is True
    assert row.sample_scores == [1.0, 0.0]
    assert row.sample_outputs == ["hello <x>", "other"]
    assert row.score_sd == 0.5
    assert row.n_samples == 2
    assert row.tags == ["t"]
    assert loaded.cost is not None and loaded.cost.model == "mock"
    assert loaded.cost.usd == 0.5
    assert loaded.total_tokens == 9


def test_older_run_report_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "config_name": "c",
                "suite_name": "s",
                "results": [{"case_id": "a", "score": 1, "passed": True, "output": "hi"}],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_run_report(path)
    assert loaded.results[0].output == "hi"
    assert loaded.results[0].n_samples == 1
    assert loaded.results[0].prompt_tokens == 0
    assert loaded.cost is None


def test_save_baseline_compare_reproduces_the_demo_without_calling_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The baseline template would score the brief answers if it were executed.

    The stored run is the full-answer demo baseline. Compare still reports
    that baseline mean, so the brief template was not sent to the provider.
    """
    monkeypatch.chdir(tmp_path)
    suite_path = ROOT / "examples" / "support_qa.json"
    suite = load_suite(suite_path)
    rules = json.dumps(demo_rules(suite))
    out = tmp_path / "run.json"
    assert (
        main(
            [
                "run",
                str(suite_path),
                "--provider",
                "mock",
                "--mock-rules",
                rules,
                "--mock-default",
                "I do not know.",
                "--config-name",
                "full",
                "--template",
                DEMO_BASELINE_TEMPLATE,
                "--out",
                str(out),
                "--save-baseline",
                "support",
                "--no-history",
            ]
        )
        == 0
    )
    saved = json.loads(
        (tmp_path / ".promptregress" / "baselines" / "support.json").read_text(encoding="utf-8")
    )
    password = next(row for row in saved["results"] if row["case_id"] == "password_reset")
    assert "30 minutes" in password["output"]
    assert saved["total_tokens"] > 0

    report = tmp_path / "compare.json"
    code = main(
        [
            "compare",
            str(suite_path),
            "--provider",
            "mock",
            "--mock-rules",
            rules,
            "--mock-default",
            "I do not know.",
            "--baseline-template",
            DEMO_CANDIDATE_TEMPLATE,
            "--candidate-template",
            DEMO_CANDIDATE_TEMPLATE,
            "--baseline-name",
            "support",
            "--report-json",
            str(report),
            "--no-history",
        ]
    )
    assert code == 1
    text = capsys.readouterr().out
    assert "baseline_mean=1.000000" in text
    assert "candidate_mean=0.785714" in text
    assert "mean_diff=-0.214286" in text
    assert "ci_low=-0.428571" in text
    assert "ci_high=0.000000" in text
    assert "mcnemar_p=0.250000" in text
    assert "wins=0" in text
    assert "losses=3" in text
    assert "ties=11" in text
    assert f"baseline_total_tokens={saved['total_tokens']}" in text
    payload = json.loads(report.read_text(encoding="utf-8"))
    live, _verdict = run_demo()
    drifted = with_drift(
        live,
        measure_drift(live, make_embedder("hash"), embedder_name="hash"),
    )
    assert payload["mean_diff"] == live.mean_diff
    assert payload["ci_low"] == live.ci_low
    assert payload["ci_high"] == live.ci_high
    assert payload["mcnemar_p"] == live.mcnemar_p
    assert payload["losses"] == 3
    assert drifted.drift is not None
    assert payload["drift"]["mean_drift"] == drifted.drift.mean_drift
    lost = next(row for row in payload["per_case"] if row["case_id"] == "password_reset")
    live_lost = next(row for row in drifted.per_case if row.case_id == "password_reset")
    assert lost["baseline_output"] == live_lost.baseline_output
    assert "30 minutes" in lost["baseline_output"]
    assert lost["candidate_output"] == "Use the reset link."

    listed = main(["baseline", "list"])
    assert listed == 0
    table = capsys.readouterr().out
    assert table.startswith("name  date  suite  mean\n") or "name" in table.splitlines()[0]
    assert "support" in table
    assert "support_qa" in table
    assert "1.000000" in table
    assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", table)
    assert main(["baseline", "delete", "support"]) == 0
    assert "deleted baseline support" in capsys.readouterr().out
    assert main(["baseline", "list"]) == 0
    assert "(no baselines)" in capsys.readouterr().out
    missing = main(
        [
            "compare",
            str(suite_path),
            "--provider",
            "echo",
            "--candidate-template",
            "{input}",
            "--baseline-name",
            "support",
            "--no-history",
        ]
    )
    assert missing == 2
    assert "not found" in capsys.readouterr().err


def test_baseline_name_must_be_a_single_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("../outside", "nested/name", "has space", "white\\name"):
        code = main(
            [
                "run",
                str(tmp_path / "missing.json"),
                "--provider",
                "echo",
                "--config-name",
                "x",
                "--out",
                str(tmp_path / "out.json"),
                "--save-baseline",
                name,
            ]
        )
        assert code == 2
        assert "baseline name" in capsys.readouterr().err
    assert not (tmp_path / ".promptregress").exists()


def test_overwrite_warns_and_replaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = _tiny(tmp_path / "suite.json")
    first = main(
        [
            "run",
            str(suite),
            "--provider",
            "echo",
            "--config-name",
            "echo",
            "--template",
            "{input}",
            "--out",
            str(tmp_path / "a.json"),
            "--save-baseline",
            "v1.0_ok-a",
            "--no-history",
        ]
    )
    assert first == 0
    saved = capsys.readouterr()
    assert "warning: overwriting" not in saved.err
    assert "saved baseline v1.0_ok-a" in saved.out
    second = main(
        [
            "run",
            str(suite),
            "--provider",
            "echo",
            "--config-name",
            "echo",
            "--template",
            "nope {input}",
            "--out",
            str(tmp_path / "b.json"),
            "--save-baseline",
            "v1.0_ok-a",
            "--no-history",
        ]
    )
    assert second == 0
    again = capsys.readouterr()
    assert "warning: overwriting baseline v1.0_ok-a" in again.err
    stored = load_run_report(tmp_path / ".promptregress" / "baselines" / "v1.0_ok-a.json")
    assert stored.mean_score == 0.0


def test_suite_name_mismatch_names_both_suites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    saved = _tiny(tmp_path / "alpha.json", name="alpha")
    assert (
        main(
            [
                "run",
                str(saved),
                "--provider",
                "echo",
                "--config-name",
                "echo",
                "--out",
                str(tmp_path / "run.json"),
                "--save-baseline",
                "alpha",
                "--no-history",
            ]
        )
        == 0
    )
    current = _tiny(tmp_path / "beta.json", name="beta")
    code = main(
        [
            "compare",
            str(current),
            "--provider",
            "echo",
            "--candidate-template",
            "{input}",
            "--baseline-name",
            "alpha",
            "--no-history",
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "alpha" in err and "beta" in err
    assert "first differing case id" not in err


def test_case_id_mismatch_lists_the_first_current_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    saved = _tiny(tmp_path / "old.json", name="gamma", case_ids=("aaa_shared", "mmm_only_stored"))
    assert (
        main(
            [
                "run",
                str(saved),
                "--provider",
                "echo",
                "--config-name",
                "echo",
                "--out",
                str(tmp_path / "run.json"),
                "--save-baseline",
                "gamma",
                "--no-history",
            ]
        )
        == 0
    )
    current = _tiny(
        tmp_path / "new.json",
        name="gamma",
        case_ids=("zzz_only_current", "aaa_shared"),
    )
    code = main(
        [
            "compare",
            str(current),
            "--provider",
            "echo",
            "--candidate-template",
            "{input}",
            "--baseline-name",
            "gamma",
            "--no-history",
        ]
    )
    assert code == 2
    assert "first differing case id: zzz_only_current" in capsys.readouterr().err


def test_reordered_case_ids_still_compare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = _tiny(tmp_path / "suite.json")
    assert (
        main(
            [
                "run",
                str(suite),
                "--provider",
                "echo",
                "--config-name",
                "echo",
                "--template",
                "{input}",
                "--out",
                str(tmp_path / "run.json"),
                "--save-baseline",
                "ordered",
                "--no-history",
            ]
        )
        == 0
    )
    path = tmp_path / ".promptregress" / "baselines" / "ordered.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"].reverse()
    path.write_text(json.dumps(payload), encoding="utf-8")
    code = main(
        [
            "compare",
            str(suite),
            "--provider",
            "echo",
            "--candidate-template",
            "{input}",
            "--baseline-name",
            "ordered",
            "--no-history",
        ]
    )
    assert code == 0
    assert "mean_diff=0.000000" in capsys.readouterr().out


def test_delete_missing_baseline_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["baseline", "list"]) == 0
    assert "(no baselines)" in capsys.readouterr().out
    assert main(["baseline", "delete", "missing"]) == 2
    assert "not found" in capsys.readouterr().err
    assert main(["baseline", "delete", "a/b"]) == 2
    assert "baseline name" in capsys.readouterr().err
