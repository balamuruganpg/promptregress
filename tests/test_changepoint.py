"""CUSUM change point on stored run-history mean scores."""

import json
import math
import random
from pathlib import Path

import pytest

from promptregress.changepoint import (
    CHANGEPOINT_NOTE,
    DEFAULT_PERMUTATIONS,
    DEFAULT_SEED,
    MIN_RUNS,
    changepoint_to_dict,
    detect_changepoint,
    format_changepoint,
)
from promptregress.cli import (
    DEMO_BASELINE_TEMPLATE,
    DEMO_CANDIDATE_TEMPLATE,
    bundled_suite_path,
    demo_rules,
    main,
)
from promptregress.history import append_history, load_history, make_run_record
from promptregress.loading import load_suite
from promptregress.providers import MockProvider
from promptregress.runner import run_suite

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "examples" / "changepoint" / "history.jsonl"
OUTPUT = ROOT / "examples" / "changepoint" / "output.txt"
_LIMIT = (
    "It reports a detected shift in the recorded means. It is not proof of "
    "cause, and it does not change the gate."
)


def _rows(scores: list[float]) -> list[dict[str, object]]:
    return [{"metrics": {"mean_score": score}} for score in scores]


def _max_cusum(scores: list[float]) -> float:
    """Independent C_k maximiser. The smallest k wins a tie."""
    n = len(scores)
    best = -1.0
    for k in range(1, n):
        before = math.fsum(scores[:k]) / k
        after = math.fsum(scores[k:]) / (n - k)
        stat = math.sqrt((k * (n - k)) / n) * abs(before - after)
        if stat > best:
            best = stat
    return best


def write_demo_history(path: Path) -> None:
    """Eight full demo runs, then eight brief ones, via the history writers."""
    suite = load_suite(bundled_suite_path())
    provider = MockProvider(
        rules=demo_rules(suite),
        default="I do not know.",
        name="mock",
        model="demo",
    )
    full = run_suite(suite, provider, "full", prompt_template=DEMO_BASELINE_TEMPLATE)
    brief = run_suite(suite, provider, "brief", prompt_template=DEMO_CANDIDATE_TEMPLATE)
    if path.exists():
        path.unlink()
    for index in range(8):
        append_history(
            make_run_record(
                full,
                provider="mock",
                model="demo",
                template=DEMO_BASELINE_TEMPLATE,
                timestamp=f"2026-11-12T00:00:{index:02d}Z",
            ),
            path,
        )
    for index in range(8):
        append_history(
            make_run_record(
                brief,
                provider="mock",
                model="demo",
                template=DEMO_CANDIDATE_TEMPLATE,
                timestamp=f"2026-11-12T00:00:{8 + index:02d}Z",
            ),
            path,
        )


def test_step_change_matches_the_closed_form() -> None:
    scores = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    report = detect_changepoint(_rows(scores), seed=13, permutations=25)
    assert report.n == 8
    assert report.change_index == 4
    assert report.before_n == 4
    assert report.after_n == 4
    assert report.before_mean == 0.0
    assert report.after_mean == 1.0
    assert report.mean_shift == 1.0
    assert report.statistic == math.sqrt((4 * 4) / 8)
    assert report.skipped == 0
    assert report.seed == 13
    assert report.permutations == 25
    assert report.note == CHANGEPOINT_NOTE
    assert _LIMIT in report.note


def test_drop_flips_the_shift_and_keeps_the_statistic() -> None:
    report = detect_changepoint(_rows([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]), permutations=10)
    assert report.change_index == 4
    assert report.mean_shift == -1.0
    assert report.statistic == math.sqrt(2.0)


def test_late_shift_is_the_weighted_split() -> None:
    report = detect_changepoint(_rows([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]), permutations=10)
    assert report.change_index == 6
    assert report.statistic == math.sqrt((6 * 2) / 8)


def test_tie_keeps_the_smallest_index() -> None:
    scores = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    report = detect_changepoint(_rows(scores), permutations=10)
    assert report.change_index == 1
    weight = math.sqrt((1 * 7) / 8)
    assert report.statistic == weight * abs(1.0 - (1.0 / 7))
    assert report.before_mean == 1.0
    assert report.after_mean == 1.0 / 7
    assert report.mean_shift == report.after_mean - report.before_mean


def test_p_value_is_the_permutation_count() -> None:
    values = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    permutations = 40
    seed = 13
    report = detect_changepoint(_rows(values), seed=seed, permutations=permutations)
    rng = random.Random(seed)
    pool = list(values)
    exceed = 0
    observed = _max_cusum(values)
    assert report.statistic == observed
    for _ in range(permutations):
        rng.shuffle(pool)
        if _max_cusum(pool) >= observed:
            exceed += 1
    assert report.p_value == (exceed + 1) / (permutations + 1)


def test_same_seed_repeats_and_does_not_touch_the_global_generator() -> None:
    scores = [0.0, 0.0, 0.0, 0.2, 0.2, 0.9, 0.9, 0.9]
    random.seed(12345)
    first_draw = random.random()
    random.seed(12345)
    left = detect_changepoint(_rows(scores), seed=4, permutations=30)
    right = detect_changepoint(_rows(scores), seed=4, permutations=30)
    assert left.p_value == right.p_value
    assert left.statistic == right.statistic
    assert random.random() == first_draw
    other = detect_changepoint(_rows(scores), seed=5, permutations=30)
    assert other.p_value != left.p_value


def test_defaults_and_caller_series_are_unchanged() -> None:
    records = _rows([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    snapshot = json.dumps(records)
    report = detect_changepoint(records)
    assert json.dumps(records) == snapshot
    assert report.seed == DEFAULT_SEED
    assert report.permutations == DEFAULT_PERMUTATIONS
    assert MIN_RUNS == 8
    assert 0.0 < report.p_value <= 1.0


def test_file_order_beats_timestamps() -> None:
    records = []
    for index, score in enumerate([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]):
        records.append(
            {
                "timestamp": f"2026-11-12T00:00:{7 - index:02d}Z",
                "command": "run",
                "suite": "s",
                "metrics": {"mean_score": score},
            }
        )
    report = detect_changepoint(records, permutations=5)
    assert report.change_index == 4
    assert report.before_runs[-1].timestamp == "2026-11-12T00:00:04Z"
    assert report.after_runs[0].timestamp == "2026-11-12T00:00:03Z"


def test_unscored_rows_are_skipped_and_record_indexes_remain() -> None:
    records: list[dict[str, object]] = [{"metrics": {"mean_score": 0.0}} for _ in range(4)]
    records.append({"note": "not a run"})
    records.extend({"metrics": {"mean_score": 1.0}} for _ in range(4))
    report = detect_changepoint(records, permutations=5)
    assert report.n == 8
    assert report.skipped == 1
    assert report.change_index == 4
    assert report.after_runs[0].index == 4
    assert report.after_runs[0].record_index == 5
    assert [ref.record_index for ref in report.before_runs] == [0, 1, 2, 3]


def test_compare_row_keeps_the_names_it_carries() -> None:
    def row(score: float, stamp: str) -> dict[str, object]:
        return {
            "timestamp": stamp,
            "command": "compare",
            "suite": "support_qa",
            "config": {
                "provider": "mock",
                "model": "demo",
                "candidate_name": "brief",
                "baseline_name": "full",
            },
            "metrics": {"mean_score": score},
        }

    records = [row(0.0, f"b{index}") for index in range(4)]
    records.extend(row(1.0, f"a{index}") for index in range(4))
    report = detect_changepoint(records, permutations=5)
    edge = report.after_runs[0]
    assert edge.command == "compare"
    assert edge.suite == "support_qa"
    assert edge.config_name is None
    assert edge.candidate_name == "brief"
    assert edge.baseline_name == "full"
    assert edge.provider == "mock"
    assert edge.model == "demo"
    text = format_changepoint(report)
    assert "candidate_name=brief" in text
    assert "config_name=" not in text


def test_blank_identifiers_are_omitted() -> None:
    records = []
    for index, score in enumerate([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]):
        records.append(
            {
                "timestamp": "",
                "command": "run",
                "suite": "a\nb" if index == 0 else "suite",
                "config": {"config_name": ""},
                "metrics": {"mean_score": score},
            }
        )
    report = detect_changepoint(records, permutations=5)
    assert report.before_runs[0].timestamp is None
    assert report.before_runs[0].config_name is None
    line = format_changepoint(report).splitlines()
    run_line = next(item for item in line if item.startswith("[0]"))
    assert "suite=a b" in run_line
    assert "timestamp=" not in run_line
    payload_runs = report.before_runs[0]
    assert payload_runs.suite == "a\nb"


def test_integer_scores_are_scores() -> None:
    report = detect_changepoint(_rows([0, 0, 0, 0, 1, 1, 1, 1]), permutations=5)
    assert report.change_index == 4
    assert report.mean_shift == 1.0


def test_refusals() -> None:
    with pytest.raises(ValueError, match="empty history$"):
        detect_changepoint([])
    with pytest.raises(ValueError, match="empty history: no mean_score rows"):
        detect_changepoint([{"command": "run"}, {"metrics": {}}])
    with pytest.raises(ValueError, match="at least 8 scored runs, got 7"):
        detect_changepoint(_rows([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="constant series: no variance to test"):
        detect_changepoint(_rows([0.5] * 8))
    with pytest.raises(ValueError, match="constant series"):
        detect_changepoint(_rows([0.0] * 9))
    with pytest.raises(ValueError, match="record 3 \\(t\\): mean_score is non-finite"):
        records = _rows([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        records[3] = {"timestamp": "t", "metrics": {"mean_score": float("nan")}}
        detect_changepoint(records)
    for bad in (float("inf"), float("-inf")):
        records = _rows([0.0] * 8)
        records[0] = {"metrics": {"mean_score": bad}}
        with pytest.raises(ValueError, match="non-finite"):
            detect_changepoint(records)
    for bad in (True, False, "1", None):
        records = _rows([0.0] * 8)
        records[1] = {"metrics": {"mean_score": bad}}
        with pytest.raises(ValueError, match="must be a finite number"):
            detect_changepoint(records)
    with pytest.raises(ValueError, match="non-finite"):
        detect_changepoint([{"metrics": {"mean_score": float("nan")}}])
    with pytest.raises(ValueError, match="row is not an object"):
        detect_changepoint(["nope"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="seed must be an integer"):
        detect_changepoint(_rows([0.0] * 4 + [1.0] * 4), seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="permutations must be an integer >= 1"):
        detect_changepoint(_rows([0.0] * 4 + [1.0] * 4), permutations=0)
    with pytest.raises(ValueError, match="permutations must be an integer >= 1"):
        detect_changepoint(_rows([0.0] * 4 + [1.0] * 4), permutations=True)  # type: ignore[arg-type]


def test_text_rounds_and_json_keeps_the_floats() -> None:
    report = detect_changepoint(_rows([0.0] * 4 + [1.0] * 4), permutations=8, seed=2)
    text = format_changepoint(report)
    assert text.startswith("changepoint\n")
    assert text.endswith("\n")
    assert f"statistic={report.statistic:.6f}" in text
    assert f"p_value={report.p_value:.6f}" in text
    assert "before:" in text
    assert "after:" in text
    assert _LIMIT in text
    payload = changepoint_to_dict(report)
    assert payload["statistic"] == report.statistic
    assert payload["before_mean"] == 0.0
    assert payload["after_runs"][0]["timestamp"] is None
    assert payload["after_runs"][0]["index"] == 4


def test_cli_warnings_json_and_refusals(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    for index, score in enumerate([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]):
        append_history(
            {
                "timestamp": f"2026-11-12T00:00:{index:02d}Z",
                "command": "run",
                "suite": "tiny",
                "config": {"config_name": "early" if score == 0.0 else "late"},
                "metrics": {"mean_score": score},
            },
            path,
        )
    out = tmp_path / "out" / "report.json"
    code = main(
        ["changepoint", "--file", str(path), "--json", "--out", str(out), "--permutations", "15"]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "warning:" in captured.err
    assert "invalid JSON" in captured.err
    payload = json.loads(captured.out)
    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert payload["skipped"] == 0
    assert payload["change_index"] == 4
    assert payload["before_runs"][-1]["config_name"] == "early"
    assert payload["after_runs"][0]["config_name"] == "late"

    text_out = tmp_path / "text.json"
    code = main(
        ["changepoint", "--file", str(path), "--out", str(text_out), "--permutations", "15"]
    )
    assert code == 0
    shown = capsys.readouterr().out
    assert shown.startswith("changepoint\n")
    assert "change_index=4" in shown
    written = json.loads(text_out.read_text(encoding="utf-8"))
    assert written["change_index"] == 4
    assert "change_index" not in shown.splitlines()[0]

    short = tmp_path / "short.jsonl"
    append_history({"metrics": {"mean_score": 1.0}}, short)
    assert main(["changepoint", "--file", str(short)]) == 2
    assert "at least 8" in capsys.readouterr().err

    constant = tmp_path / "flat.jsonl"
    for _ in range(8):
        append_history({"metrics": {"mean_score": 0.25}}, constant)
    assert main(["changepoint", "--file", str(constant)]) == 2
    assert "constant series" in capsys.readouterr().err

    poisoned = tmp_path / "nan.jsonl"
    for score in (0.0, 0.0, 0.0, float("nan"), 1.0, 1.0, 1.0, 1.0):
        append_history({"metrics": {"mean_score": score}}, poisoned)
    assert main(["changepoint", "--file", str(poisoned)]) == 2
    assert "non-finite" in capsys.readouterr().err

    junk = tmp_path / "junk.jsonl"
    junk.write_text("{}\nnot-json\n", encoding="utf-8")
    code = main(["changepoint", "--file", str(junk)])
    assert code == 2
    captured = capsys.readouterr()
    assert "warning:" in captured.err
    assert "empty history" in captured.err
    assert captured.err.index("warning:") < captured.err.index("empty history")


def test_shipped_history_is_the_demo_writers(tmp_path: Path) -> None:
    built = tmp_path / "history.jsonl"
    write_demo_history(built)
    assert built.read_bytes() == HISTORY.read_bytes()
    records, notes = load_history(HISTORY)
    assert notes == []
    assert len(records) == 16
    assert [row["metrics"]["mean_score"] for row in records[:8]] == [1.0] * 8
    assert [row["metrics"]["mean_score"] for row in records[8:]] == [11 / 14] * 8
    report = detect_changepoint(records, permutations=20, seed=13)
    assert report.change_index == 8
    assert report.before_mean == 1.0
    assert report.after_mean == 11 / 14
    assert report.mean_shift == (11 / 14) - 1.0
    assert report.statistic == math.sqrt((8 * 8) / 16) * abs(1.0 - (11 / 14))
    assert report.before_runs[-1].config_name == "full"
    assert report.after_runs[0].config_name == "brief"
    assert report.before_runs[0].timestamp == "2026-11-12T00:00:00Z"
    assert report.after_runs[-1].timestamp == "2026-11-12T00:00:15Z"
    assert report.before_runs[0].command == "run"
    assert report.before_runs[0].suite == "support_qa"


def test_shipped_output_is_the_command(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["changepoint", "--file", str(HISTORY)])
    assert code == 0
    assert capsys.readouterr().out == OUTPUT.read_text(encoding="utf-8")


def test_docs_state_the_limit_and_quote_the_output() -> None:
    output = OUTPUT.read_text(encoding="utf-8").strip()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert output in readme
    assert _LIMIT in readme
    assert _LIMIT in changelog
    assert "## Unreleased" in changelog
    # SPEC.md is a local working document and is not part of the pushed
    # tree, so its assertions only run where the file exists.
    spec_path = ROOT / "SPEC.md"
    if spec_path.exists():
        spec = spec_path.read_text(encoding="utf-8")
        assert _LIMIT in spec
        assert "C_k" in spec
        assert "random.Random" in spec
