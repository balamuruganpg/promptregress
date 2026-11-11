# PromptRegress user guide

Offline walkthrough. Every command was run with the mock provider or `promptregress demo`, with no API key. Scores, intervals, and the gate reproduce at the default seed (13) and 2000 resamples. Latency is wall-clock from this run. Mock token totals are character lengths, not a tokenizer.

Save the files below in an empty directory and run the commands there. This page used `/tmp/pr-hello`, which is the directory printed by `baseline list` and `doctor`.

## Contents

- [Install](#install)
- [A first suite](#a-first-suite)
- [Score a run](#score-a-run)
- [Compare and read the gate](#compare-and-read-the-gate)
- [Save a baseline](#save-a-baseline)
- [Judge with an ensemble](#judge-with-an-ensemble)
- [GitHub Action](#github-action)
- [Maintenance](#maintenance)
  - [flaky](#flaky)
  - [power](#power)
  - [calibrate](#calibrate)
  - [doctor](#doctor)

## Install

From a checkout:

```bash
python -m pip install -e ".[dev]"
promptregress --version
```

This machine already had 0.5.0, so pip replaced it, ended with `Successfully installed promptregress-0.5.0`, and `promptregress --version` printed `promptregress 0.5.0`. `python -m pip install promptregress` is the non-editable install. `promptregress init DIR` writes a two-case `suite.json`.

`run` exits 0 after scoring, including when cases fail. `compare` and `demo` exit 1 when the gate fails, and 2 when the suite or the arguments are rejected.

## A first suite

`hello.yaml`. Assertions score the output. The reference is the full answer the mock returns for the question text.

```yaml
name: hello
description: Three support questions. The brief template drops the password-reset details.
cases:
  - id: password_reset
    input: How do I reset my password?
    reference: Go to Settings, choose Security, then Reset password. Use the email link within 30 minutes.
    tags: [account]
    assertions:
      - {kind: contains, value: Settings}
      - {kind: contains, value: "30 minutes"}
  - id: refund_policy
    input: What is your refund policy?
    reference: Refunds are available within 30 days for unused items in original packaging.
    tags: [billing]
    assertions:
      - {kind: contains, value: "30 days"}
      - {kind: contains, value: unused}
  - id: shipping_time
    input: How long does standard shipping take?
    reference: Standard shipping takes 5 business days.
    tags: [shipping]
    assertions:
      - {kind: contains, value: business days}
      - {kind: numeric_tolerance, value: 5, params: {tol: 0}}
```

`rules.json`. The first matching needle wins. The brief password rule is first, because the question text is a substring of both prompts.

```json
[
  ["Answer briefly.\nHow do I reset my password?", "Use the reset link."],
  ["How long does standard shipping take?", "Standard shipping takes 5 business days."],
  ["How do I reset my password?", "Go to Settings, choose Security, then Reset password. Use the email link within 30 minutes."],
  ["What is your refund policy?", "Refunds are available within 30 days for unused items in original packaging."]
]
```

`promptregress validate hello.yaml` prints `hello.yaml: OK`. The suite name stored on a run is the `name` field, `hello`, not the file name.

## Score a run

```bash
promptregress run hello.yaml \
  --provider mock \
  --mock-rules "$(cat rules.json)" \
  --mock-default "I do not know." \
  --config-name complete \
  --template "Answer completely.
{input}" \
  --out run-complete.json \
  --no-history
```

```text
suite=hello config=complete mean_score=1.000000 pass_rate=1.000000 cases=3 mean_latency_ms=0.015236 p95_latency_ms=0.022979 total_tokens=355
```

Exit 0. The template is `Answer completely.` plus the question, so each prompt hits the full-answer rule. `--no-history` skips `.promptregress/history.jsonl`. `--out` is the same JSON `--save-baseline` stores.

## Compare and read the gate

```bash
promptregress compare hello.yaml \
  --provider mock \
  --mock-rules "$(cat rules.json)" \
  --mock-default "I do not know." \
  --baseline-template "Answer completely.
{input}" \
  --candidate-template "Answer briefly.
{input}" \
  --max-regression 0 \
  --report-json compare.json \
  --no-history
```

Exit 1. The block drops latency lines. This run used 355 baseline tokens and 274 candidate tokens. `wilcoxon_p=1.000000` and `ttest_p=0.422650`. `top_drift` also lists `refund_policy` and `shipping_time` at `0.000000`.

```text
suite=hello
baseline=baseline
candidate=candidate
baseline_mean=1.000000
candidate_mean=0.666667
baseline_pass_rate=1.000000
candidate_pass_rate=0.666667
mean_diff=-0.333333
ci_low=-1.000000
ci_high=0.000000
mcnemar_p=1.000000
mcnemar_b=1
mcnemar_c=0
cohens_dz=-0.577350
effect_size=medium
wins=0
losses=1
ties=2
mean_drift=0.161201
max_drift=0.483602
top_drift:
- password_reset 0.483602
gate_passed=false
reasons:
- FAIL regression: upper bound exceeds max_regression (mean_diff -0.333333, 95% CI [-1.000000, 0.000000], regression upper bound 1.000000, max_regression 0.000000).
- PASS pass rate: candidate meets the minimum (candidate pass rate 0.666667, minimum 0.000000).
- PASS significance: require_significant is off.
per_case:
- password_reset baseline=1.000000 candidate=0.000000 diff=-1.000000 baseline_passed=true candidate_passed=false loss
- refund_policy baseline=1.000000 candidate=1.000000 diff=0.000000 baseline_passed=true candidate_passed=true tie
- shipping_time baseline=1.000000 candidate=1.000000 diff=0.000000 baseline_passed=true candidate_passed=true tie
```

`mean_diff` is candidate minus baseline. `-ci_low` is the regression upper bound: `1.000000 > 0`, so the first reason is `FAIL` and the process exits 1. McNemar `b` is baseline-pass / candidate-fail. One discordant pair gives `mcnemar_p=1.000000`, and the gate still fails. `Use the reset link.` misses both `contains` checks. The other two prompts still match the full-answer rules, so the pass-rate check (`--min-pass-rate`, default 0) passes. Drift is `1 - cosine`, raw, in `[0, 2]`. It is printed on every compare (`drift_embedder=hash`, `ci_level=95`, `ci_method=percentile` on this run) and changes the gate only when `--max-mean-drift` is set. The `account` slice is the loss: `n=1 mean_diff=-1.000000`.

The same template on both sides exits 0: `mean_diff=0.000000`, `wins=0`, `losses=0`, `ties=3`, `gate_passed=true`. The bound prints as `-0.000000`. The check is strict greater-than, so zero passes.

## Save a baseline

Repeat the run above with `--out run-v1.json --save-baseline v1`. Names match `[A-Za-z0-9._-]+`.

```text
suite=hello config=complete mean_score=1.000000 pass_rate=1.000000 cases=3 mean_latency_ms=0.015356 p95_latency_ms=0.023466 total_tokens=355
saved baseline v1 to /tmp/pr-hello/.promptregress/baselines/v1.json
```

`--baseline-name` scores only the candidate. Omit `--baseline-template`.

```bash
promptregress compare hello.yaml \
  --provider mock \
  --mock-rules "$(cat rules.json)" \
  --mock-default "I do not know." \
  --baseline-name v1 \
  --candidate-template "Answer briefly.
{input}" \
  --candidate-name brief \
  --max-regression 0 \
  --no-history
```

Exit 1. The report starts `baseline=complete` and `candidate=brief` (the stored config name, not a new baseline call). Means, interval, and per-case rows match the live compare, including `baseline_mean_latency_ms=0.015356` from the saved run and `gate_passed=false`.

Store the brief template as `v2` (`--config-name brief --save-baseline v2`). That run prints `mean_score=0.666667` and `total_tokens=274`.

`promptregress baseline list` prints the file mtime in UTC. A later run prints a later stamp. These rows are from this page:

```text
name  date                  suite  mean
v1    2026-11-11T20:03:07Z  hello  1.000000
v2    2026-11-11T20:03:08Z  hello  0.666667
```

`baseline delete NAME` removes one file.

## Judge with an ensemble

`params.judges` scores each judge and takes the unweighted mean. No CLI judge flag is involved. `strict` returns `1` only when the judge prompt contains the reference output, else `0`. `lenient` always returns `0.8`.

`judged.yaml`:

```yaml
name: judged
description: Two answers scored by a two-judge mock ensemble.
cases:
  - id: support_hours
    input: What are your support hours?
    reference: Support is available Monday to Friday, 09:00 to 18:00 UTC.
    tags: [support]
    assertions:
      - kind: llm_judge
        params:
          rubric: Score 1 when the output matches the reference.
          judges:
            - {provider: mock, model: strict, rubric: "Score 1 only when the output equals the reference.", mock_default: "0", mock_rules: [["\nOutput: Support is available Monday to Friday, 09:00 to 18:00 UTC.\n", "1"]]}
            - {provider: mock, model: lenient, rubric: "Be generous, but not automatic.", mock_default: "0.8"}
  - id: refund_policy
    input: What is your refund policy?
    reference: Refunds are available within 30 days for unused items in original packaging.
    tags: [billing]
    assertions:
      - kind: llm_judge
        params:
          rubric: Score 1 when the output matches the reference.
          judges:
            - {provider: mock, model: strict, rubric: "Score 1 only when the output equals the reference.", mock_default: "0", mock_rules: [["\nOutput: Refunds are available within 30 days for unused items in original packaging.\n", "1"]]}
            - {provider: mock, model: lenient, rubric: "Be generous, but not automatic.", mock_default: "0.8"}
```

`judge-task-rules.json` is the model under test, not the judges:

```json
[
  ["Answer briefly.\nWhat are your support hours?", "Weekdays."],
  ["What are your support hours?", "Support is available Monday to Friday, 09:00 to 18:00 UTC."],
  ["What is your refund policy?", "Refunds are available within 30 days for unused items in original packaging."]
]
```

```bash
promptregress compare judged.yaml \
  --provider mock \
  --mock-rules "$(cat judge-task-rules.json)" \
  --mock-default "I do not know." \
  --baseline-template "{input}" \
  --candidate-template "Answer briefly.
{input}" \
  --max-regression 0 \
  --no-history
```

Exit 1. A reference scores `(1 + 0.8) / 2 = 0.900000`. `Weekdays.` scores `(0 + 0.8) / 2 = 0.400000` and fails the graded cut of 0.5. Candidate mean `(0.400000 + 0.900000) / 2 = 0.650000`.

```text
baseline_mean=0.900000
candidate_mean=0.650000
mean_diff=-0.250000
ci_low=-0.500000
ci_high=0.000000
wins=0
losses=1
ties=1
judge_agreement:
- side=baseline config=baseline n_cases=2 n_judges=2 mean_pairwise_abs=0.200000 fleiss_kappa=1.000000 pass_threshold=0.500000 per_judge_mean=mock/strict:1.000000,mock/lenient:0.800000
- side=candidate config=candidate n_cases=2 n_judges=2 mean_pairwise_abs=0.500000 fleiss_kappa=-0.333333 pass_threshold=0.500000 per_judge_mean=mock/strict:0.500000,mock/lenient:0.800000
gate_passed=false
reasons:
- FAIL regression: upper bound exceeds max_regression (mean_diff -0.250000, 95% CI [-0.500000, 0.000000], regression upper bound 0.500000, max_regression 0.000000).
per_case:
- support_hours baseline=0.900000 candidate=0.400000 diff=-0.500000 baseline_passed=true candidate_passed=false loss
- refund_policy baseline=0.900000 candidate=0.900000 diff=0.000000 baseline_passed=true candidate_passed=true tie
```

Agreement is printed only for an ensemble. The command also prints a one-line definition between `judge_agreement:` and the side rows. Baseline kappa is `1.000000` because both judges pass both cases (score at least 0.5). Candidate kappa is `-0.333333` because `strict` fails `support_hours` and `lenient` does not. Agreement does not change the gate. The bound `0.500000 > 0` does.

## GitHub Action

Checkout stays in the caller job. `uses: ./` installs this repo when `action.yml` is next to `pyproject.toml`. A copied action file with no `pyproject.toml` installs `promptregress` from PyPI. The step fails when the gate fails.

`run` and `compare` accept `--junit-xml FILE`. The file has one `<testsuite>` per run (compare writes the baseline and the candidate, named with the config labels). A case that fails is a JUnit `<failure>`. A candidate case that still passes but scores lower than the baseline is a `<system-out>` note, `regressed: baseline X -> candidate Y`, not a failure. The exit code is still the gate. `promptregress audit SUITE...` lints suite files and exits 1 when any finding is an error.

`examples/ci/baseline.yaml` is the mock provider, the full template, and the rules. `examples/ci/candidate.yaml` is the brief template. The suite is `examples/support_qa.json`.

```yaml
name: promptregress
on: [pull_request]
jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./
        with:
          suite: examples/support_qa.json
          baseline-config: examples/ci/baseline.yaml
          candidate-config: examples/ci/candidate.yaml
          max-regression: "0"
          report-html: report.html
```

`promptregress demo` runs that suite. Exit 1.

`promptregress demo` prints `provider=mock`, `baseline_template='Answer completely.\n{input}'`, `candidate_template='Answer briefly.\n{input}'`, `max_regression=0`, then `baseline_mean=1.000000`, `candidate_mean=0.785714`, `mean_diff=-0.214286`, `ci_low=-0.428571`, `ci_high=0.000000`, `mcnemar_p=0.250000`, `mcnemar_b=3`, `mcnemar_c=0`, `wins=0`, `losses=3`, `ties=11`, `gate_passed=false`. The losses are `password_reset`, `refund_policy`, and `shipping_time`, each `diff=-1.000000`. The other 11 rows are ties at `0.000000`. The compare the action builds from those two YAML files prints the same score lines, plus drift: `mean_drift=0.132971`, `max_drift=0.711325`, highest on `shipping_time`. `demo` does not print a drift section.

## Maintenance

### flaky

`flaky` keeps stored baselines whose suite name matches. The argument is that name. With `v1` then `v2`, only `password_reset` flips.

```bash
promptregress flaky hello
```

```text
case            runs  flips  flip_rate  mean      sd        min       max       note
password_reset  2     1      1.000000   0.500000  0.500000  0.000000  1.000000
refund_policy   2     0      0.000000   1.000000  0.000000  1.000000  1.000000
shipping_time   2     0      0.000000   1.000000  0.000000  1.000000  1.000000
sd is the population standard deviation (divide by N). flip_rate is pass/fail flips between consecutive runs divided by (runs - 1). Runs are ordered by file mtime, then name. A case missing from any run is excluded from flip math.
```

`--fail-on-flaky 1` exits 1 and adds `flaky: password_reset flip_rate=1.000000 >= 1.000000`. Fewer than two stored runs prints why, exits 0, and does not apply the flag.

### power

```bash
promptregress power --effect 0.5
```

A normal approximation for a paired z-test, not the bootstrap gate. The command then prints the same grid as JSON.

```text
effect_dz=0.500000
alpha=0.050000

| power | n | n_continuous | analytic_power | simulated_power |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 16 | 15.365835 | 0.516005 | 0.504000 |
| 0.80 | 32 | 31.395519 | 0.807430 | 0.803000 |
| 0.95 | 52 | 51.978840 | 0.950076 | 0.943500 |
```

The printed table also has 0.55, 0.60, 0.65, 0.70, 0.75, 0.85, and 0.90. At 0.60, `n=20`. At 0.90, `n=43`.

`promptregress power --from-report compare.json` reads the hello compare. `observed_dz=-0.577350`, `achieved_power=0.170075`. The grid uses `effect_dz=0.577350` and asks for `n=24` at power 0.80 and `n=39` at power 0.95. Three cases fail the gate and are underpowered for that effect.

### calibrate

`labels.jsonl`, one object per line:

```json
{"input": "What are your support hours?", "reference": "Support is available Monday to Friday, 09:00 to 18:00 UTC.", "output": "Support is available Monday to Friday, 09:00 to 18:00 UTC.", "human_score": 1.0}
{"input": "What is your refund policy?", "reference": "Refunds are available within 30 days for unused items in original packaging.", "output": "Refunds are available within 30 days.", "human_score": 0.8}
{"input": "How long does standard shipping take?", "reference": "Standard shipping takes 5 business days.", "output": "Shipping varies.", "human_score": 0.2}
{"input": "How do I reset my password?", "reference": "Go to Settings, choose Security, then Reset password. Use the email link within 30 minutes.", "output": "Use the reset link.", "human_score": 0.0}
```

`cal-rules.json` maps `\nOutput: <that output>\n` to `"1"`, `"0.75"`, `"0.25"`, and `"0"`.

```bash
promptregress calibrate labels.jsonl \
  --rubric "Score the answer from 0 to 1 against the reference." \
  --judge-provider mock \
  --mock-rules "$(cat cal-rules.json)" \
  --mock-default "0"
```

```text
calibration
n=4
pearson=0.997054
spearman=1.000000
mae=0.025000
threshold=0.500000
kappa=1.000000
raw_agreement=1.000000
both_pass=2
judge_only=0
human_only=0
both_fail=2
```

The text report rounds. The JSON after it repeats the rows at full precision (`pearson` `0.9970544855015814`). The judge scores are `1.000000`, `0.750000`, `0.250000`, `0.000000`. Kappa uses the threshold (default 0.5): both sides pass the first two rows and fail the last two, so kappa is `1.000000` even though the scores differ. MAE is `0.025000`.

### doctor

```bash
promptregress doctor
```

Exit 0. Absent provider variables pass. The cache path is `.promptregress/cache` in the working directory.

```text
PromptRegress doctor
python: ok 3.12.3
package: ok promptregress 0.5.0
providers: ok OPENAI_API_KEY=absent XAI_API_KEY=absent ANTHROPIC_API_KEY=absent PROMPTREGRESS_API_KEY=absent PROMPTREGRESS_BASE_URL=absent PROMPTREGRESS_MODEL=absent
cache: ok /tmp/pr-hello/.promptregress/cache writable
scorer_plugins: ok none
smoke: ok mock score=1.000000 passed=true
result: pass (6 passed, 0 failed)
```

`result: fail` exits 1.
