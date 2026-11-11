# Changelog

## 0.6.0 — 2026-11-12

### Added

- `AnthropicProvider` calls the Anthropic Messages API
  (`POST https://api.anthropic.com/v1/messages`) with `x-api-key`,
  `anthropic-version: 2023-06-01`, and `max_tokens` (default 1024).
  The key is the constructor argument or `ANTHROPIC_API_KEY`. A missing
  or blank key raises `ProviderError` and does not send a request.
  Text content blocks are concatenated. `usage.input_tokens` and
  `usage.output_tokens` become `prompt_tokens` and `completion_tokens`,
  and `total_tokens` is their sum. The default model is `claude-sonnet-4-6`.
- `--provider anthropic` selects that client. Retries match the
  OpenAI-compatible provider (at most 3 attempts, exponential backoff).
  A non-200 response is a `ProviderError` that includes the status.
- `doctor` reports whether `ANTHROPIC_API_KEY` is present or absent and
  does not print the value.

## 0.5.1 — 2026-11-12

### Fixed (third independent review)

- Stored run JSON is now validated on load: non-finite scores (NaN/Infinity)
  in any numeric field are rejected instead of poisoning means, flaky stats
  and baseline listings; duplicate case ids are rejected (mean and per-case
  lookups disagreed); absurd or inconsistent `n_samples` values are rejected
  instead of falsely enabling multi-sample stability reporting.

### Documentation

- CLI `--help` now states, for each subcommand, what the command reads, which defaults apply, and how it exits. Wording and formatting only. Command behavior is unchanged.
- Added `docs/GUIDE.md`, an offline walkthrough of a first suite, `run` / `compare` and the gate, stored baselines, an `llm_judge` ensemble, the GitHub Action, and `flaky`, `power`, `calibrate`, and `doctor`. The README quickstart links to it.

### Tests

- Cluster tests cover empty TF-IDF input, a stubbed sentence-transformer embedder (no model download), k-means argument checks, empty-cluster reseeding, and `cluster_failures` input checks.

## 0.5.0 — 2026-11-12

### Flakiness

- `promptregress flaky SUITE` (or `flaky --suite SUITE`) reads
  `.promptregress/baselines/` and keeps runs whose stored suite name matches.
  The argument is that name, not a file path. Runs are ordered by file
  modification time, then baseline name.
- For each case: pass/fail flip count between consecutive runs (the stored
  `passed` bit), flip rate `flips / (n_runs - 1)`, and score mean, population
  standard deviation (divide by N), min, and max. The terminal table sorts by
  flip rate descending, then standard deviation descending, then case id.
- A case missing from any run of the suite is listed with how many runs
  contain it. It is excluded from flip math. Its score statistics use the
  runs where it appears.
- Fewer than 2 stored runs prints why and exits 0. `--fail-on-flaky RATE`
  is not applied in that case. `RATE` must be finite and in `[0, 1]`. When
  at least two runs exist, the command exits 1 if any included case has
  flip rate `>= RATE`. Excluded cases do not trip the flag.
- `--out FILE` writes the per-case JSON, including runs that were not enough.

### Judges

- `llm_judge` accepts `params.judges`, a list of judge configs. Each entry
  uses the single-judge shape: `provider` (`mock`, `echo`, or `openai`),
  optional `model`, `rubric` (or the assertion `params.rubric` as a
  fallback), and for `mock` optional `mock_rules` and `mock_default`.
  `openai` also accepts `api_key`, `base_url`, and `timeout`.
- The assertion score is the unweighted mean of the judge scores. A judge
  that fails to score contributes 0 and records an error. Details include
  each judge's score. Omitting `judges` keeps the single context judge, with
  the same score and the same failure records.
- Compare reports add a judge-agreement section only when an ensemble was
  used. It reports the mean pairwise absolute score difference, Fleiss'
  kappa on pass/fail (a judge passes when its score is at least 0.5), and
  each judge's mean score. It does not change the gate.

## 0.4.0 — 2026-11-12

### Workflow

- `run --save-baseline NAME` stores the run report (the same JSON as `--out`,
  including each case `output`) at `.promptregress/baselines/NAME.json`.
  Names match `[A-Za-z0-9._-]+`. Overwriting prints a warning.
- `compare --baseline-name NAME` loads that file as the baseline and runs
  only the candidate. The baseline provider is not called. The stored suite
  name and case ids must match the current suite. A live baseline's config
  label is `--baseline-config-name` (default `baseline`).
- `baseline list` prints name, date, suite, and mean. `baseline delete NAME`
  removes one file.
- HTML compare reports add a collapsible side-by-side output diff for cases
  whose score or output changed, with the score delta and drift when drift
  was measured. Unchanged cases stay in the per-case table. Markdown quotes
  outputs for score changes only.
- `schema` prints a draft 2020-12 JSON Schema for the suite document.
  `validate FILE...` loads each suite and exits 1 when any file is invalid.

## 0.3.1 — 2026-11-12

### Documentation

- Sample HTML report regenerated with the v0.3.0 sections (semantic drift,
  stability); new `docs/assets/drift.png` chart from real demo compare
  data, embedded in the README; the deep-dive gate description now lists
  all four gate checks including the drift budget.

## 0.3.0 — 2026-11-12

### Semantic drift

- `compare` embeds each case's baseline and candidate outputs with the
  configured embedder (`--embedder hash|tfidf|sentence-transformer`, default
  `hash`, the same flag failure clustering uses). Per-case drift is
  `1 - cosine_similarity`. Cosine is clamped to `[-1, 1]`, so drift is in
  `[0, 2]` and is reported raw (not rescaled to `[0, 1]`). Identical output
  strings have drift 0. Two zero vectors have cosine 1. A zero vector next
  to a non-zero vector has cosine 0 (drift 1).
- JSON, Markdown, and HTML reports include mean drift, max drift, and the
  five highest-drift cases (id and raw drift). Ties break by case id.
- `--max-mean-drift` is optional. Unset leaves the gate unchanged. When set,
  the gate fails if mean drift exceeds it, and the drift reason is listed
  with the other checks. An equal mean passes.

### Multi-sample runs

- `run` and `compare` take `--samples N` (default 1). `N=1` keeps the
  single-call output, score, and pass bit. With `N>1`, the case score is the
  mean of the sample scores and the case passes when that mean is at least
  0.5. `score_sd` is the population standard deviation of the sample scores
  (divide by N). Compare still pairs cases on those mean scores.
- The cache key stays provider + model + prompt. Repeated samples of the
  same prompt share one entry, so later samples replay the stored output.
  The runner passes the sample index to providers that implement
  `complete_at`.
- Reports include a stability section when a run used more than one sample:
  the mean within-case sd, and the five least stable cases. Run JSON always
  carries the same aggregates.
- `MockProvider(vary=True)` (CLI `--mock-vary`) cycles a list of alternate
  outputs on the matched rule, by sample index. A string response is the
  fallback and does not change. With `vary` off, a list uses its first string.

### Cost

- `--pricing` reads a YAML or JSON map of model name to
  `{prompt_per_1k, completion_per_1k}` in USD. Cost is
  `prompt_tokens / 1000 * prompt_per_1k + completion_tokens / 1000 * completion_per_1k`.
  Compare reports include each run's cost and the candidate-minus-baseline
  delta. An unknown model with tokens is `null` / `unknown` plus a note.
  Zero prompt and completion tokens cost 0, including echo. A malformed
  pricing file is a config error.

## 0.2.4 — 2026-11-11

### Fixed (second independent review)

- Pairwise preference parsing: when both compared outputs are the same text,
  a judge reply matching that text is now a tie instead of raising
  `ValueError` (the position-bias and verbosity audits no longer crash on
  identical outputs).
- GitHub Action: the compare step now forwards `--mock-default` whenever it
  is set, not only when mock rules are also present — a mock config with a
  default response and no rules previously scored empty outputs.

### Verified (no changes needed)

- `trace_match` dynamic program matches a brute-force LCS checker on 400
  seeded random traces; history reading tolerates a corrupt trailing line
  and concurrent appends; matrix ranking is deterministic; every `doctor`
  check failing alone exits 1 with a readable report.

## 0.2.3 — 2026-11-11

### Scorers

- `trace_match` (graded) scores an agent trace. The assertion `value` is the
  expected list of `{tool, args}` steps. That is the existing structured
  `value` field (the same one `json_schema` uses); `params` is not read.
  The model output must be a JSON list of step objects. The score is the
  number of expected steps in the longest in-order matching, divided by the
  number of expected steps. A step matches when the tool names are equal and
  every expected arg value's comparison text is a case-sensitive substring of
  the actual arg value. Strings compare as themselves. Other JSON values use
  `json.dumps` (`true`, `false`, `null`). Extra actual steps are ignored and
  cannot be reused. This is a subsequence match, not a greedy left-to-right
  scan: a missing middle step still credits later steps that stay in order.
  An empty expected trace scores 1.0 and does not parse the output. Output
  that is not a JSON list scores 0.0.

### Examples

- `examples/traces.json`: 8 agent-task cases (support, data, ops, and
  others). The baseline mock returns the reference trace. The brief template
  drops steps on `refund` and `incident` and reverses `sales_chart`.

## 0.2.2 — 2026-11-11

### Release prep

- `CONTRIBUTING.md`: how to run the full gate, add a scorer, and what a PR
  is expected to include.
- `SECURITY.md`: vulnerability reporting via private GitHub advisories;
  0.2.x supported, 0.1.x not.
- `.pre-commit-config.yaml`: ruff check + format and mypy hooks, pinned.
- `.github/workflows/release.yml`: on a `v*` tag push, builds the sdist and
  wheel and uploads them as a workflow artifact. It does not publish to
  PyPI (no token).
- `tests/test_release_prep.py` covers the release workflow triggers, the
  contributor docs, the pre-commit pins, and that `dist/` is gitignored.

### Packaging

- License metadata is now the SPDX expression `license = "MIT"` with
  `license-files = ["LICENSE"]` (the old `license = {text = "MIT"}` form
  stops building in setuptools after 2027-02-18); `setuptools>=77`.
- Verified the sdist + wheel build: the wheel holds only the package
  (including `py.typed` and `data/support_qa.json`) plus dist-info.

### Docs

- README tests badge now reads 218 tests passed, matching the suite.

## 0.2.1 — 2026-11-11

### Fixed (independent review)

- `parse_preference` now matches a judge reply that is the exact text of an
  output with leading/trailing whitespace, so the position-bias and verbosity
  audits score padded outputs instead of raising.
- `json_valid` / `json_schema` scorers reject `NaN` / `Infinity` (not valid
  RFC 8259 JSON); schema `number` also requires a finite value.
- Fuzz hardening: loaders surface malformed UTF-8, parser recursion, and
  oversized integers as `SuiteError`; `numeric_tolerance` scores non-finite
  or overflowing parses as 0 instead of crashing.

### Examples

- `examples/rag_qa.json`: 12 retrieval questions. Each input includes a context
  passage, and the reference answers with a fact from that passage plus a
  citation. A brief-style mock candidate drops the citation or the grounded
  fact on four cases.
- `examples/code_gen.json`: 10 short code-writing tasks scored with contains,
  regex, and JSON structural checks. Mock outputs are the reference snippets.
- `examples/summarization.json`: 10 source passages scored with TF-IDF cosine
  against the reference summary, plus contains checks for key points.
- `examples/README.md` maps each suite to what it demonstrates.

### Quality

- Seeded fuzz tests (`tests/test_fuzz.py`) for suite loading, paired
  statistics, and the deterministic scorers. Random malformed JSON, YAML, and
  JSONL either load or raise `SuiteError`.
- `numeric_tolerance` scores 0 when the parsed number overflows or is not
  finite, instead of raising.
- Suite loading raises `SuiteError` for invalid UTF-8, documents nested past
  the parser stack, and integers JSON will not decode.
- `tests/test_perf.py` compares a generated 1000-case suite with
  `MockProvider` and the response cache inside a measured time budget.

### CLI

- `promptregress doctor` prints the Python version, the package version,
  whether OpenAI and xAI provider env vars are set (names only, never the
  values), whether the cache directory is writable, which
  `promptregress.scorers` entry points were found, and a one-case
  `MockProvider` smoke run. The process exits 0 only when every check passes.

## 0.2.0 — 2026-11-11

The gate is unchanged: it still fails from the bootstrap CI upper bound on
regression, not from Wilcoxon, the t-test, pairwise judgement, or clusters.

### Statistics

- BCa bootstrap confidence interval for the mean paired difference, selectable
  with `--ci-method {percentile,bca}`. Default remains percentile, so existing
  gates and the demo numbers are unchanged. Constant diffs, `n < 3`, and a
  degenerate jackknife or bias correction fall back to the percentile interval
  and record that in `details`.
- Wilcoxon signed-rank test (two-sided) on paired score diffs: exact rank-sum
  distribution for `n <= 20` non-zero diffs, normal approximation with
  continuity correction and tie handling otherwise.
- Paired t-test with a stdlib regularised incomplete beta (continued fraction;
  no SciPy). Zero-variance diffs are handled explicitly.
- Cohen's dz for paired diffs, with a negligible / small / medium / large
  label at 0.2 / 0.5 / 0.8.
- Per-tag slice summaries (mean diff, CI, wins / losses / ties, n) from case
  `tags`. Suites without tags load as before.
- `promptregress power` prints the normal-approximation paired sample size
  for powers 0.50–0.95, a seeded normal simulation beside the analytic power,
  and achieved power from `--from-report`.
- `wilcoxon_p` and `ttest_p` are reported next to `mcnemar_p`. They are
  evidence; they do not change gate semantics.

### ML analytics

- Failure clustering on `compare --cluster-failures`. Embedders: `hash`
  (256-d hashed bag of words), `tfidf` (the scorer's smoothed TF-IDF), and
  optional `sentence-transformer` (`ml` extra, imported only when constructed).
  Spherical k-means, top TF-IDF labels, singletons kept. The gate is unchanged.
- `promptregress irt` fits a 2PL model (difficulty, discrimination, ability)
  to two or more run reports by fixed-iteration gradient descent. Graded
  scores are soft binary labels. Discrimination is pulled toward 1.

### Judging

- `promptregress calibrate` scores a JSONL file of
  `{input, reference, output, human_score}` with `llm_judge` and reports
  Pearson, Spearman, MAE, and Cohen's kappa at a pass/fail threshold.
  `position_bias_audit` and `verbosity_bias` measure order flips and length
  preference. Offline with `MockProvider`.
- `compare --pairwise-judge` asks a judge which output is better, in both
  presentation orders. The candidate scores 1 for winning both, 0.5 for a
  split or a tie, and 0 for losing both. Reports include that mean and a
  bootstrap confidence interval. The gate does not read it.

### Operations

- Suite `scoring.weights` maps scorer kinds to non-negative weights. The case
  score is that weighted mean (each assertion weight times its kind's weight).
  Omitting the map, or setting every weight equal, keeps the previous mean.
  Unknown scorers, negative weights, and all-zero weights are config errors.
  Per-scorer subscores stay on the case result.
- Case results carry provider `usage` (`prompt_tokens`, `completion_tokens`,
  `total_tokens`) and a `cached` flag. `OpenAICompatibleProvider` copies the
  response `usage` object. `MockProvider` uses character lengths. Cache hits
  record latency 0 and the stored usage. Runs aggregate total tokens and mean
  and p95 latency. `total_latency_ms` is unchanged.
- `run` and `compare` append a JSON line to `.promptregress/history.jsonl`.
  `--no-history` skips it. A failed history write warns and does not fail the
  command. `history` prints a table. `trend` writes a self-contained HTML
  sparkline of mean score for one suite and prints an ASCII sparkline.
- `matrix` runs two or more named configs from a YAML file and writes a
  ranking table sorted by mean score. Each row has its paired difference from
  the best config and a bootstrap confidence interval.

### Reports

- JSON, Markdown, and HTML reports include the interval (percentile or BCa),
  Wilcoxon, the paired t-test, Cohen's dz, slices, latency and token usage,
  pairwise judgement when requested, and failure clusters when requested.

### CI

- Composite action `PromptRegress gate` (`action.yml`). The caller checks out
  the repo. The action sets up Python, installs promptregress from the repo
  when `uses: ./` (or from PyPI when the action directory has no project),
  runs `compare`, and fails the step when the gate fails. JSON, Markdown, and
  HTML reports are uploaded when those paths are set.
- `.github/workflows/pr-selftest.yml` runs that action on the bundled demo
  for every push. The demo gate fails on purpose. The workflow asserts exit
  code 1, so CI stays green.

## 0.1.0 — 2026-11-11

Initial release.

- Load eval suites from JSON, YAML, JSONL, and CSV.
- Deterministic scorers: exact match, contains, regex, JSON validity, a JSON
  Schema subset, numeric tolerance, normalized Levenshtein similarity, and
  TF-IDF cosine.
- Paired bootstrap confidence interval and McNemar's test on pass/fail flips.
- CI gate with a non-zero exit code when a candidate prompt regresses.
- Offline `promptregress demo` driven by `MockProvider` (no API key, no network).
- JSON, Markdown, and self-contained HTML comparison reports. `compare
  --report-html` writes the HTML file.
- `llm_judge` scorer. A judge provider scores an output against
  `params.rubric`. The score is parsed from a plain number, JSON
  `{"score": ...}`, or `Score: x`. Judge failures score 0 and record an
  error instead of raising.
- Scorer plugins from the `promptregress.scorers` entry-point group.
- A response-cache file that is not valid UTF-8 counts as a miss. It no
  longer raises out of `ResponseCache.get`.
