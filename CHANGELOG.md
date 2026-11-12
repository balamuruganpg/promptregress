# Changelog

## Unreleased

## 0.12.1 — 2026-11-12

### Fixed

- The changepoint documentation test read `SPEC.md` unconditionally; SPEC.md is a local working document that is not part of the pushed tree, so the test failed in CI on every Python version. The SPEC assertions now run only where the file exists.

## 0.12.0 — 2026-11-12

### Added

- `promptregress changepoint` reads the run history (the same `.promptregress/history.jsonl` file as `history`, or `--file`) and tests the series of per-run `metrics.mean_score` values, in file order, for one mean shift. `compare` rows contribute the candidate mean. The change index maximises the CUSUM statistic `C_k = sqrt(k*(n-k)/n) * abs(before_mean - after_mean)` for `k` from 1 to `n-1` (each segment mean is `math.fsum` of that segment divided by its length; the smallest `k` wins a tie). A permutation test shuffles the series with `random.Random(--seed)` (default seed 13, default 2000 permutations, no global RNG) and reports `p = (1 + #{maximum C >= observed C}) / (permutations + 1)`. The report gives the change index, the run identifiers on each side, the before and after means, the mean shift (after minus before), the statistic, and the p-value. `--json` prints that report as JSON. `--out` writes the JSON. Fewer than 8 scored runs, a constant series, a non-finite `mean_score`, or an empty history raises `ValueError` (exit 2). It reports a detected shift in the recorded means. It is not proof of cause, and it does not change the gate.

## 0.11.1 — 2026-11-12

### Fixed (twenty-second independent review)

- `json_schema` `enum` used Python equality, so `true` matched `1` and `false` matched `0`, including inside nested arrays and objects. Enum now uses JSON equality (`1` still matches `1.0`).
- A JSON integer longer than the 4300-digit cap raised `ValueError` from `json_valid`, `json_schema`, and `trace_match`; those scorers now score 0. `type: number` no longer raises `OverflowError` on a big integer, and a weight that cannot convert to a finite float raises `ScoreError`.
- `llm_judge` crashed when a JSON `score` was an integer too large for `float` (now scores 0 with a note), aborted its scan on an earlier unreadable `{` object (now skipped), and read `Score: 1/5`, `Score: 1 / 2`, and `Score: 0.5%` as the leading number (a following `/` or `%` is now rejected).
- Word TF-IDF treated punctuation-only text as empty, scoring `!!!` vs `???` as 1; such texts now score 0 unless the stripped, casefolded text matches.
- The cache key joined fields with NUL, so distinct (model, prompt) pairs could collide; backslash and NUL inside a field are now escaped, and keys without those characters are unchanged.
- Provider `timeout` and `backoff_base` accepted `inf` and `nan`; construction now rejects non-finite values.

## 0.11.0 — 2026-11-12

### Added

- `promptregress pareto FILE [FILE...]` reads comparison JSON (`compare --report-json`) and run JSON (`run --out`) and prints the Pareto frontier. A comparison contributes its baseline and its candidate. Score is maximised. Estimated USD cost and mean case latency are minimised. A variant is dominated when another is at least as good on every axis in use and strictly better on one. The table gives the variant, score, cost, latency, whether it is on the frontier, and the closest dominating variant (`dominated_by`), which may itself be dominated. `--json` prints that report as JSON. `--out` writes the JSON. If every variant is missing cost, or every variant is missing latency, that axis is omitted and named. A mix of present and missing on one axis, a non-finite value, a duplicate name, or a set with neither cost nor latency raises `ValueError` (exit 2). Matrix ranking JSON is rejected: it stores a mean score and no per-config cost or latency. The frontier describes only these runs. It is not a confidence interval, and it does not change the gate.

## 0.10.0 — 2026-11-12

### Added

- `promptregress synth` writes a new suite of surface variants of each case input. Operators: `whitespace_casing`, `punctuation`, `synonym` (a fixed ten-word map), `instruction_rephrase`, `distractor`, `homoglyph`, and `number_format`. `--operators`, `--per-case` (default 1), and `--seed` (default 13) control the set. The input file is not modified. The output is a normal suite: `run` and `compare` score it with no extra flags. Each variant records `source_case_id`, `operator`, `variant_index`, and `changed` in metadata. An unknown operator, an empty suite, or an unwritable output path raises `ValueError` (CLI exit 2). The edits test surface form. They do not create new knowledge tests, and keeping the original answer is assumed, not proven.

## 0.9.0 — 2026-11-12

### Added

- `promptregress ablate` measures each `##` section of a candidate template by re-running the suite with that section removed. The report gives the ablated mean score, the mean paired delta versus the full prompt (`score(without) - score(full)`), a bootstrap confidence interval for that delta (the same `paired_bootstrap_ci` helper as `compare`; default percentile, 95%, seed 13, 2000 resamples), and the cases whose score changed. `--json` prints that report as JSON. `--out` writes it. Provider flags match `run` / `compare`, including `--cache`: the full prompt and every leave-one-out run share one response cache, so a case whose prompt text does not change is not called again. A template with no `##` sections, or with two sections of the same name, exits 2. The interval is the section's marginal contribution on this suite and provider, not a causal effect in general.

## 0.8.1 — 2026-11-12

### Fixed (nineteenth independent review)

- `strip_utf8_bom_text` now takes a `backslash_escapes` dialect flag: CSV loading passes `False`, so a quoted Windows path ending in `\` still closes at the following quote instead of keeping a mid-file chunk BOM on the next row's case id.
- YAML single-quoted scalars are now tracked (`''` escapes a quote, backslash is literal), so a line-leading `U+FEFF` in a single-quoted continuation is kept instead of being deleted.
- A plain-scalar apostrophe (`it's`) is no longer mistaken for a single-quote opener — openers are recognised only after a YAML token boundary — so a later chunk BOM still strips.

## 0.8.1 — 2026-11-12

### Fixed (nineteenth independent review)

- `strip_utf8_bom_text` quote tracking now handles quote dialects correctly: a CSV field ending in `\\` still closes at the following quote (a doubled backslash is an escaped backslash, not an escape of the quote), YAML single-quoted scalar continuations keep a legitimate line-leading `U+FEFF` (the protection double-quoted scalars got in reviews #16/#18), and an apostrophe in a plain YAML scalar (`it's`) no longer starts quote tracking that would hide a later chunk BOM.

## 0.8.0 — 2026-11-12

### Added

- `promptregress risk` trains a logistic regression on one or more comparison reports (`compare --report-json`, oldest first) and ranks cases by regression risk. Features are baseline-side only: baseline score, baseline latency, input length, baseline output length, tag count, and a 0/1 indicator for each common scorer kind. The label is 1 when the candidate score is strictly below the baseline score. The text report (or `--json` / `--out`) gives the training size, the holdout AUC, and, with `--target`, the top-N riskiest baseline cases with their largest feature contributions (coefficient times the standardized value). With two or more files the last file is the holdout and is not used to fit coefficients. Fewer than 20 labeled cases, a single class, or a one-class split raises `ValueError` instead of inventing a ranking. The score is a ranking aid, not a guarantee, and it does not change the gate.

### Fixed (eighteenth independent review)

- `strip_utf8_bom_text` quote tracking now treats `\` as escaping the next character inside double quotes, so a `\"` in model output no longer desyncs it: a later mid-file BOM is stripped instead of breaking `load_run_report` / `load_report_diffs`, and an in-scalar `U+FEFF` is no longer deleted.
- JUnit `_seconds(-0.0)` no longer emits `time="-0.000000"`: signed zero is normalised via `+ 0.0`.

## 0.7.1 — 2026-11-12

### Fixed (seventeenth independent review)

- OpenAI-compatible providers: chat-completions `message.content` may be a list of content parts — text parts are now concatenated (as Anthropic already did) instead of failing after retries.
- Provider config strings (base URL, API key, model) are stripped of leading/trailing whitespace, so a trailing newline from a `.env` line no longer corrupts URLs or auth headers.
- JUnit XML: non-finite or negative latency now exports as `time="0.000000"` (negatives previously became e.g. `time="-0.005000"`).
- JUnit XML 1.0: `_ILLEGAL_XML` now also strips DEL and C1 controls except NEL (U+0085), matching the Char production.
- JSON reports: `write_json` / `write_run_json` use `allow_nan=False`, so RFC-illegal NaN/Infinity literals are never written.

## 0.7.0 — 2026-11-12

### Added


- Split-conformal false-pass threshold for LLM-judge decisions. `promptregress calibrate` takes `--conformal-alpha` (default 0.1) and prints a `conformal` block (text and JSON): the score cut, the number of gold-fail calibration rows, the finite-sample bound `floor(alpha * (n + 1)) / (n + 1)`, and the empirical false-pass rate at that cut. Gold pass/fail uses the same `--threshold` cut as Cohen's kappa. The library lives in `src/promptregress/conformal.py` (stdlib: `calibrate_threshold`, `predict_pass`, `evaluate_threshold`). The quantile is the `ceil((n + 1) (1 - alpha))`-th order statistic of the gold-fail judge scores, or `+inf` (never pass) when that index is `n + 1`.
- The compare/run gate is unchanged. Conformal needs gold labels from the calibration JSONL, which a `Comparison` does not carry, so a gate check would have to take a second file and a different pass rule than the rest of the gate.

### Fixed (sixteenth independent review)

- `strip_utf8_bom_text` no longer deletes a legitimate `U+FEFF` at the start of a double-quoted CSV/YAML continuation line — stripping is now quote-aware.
- Run/compare report loaders (`load_run_report` / `load_report_diffs`) now strip mid-file BOMs too: a line-leading BOM on pretty-printed JSON previously failed `json.loads`.

## 0.6.15 — 2026-11-12

### Fixed (fifteenth independent review)

- CSV suites: a mid-file line-leading BOM no longer prefixes case ids.
- Matrix YAML: a mid-file BOM no longer renames later config keys.
- Pricing YAML: a mid-file BOM no longer renames later model keys, so cost lookup works.
- History JSONL: invalid UTF-8 now raises `ValueError` (… not valid UTF-8) instead of a bare `UnicodeDecodeError`.
- Shared helper `strip_utf8_bom_text` in the loader strips leading and per-line residual `U+FEFF`.

## 0.6.14 — 2026-11-12

### Fixed (fourteenth independent review)

- JSONL with a BOM mid-file now loads: concatenating Windows-saved chunks left `U+FEFF` on later lines, which broke suite and calibration JSONL and silently dropped history rows — a residual BOM is now stripped per line.
- A double BOM on a matrix YAML no longer corrupts the first config name (`\ufeffa`) or raises a cryptic YAML error: matrix files now read with `utf-8-sig` plus the residual strip, like suite/pricing YAML.
- Matrix files with invalid UTF-8 now raise the documented `MatrixError` instead of a bare `UnicodeDecodeError`.
- Run reports, compare reports, and calibration JSONL with invalid UTF-8 now raise a clear `ValueError: … not valid UTF-8` instead of a bare codec error, matching the 0.6.13 CSV/pricing behaviour.

## 0.6.13 — 2026-11-12

### Fixed (thirteenth independent review)

- Stored baselines and run reports saved with a UTF-8 BOM now load: `load_run_report` reads with `utf-8-sig`, so `compare --baseline-name` and `irt` work after a Windows re-save.
- A double BOM no longer breaks suite/pricing JSON or CSV headers: a residual `U+FEFF` is stripped after `utf-8-sig`, and CSV suites now read via `_read_text`.
- Calibration JSONL with a BOM now loads instead of failing on line 1.
- `power --from-report` now accepts BOM-prefixed compare JSON.
- History JSONL with a BOM no longer drops the first record as "invalid JSON".
- CSV suites with invalid UTF-8 now raise the documented `SuiteError` instead of a bare `UnicodeDecodeError`.
- Pricing files with invalid UTF-8 now raise `PricingError` instead of a bare `UnicodeDecodeError`.

## 0.6.12 — 2026-11-12

### Fixed (twelfth independent review)

- Suites and pricing files saved with a UTF-8 BOM (common from Windows editors) now load: JSON/JSONL suites and pricing JSON are read with `utf-8-sig`, and CSV suites strip the BOM so the first header is `id`, not `\ufeffid`. YAML suites already loaded via PyYAML.
- JSON suites with a BOM keep correct `source_line` values for `--github-annotations`.

## 0.6.11 — 2026-11-12

### Fixed (eleventh independent review)

- Non-finite `--max-regression` / `--min-candidate-pass-rate` values (e.g. `nan`) no longer silently pass: they raise `ValueError`, matching the cost, latency, and drift budget guards.
- A non-finite regression upper bound (`-ci_low` of `nan` or `+inf`) now fails the regression check instead of falsely passing any finite ceiling.

## 0.6.10 — 2026-11-12

### Fixed (tenth independent review)

- Deep JSON suites (e.g. a nested `json_schema` assertion) no longer crash `load_suite` with `RecursionError`: when `yaml.compose` fails on nesting `json.loads` still accepts, `source_line` is left unset instead of aborting the load.
- A non-finite or negative `mean_drift` now fails a set `--max-mean-drift` budget instead of falsely passing (`nan > budget` and `-1 > budget` are false in IEEE comparison), matching the cost and latency budget guards.

## 0.6.9 — 2026-11-12

### Fixed (ninth independent review)

- JSON `source_line` now uses the same compose-mark attachment as YAML: a nested `"id"` inside `metadata` or a `json_schema` no longer steals a later case's annotation line. The old regex scanner (and its `_consume_id` helper) is removed.
- YAML duplicate top-level `cases` keys now use the last block, matching `safe_load`: previously the compose marks pointed at the discarded first block's ids.
- A JSON case id containing an escaped newline now records its real line (1) instead of `None`.

## 0.6.8 — 2026-11-12

### Fixed (eighth independent review)

- A negative p95 latency now fails a set latency budget instead of falsely passing (`-1 > budget` is false in IEEE comparison).
- The runner's wall-clock latency fallback is clamped: a backwards clock step (or a non-finite delta) now records 0 ms instead of a negative duration, via a shared `_elapsed_ms` helper also used on the provider-exception path.
- YAML `source_line` for a case with duplicate `id` keys now uses the last key, matching PyYAML `safe_load`'s last-wins rule.

## 0.6.7 — 2026-11-12

### Fixed (seventh independent review)

- YAML `source_line` no longer false-matches an `id:` line inside a block-scalar case input: annotation file lines now come from the YAML compose-node marks of the real `id` fields.
- `EchoProvider` now records a fixed 1 ms latency like `MockProvider`, so offline `--max-p95-latency-ms` checks stay deterministic instead of using wall-clock latency.
- A non-finite p95 latency now fails a set latency budget instead of falsely passing (`nan > budget` is false in IEEE comparison).
- A non-finite or negative `cost.usd` now fails a set cost budget instead of falsely passing.
- `estimate_cost` no longer raises `OverflowError` on huge token totals: it returns an unknown cost so `--pricing` reports it and a cost budget fails cleanly.
- The runner no longer stores a non-finite or negative provider `latency_ms` as-is: invalid recorded latency falls back to a measured finite duration so it cannot poison p95.

## 0.6.6 — 2026-11-12

### Fixed (sixth independent review)

- GitHub Actions annotation property values now escape `,` as `%2C` (was `%3B`, which the runner decodes to `;`): a suite path such as `suites/prod,staging/qa.json` previously had its annotation rewritten to `suites/prod;staging/qa.json`, pointing at the wrong file.
- `run` no longer crashes (exit 2, `UnicodeEncodeError`) when the suite name carries an unpaired surrogate (e.g. from a `\ud800` JSON escape): the run summary now strips surrogates the same way `compare` already did.
- Gate-summary annotations now strip unpaired surrogates too, so a cost note quoting a model id from a `\ud800` JSON escape cannot crash UTF-8 stderr.

## 0.6.5 — 2026-11-12

### Added

- `run` and `compare` accept `--github-annotations`. Each failed case writes one GitHub Actions `::error` command to stderr: suite file, source line when the loader recorded one, case id, failing assertion, and score. The same commands are written when `GITHUB_ACTIONS=true`. A gate failure with no failed case (drift, or a cost or latency budget) writes one `::error::<reason>` summary. Property values escape `%`, CR, LF, `:`, and `,`. Message text escapes `%`, CR, and LF.
- `run` and `compare` accept `--max-cost-usd` and `--max-p95-latency-ms`. A candidate whose estimated cost or p95 case latency is above the budget fails the gate. The reason names the numbers (`cost $X exceeds budget $Y`, `p95 latency NNNms exceeds budget MMMms`). Cost reuses `estimate_cost`. An equal value passes. Unset, neither check runs. `run` exits 1 when a set budget fails and still exits 0 when only cases fail.
- `MockProvider` records a fixed latency of 1 ms on each completion (`latency_ms` overrides it). The runner stores that value instead of wall-clock time, so offline latency budgets are deterministic. Cache hits stay at 0.
- Package statement coverage is 96.2% (5459 of 5673 statements; pytest-cov reports 96% at its default precision).

## 0.6.4 — 2026-11-12

### Fixed (fifth independent review)

- HTML and Markdown compare reports no longer crash (`UnicodeEncodeError`
  → CLI exit 2) when model output, case ids, or suite names contain
  unpaired surrogates (e.g. from a `\ud800` JSON escape). The surrogate
  stripping the 0.6.3 JUnit fix introduced now covers the text, Markdown,
  and HTML renderers and every report file write.
- `audit` text output no longer crashes when findings quote surrogate
  case ids from suite JSON; it now exits on the findings, not on an
  encoding error.
- Trend HTML writes tolerate surrogate characters in suite names.

## 0.6.3 — 2026-11-12

### Fixed (fourth independent review)

- JUnit XML export now strips unpaired surrogates (U+D800–U+DFFF) in
  addition to C0 controls and U+FFFE/U+FFFF. Model output containing a
  lone surrogate (e.g. from a `\ud800` JSON escape) previously produced
  numeric character references that strict XML parsers reject, making
  the whole report unreadable to consumers such as Jenkins.

## 0.6.2 — 2026-11-12

### Added

- HTML comparison reports are interactive and still one file. CSS and
  JavaScript are inline, and the page does not request scripts, stylesheets,
  images, or fonts. The per-case table is rendered in full, so it stays
  readable with JavaScript off. The script only sorts and filters those rows.
- Column buttons sort by case id, baseline score, candidate score, delta,
  drift, and candidate latency. `aria-sort` records the direction. A search
  box matches the case id and the case input recorded on the run. Outcome
  (all, wins, losses, ties, failures) and tag filters combine with the
  search. A failure is a case the candidate did not pass. The tag list is
  the tags on the paired rows. The counter shows how many rows match.
- Each row expands to the side-by-side outputs. That replaces the separate
  output diff section. Drift, when the comparison recorded it, and the case
  input, when the runner captured it, are in the expansion. The summary
  includes an inline SVG of wins, losses, ties, and the two mean scores.

## 0.6.1 — 2026-11-12

### Added

- `run` and `compare` accept `--junit-xml FILE`. The file is a JUnit
  `<testsuites>` document with one `<testsuite>` per executed run.
  Compare writes the baseline suite, then the candidate suite. Each
  suite's `name` is that run's config label. `tests` and `failures`
  count cases (`failures` is cases whose own `passed` bit is false).
  `errors` is 0: a provider exception is a failed case, not a JUnit
  error. `time` is total latency in seconds (`total_latency_ms / 1000`).
  Each `<testcase>` uses the eval suite name as `classname` and the case
  id as `name`. A failed case has `<failure message="score=...; kind=...">`
  naming the case score and the first failing per-kind subscore in
  `CaseResult.scores` (exact kinds below 0.999, other kinds below 0.5).
  Its model output is `<system-out>` when the output is non-empty. A
  candidate case that still passes but whose score dropped versus the
  baseline (a paired loss, past the 1e-12 tie) gets
  `<system-out>regressed: baseline X -> candidate Y</system-out>` and is
  not a JUnit failure. The gate still sets the process exit code.
  Attribute and text values are XML-escaped. XML 1.0 forbidden control
  characters are removed before writing.
- `promptregress audit SUITE...` lints suite files through the loader.
  A file that fails to load is an error finding, not a crash. Checks:
  no assertions (error), unknown scorer kind against the live registry
  including entry-point plugins (error), duplicate case id (error; the
  loader stops at the first duplicate, and that file is not linted
  further), identical input text (warning), empty reference or value
  for scorers that need a reference (warning). The registry does not
  record which kinds need a reference. That list is hardcoded from
  `scorers.py`: `exact_match`, `contains`, `levenshtein`, and
  `tfidf_cosine` (they use assertion value, otherwise the case
  reference). `--json` prints the findings instead of the text report.
  Exit 1 when any finding is an error. Warnings alone exit 0.

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
