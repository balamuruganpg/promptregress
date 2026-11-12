# Examples

Offline suites for `promptregress`. Each one loads with `load_suite` and can be
compared with `MockProvider`: return the case reference when the case input
occurs in the prompt. A candidate template that starts with `Answer briefly.`
can register a shorter answer first so the regression is reproducible.
`run` / `compare` accept `--samples` (and `--mock-vary` when mock rules list
alternate responses) on these suites, including `support_qa` and `traces`.

## support_qa.json

Customer-support answers used by `promptregress demo` (14 cases). Assertions
mix contains, regex, numeric tolerance, and JSON checks. The demo candidate
drops a required detail on three cases and the gate fails.

## rag_qa.json

Twelve retrieval questions. The input carries a context passage, and the
reference must use a fact from that passage plus a citation marker such as
`[1]`. A brief candidate drops the citation or the grounded fact on four
cases (`refund_window`, `link_expiry`, `ship_cutoff`, `export_delay`).

## code_gen.json

Ten short code-writing tasks. Scores use contains, regex, and JSON schema
checks on the snippet (function name, control flow, or object fields). Mock
outputs are the reference snippets. Three brief stubs omit a required
construct and fail those structural checks.

## summarization.json

Ten source passages. Each case scores TF-IDF cosine against the reference
summary and contains checks for two key points. Three brief summaries keep
one key point and drop the other, so the exact check fails.

## calibrate/

`labels.jsonl` is six `{input, reference, output, human_score}` rows for
`promptregress calibrate`. `rules.json` maps each output token to a mock
judge score (0.1, 0.2, 0.3, 0.8 on the gold-fail rows; 0.9 and 1 on the
gold-pass rows). `--conformal-alpha 0.25` yields threshold 0.8 on the four
fail scores.

## risk/

`cmp_01.json` through `cmp_04.json` are comparison reports, not suites.
They do not load with `load_suite`. `promptregress risk` trains on the
first three, oldest first, and `--target cmp_04.json` ranks that report.
The `risk_*` cases drop against the baseline. The `safe_*` cases do not.

## ablate/

`suite.json` is four exact-match questions for `promptregress ablate`.
`template.txt` has two `##` sections. `Policy` contains `POLICY::{input}`,
which is what `rules.json` matches. `Tone` is a greeting the mock ignores.
The README runs this suite and pastes the report from that command.

## pareto/

`comparison.json` and `slow.json` are reports, not suites. They do not load
with `load_suite`. `comparison.json` is a `compare --report-json` document:
baseline `cheap`, candidate `quality`. `slow.json` is a `run --out` document.
`promptregress pareto` reads both. The README pastes that command's output.

## changepoint/

`history.jsonl` is a run history, not a suite. It does not load with
`load_suite`. `make_run_record` and `append_history` wrote it from the
bundled support_qa demo: eight `Answer completely.` runs (`full`, mean 1),
then eight `Answer briefly.` runs (`brief`, mean 11/14). `output.txt` is
the stdout of `promptregress changepoint --file examples/changepoint/history.jsonl`.

## traces.json

Eight agent tasks. Each case expects an ordered tool trace, such as
`search_kb` then `get_order` then `issue_refund`, or `query_db` then `chart`
then `export`. `trace_match` scores the fraction of those steps found in
order. The brief mock drops steps on `refund` and `incident`, and reverses
`sales_chart`.
