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

## traces.json

Eight agent tasks. Each case expects an ordered tool trace, such as
`search_kb` then `get_order` then `issue_refund`, or `query_db` then `chart`
then `export`. `trace_match` scores the fraction of those steps found in
order. The brief mock drops steps on `refund` and `incident`, and reverses
`sales_chart`.
