# RAG evaluation baseline

Run from the repository root with the existing environment and Chroma index:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -B -m debug.rag_evaluation
.venv/bin/python -B -m unittest debug.rag_evaluation_test debug.rag_integration_test -v
```

The evaluator prints each query, counts, metrics, and top five filtered names,
then aggregate results and the five worst cases. Detailed results, source IDs,
tags, expectations, executed queries, dataset/benchmark hashes, and a timestamp
are saved to `/tmp/rag-evaluation.json`. Use `--output PATH` or
`--benchmarks PATH` to choose alternatives. No Gemini calls, index rebuilding,
graph execution, or production code changes are involved.

## Benchmark contract

`rag_benchmarks.json` contains 20 software scenarios. Each has `id`, `category`,
and `query`. Optional fields:

- `expected_terms`: all phrases must appear in name, description, tags, or ingredients.
- `expected_tag_groups`: AND between groups, OR within each group; exact tag values.
- `expected_ingredient_groups`: AND between groups, OR within each group; ingredient phrases.
- `forbidden_terms`: no ingredient may contain any listed phrase.
- `max_minutes`: hard stored cooking-time ceiling; omitted means the dataset ceiling of 180.
- `minimum_valid_candidates`: defaults to seven, matching the demo meal count.
- `planning_constraints`: exercises the production query builder. Otherwise `query`
  is sent literally to retrieval. Both the scenario and actual query are recorded.

All cases use the existing recipe-retriever node, including its `max(50, meals*10)`
retrieval count (70 at the default seven meals), then the production deterministic
filter. The adapter normally prefers ten candidates but accepts seven; both pool
thresholds are reported. The model-backed adapter is intentionally not invoked.

The production filter returns all hard-valid survivors, including undersized pools.
Baseline evaluation retains the original pool-size diagnostic in `filter_error`,
without invoking the production grader or changing its single retrieval attempt.
Database/model failures abort the suite rather than counting infrastructure errors
as poor retrieval.

## Metrics and interpretation

- Raw and filtered Precision@5: relevant results among the first five divided by
  **five**, even if fewer than five exist. Relevance requires all configured
  expectations plus time and exclusions. Unlabelled cases receive null, not zero.
  Averages exclude null scores. These are strict metadata proxies, not human labels.
- Usable rate: filtered/retrieved, or zero for an empty retrieval. Usable means
  passes the production hard filter, not necessarily semantically relevant.
- Time and exclusion violations: independently audit survivors with stored minutes
  and normalized ingredient phrases. Matching handles case, punctuation, and simple
  trailing-s plurals, not synonyms, ingredient families, or hidden constituents.
- Enough candidates: seven by default; preferred adapter pool: ten. Relevant-pool
  sufficiency is reported separately so hard-filter success cannot mask irrelevance.
- Dataset coverage: count matching the same relevance proxy across all 5,000 rows.
  Low coverage means fewer matches than the required pool, NOT proof a cuisine is
  absent. Missing/inconsistent tags can underestimate coverage.
- Duplicate rate: repeated normalized names / result count. Near-duplicate rate:
  fraction of results with a different earlier normalized name and SequenceMatcher
  similarity >=0.90. Reported before and after filtering. This is a title heuristic,
  not ingredient-level or culinary similarity. Empty-list rates are zero.
- Cuisine diversity: distinct tags from a documented finite `CUISINE_TAGS` set,
  tag counts, and fraction of results with at least one recognized tag. Recipes
  may have multiple tags; counts are not mutually exclusive. Untagged recipes are
  not assigned an inferred cuisine. More cuisines is not inherently better for a
  single-cuisine query.
- Provenance: retrieval's `recipe_id` must exist in the CSV; names, minutes, and
  ingredients must match. Retrieval does not yet have a `source_recipe_id` field.
  Actual adapter field preservation is **not tested by this retrieval evaluation**;
  the separate adapter integration regressions cover that contract.

Worst-case ordering: ascending filtered P@5, then relevant filtered count, then
case ID (stable tie-break). Aggregate rates are unweighted means across queries;
violation counts sum result occurrences, not unique recipes across all queries.

## Limits

Vegetarian/vegan tags are annotations, not verified dietary suitability. Easy and
comfort-food tags only approximate vague semantic intent. Main-dish tags approximate
meal suitability. Exact cuisine tags can miss relevant recipes; generic mentions
can also overstate relevance. Ingredient expectations list selected alternatives
and are not a complete ontology. No claims about nutrition or price quality are made.

Manually reviewed relevance labels would be needed for a trustworthy semantic
precision/recall benchmark, title-similarity judgments, cuisine authenticity, and
vague concepts. An LLM judge is optional, not necessary for this first baseline.
The literal-query cases evaluate semantic retrieval capabilities beyond the current
query builder's vocabulary; the three query-builder cases measure its actual output.
Keep these modes in mind before attributing all results to production query construction.


## Agentic comparison (opt-in, uses Gemini)

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -B -m debug.rag_evaluation --mode compare --cases salmon easy_comfort korean_tofu_5 --output /tmp/rag-agentic-comparison.json
```

Omit `--cases` to compare all 20; use `--mode agentic` for agentic-only evaluation.
Baseline remains the default and does not use Gemini. Benchmark definitions and
relevance metrics are identical in both modes. Generated-query scenarios still
start with the same production query-builder output. Literal scenarios start with
the same literal text; expectations are never converted to production requirements.

The debug driver calls the production retriever, hard filter, grader, rewriter,
and routing/failure functions. It stops before Recipe Adapter; it does not execute
nutrition or plan selection. The production graph's edges and adapter handoff are
covered by `debug.agentic_rag_test`. The driver receives only the original query
and user-style constraints, never expected terms/tags or relevance labels.

The grader uses Python for insufficient distinct candidate counts. With enough
candidates it calls Gemini for semantic alignment and pool repetition. The rewriter
uses Gemini, while Python retains the original intent and appends constraints.
A conservative wording guard falls back to the original intent if the proposal
contains numeric/time/exclusion directives or introduces a recognized dietary
label. This prevents specific observed overgeneralizations such as replacing
"without soy sauce" with "soy-free"; it does not prove semantic equivalence.
There are at most three retrieval calls and two rewrites. Existing API retry policy
can make multiple requests for one logical model call. Identical rewrites still
consume the bounded attempt budget. One best-attempt snapshot is preserved; no result accumulation is performed.
On success, metrics describe the selected sufficient pool; on failure they describe
the last attempted pool for diagnosis. `selected_attempt` and per-attempt traces
make that distinction explicit.

Reports include acceptance/failure separately from P@5, attempts, query history,
per-attempt grades and retrieved/filtered recipes, and hard-constraint/source-ID
audits across all attempts. A failed agentic run can have hard-valid survivors;
those survivors are scored for diagnosis but are NOT passed to the adapter.
Usable rate is final-filtered/final-retrieved, not cumulative retrieval volume.

Grader acceptance is an LLM judgment of meal usefulness, not the benchmark's strict
metadata proxy. It can accept pools with low proxy P@5 or reject sparse pools;
it is not a guarantee of measured quality. Runtime constraints are authoritative
Python state and cannot be updated by the rewriter. Query wording alone is not
proof of constraint compliance. The existing literal ingredient matcher still
has the synonym/plural limitations documented above.

Full graph recursion allowance is 40 steps to accommodate both the three-attempt
retrieval loop and the pre-existing two-revision meal-plan loop; their explicit
counters still bound each loop independently. Retrieval failure raises
`RetrievalFailure` and cannot silently finalize a relaxed plan.


## Best-attempt runtime selection

Production grading now supplies anchored 0..4 scores for top-result alignment,
meal suitability, and diversity, with evidence for each of the first five recipes.
Scores use only original intent, constraints, and retrieved content, never benchmark
labels. Python requires a positive model verdict, enough distinct names, alignment
and meal suitability >=3, and diversity >=2. The weighted runtime quality is
`3 * alignment + 2 * meal_suitability + diversity`; this is not Precision@5.

Snapshots are ranked by sufficiency first, then runtime quality; ties retain the
earlier snapshot. Strong alignment (4) ends exploration. Acceptable but improvable
alignment (3) can trigger another rewrite within the existing three-attempt limit.
At the limit, the best sufficient snapshot is restored, including its matching
query, raw pool, filtered pool, and feedback. Attempt count and complete query
history remain intact. If no attempt was sufficient, the graph fails without
passing any pool to the adapter. The state field `best_retrieval` stores one snapshot.

Scores are model judgments, so best runtime quality does not guarantee best offline
Precision@5. Evidence and scores are recorded for auditing this disagreement.
