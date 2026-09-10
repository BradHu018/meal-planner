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

The production filter raises on undersized pools. Evaluation records that error,
then calls the same filter on the same results with `meals_needed=0` solely to
observe survivors. This is not another retrieval attempt and does not relax time
or ingredient constraints. Database/model failures abort the suite rather than
silently counting an infrastructure failure as poor retrieval.

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
