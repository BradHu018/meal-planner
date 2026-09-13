# Controlled 5k versus 20k corpus experiment

Corpus selection is process-wide: `MEAL_PLANNER_CORPUS=5k` (default) or `20k`.
It selects the matching CSV, Chroma directory, and collection together. Start a
fresh process when changing profiles; loaded embeddings and ingredient caches
are not switched in a running process.

| Profile | CSV | Chroma directory | Collection |
|---|---|---|---|
| 5k | data/recipes/recipes_rag.csv | chroma_db | meal_planner_recipes |
| 20k | data/recipes/recipes_rag_20k.csv | chroma_db_20k | meal_planner_recipes_20k |

The 20k CSV and index are ignored by Git. The small corpus manifest records hashes,
seed, row counts, exact baseline reproduction, and subset verification. Preparation
and ingestion refuse to overwrite existing targets. Ingestion uses the original
one-document-per-recipe format, metadata/IDs, embedding model, normalization, and
Chroma defaults. Insertions are batched to stay within Chroma's batch limits.

## Reproduce

Download the original `RAW_recipes.csv` from the Food.com dataset linked by
https://cseweb.ucsd.edu/~jmcauley/datasets.html#foodcom
(Kaggle: shuyangli94/food-com-recipes-and-user-interactions).

```bash
.venv/bin/python -B -m debug.corpus_experiment prepare --raw /path/to/RAW_recipes.csv
MEAL_PLANNER_CORPUS=20k HF_HUB_OFFLINE=1 .venv/bin/python -B -m rag.ingest
MEAL_PLANNER_CORPUS=5k HF_HUB_OFFLINE=1 .venv/bin/python -B -m debug.rag_evaluation --mode agentic --output /tmp/corpus-5k-agentic.json
MEAL_PLANNER_CORPUS=20k HF_HUB_OFFLINE=1 .venv/bin/python -B -m debug.rag_evaluation --mode agentic --output /tmp/corpus-20k-agentic.json
.venv/bin/python -B -m debug.corpus_experiment compare /tmp/corpus-5k-agentic.json /tmp/corpus-20k-agentic.json --output /tmp/corpus-comparison.json
```

Preparation applies the original cleaning rules with seed 42, verifies that the
5k sample reproduces every existing CSV field, and verifies that the 20k sample
contains all 5k recipe IDs. The source is read using universal newline handling:
its CRLF descriptions then match the baseline's LF descriptions. No recipe text
is otherwise normalized or changed. No benchmark-driven selection is performed.

Evaluation remains debug-only; the corpus config has no benchmark dependencies.
All cases, metrics, grader/rewriter prompts, and hard filters are unchanged.
Retrieval failures are counted as failed cases, while the final pool remains
scored for diagnosis, as in the existing evaluator. Per-case changes are classified
by filtered P@5. Reports also show corpus-wide relevant counts and relevant-pool
counts to distinguish available metadata coverage from retrieval performance.

This is one fixed sample expansion and one stochastic agentic run per corpus, not
a multi-seed statistical study. Larger coverage with unchanged ranking indicates
remaining retrieval/grading issues; improved ranking alone cannot prove coverage
was the only bottleneck. Low coverage uses the existing strict metadata proxy,
not manually judged relevance. Hard-constraint or provenance violations would
invalidate a claimed quality improvement.

Regression tests (manual live critic/retrieval scripts are not unit tests):

```bash
.venv/bin/python -B -m unittest debug.corpus_experiment_test debug.agentic_rag_test debug.rag_integration_test debug.rag_evaluation_test -v
```
