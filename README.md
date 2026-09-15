# Multi-Agent Nutrition-Aware Meal Planning System

A meal planning project I built with **LangGraph** to explore multi-agent workflows, RAG, and real-world data integration.

The system takes a user's food preferences, cooking constraints, budget, pantry ingredients, and supplied nutrition targets, then builds a weekly meal plan from real recipes.

Instead of relying on the LLM for everything, I tried to keep exact calculations in Python and use the LLM mainly for reasoning and evaluation.

## How it works

At a high level:

```text
User Preferences
      |
      v
   Planner
      |
      v
Recipe Retrieval
      |
      v
Retrieval Grader
   /       \
good       weak
 |           |
 v           v
Adapter   Rewrite Query
              |
              +----> Retrieve Again
      |
      v
Nutrition + Portioning
      |
      +-------------+-------------+
      |             |             |
    Taste         Budget       Balance
      |             |             |
      +-------------+-------------+
                    |
                    v
                 Optimizer
                    |
                    v
                  Critic
                 /      \
             approve    revise
                |          |
                v          +----> Critic
             Final Plan
```

The retrieval loop and revision loop are both bounded so they cannot run forever.

## Agentic RAG

Recipes are retrieved from a **ChromaDB vector store** containing 20,000 recipes from the Food.com dataset.

The first version used a normal one-shot RAG flow:

```text
query -> retrieve -> filter -> continue
```

I later changed it to an agentic retrieval loop:

```text
query
  -> retrieve
  -> filter
  -> grade results
  -> rewrite the query if results are weak
  -> retrieve again
```

The system keeps track of previous searches and can preserve the best candidate pool if a later rewrite produces worse results.

It makes at most **3 retrieval attempts** before failing instead of silently relaxing the user's constraints.

### Retrieval results

I created a fixed 20-query benchmark to compare the original RAG pipeline with the agentic version.

| Metric | Baseline | Agentic RAG |
|---|---:|---:|
| Average filtered Precision@5 | 0.25 | 0.31 |
| Cooking-time violations | 0 | 0 |
| Excluded-ingredient violations | 0 | 0 |
| Invalid recipe IDs | 0 | 0 |

`Precision@5` measures how many of the first five retrieved recipes satisfy the benchmark's relevance checks.

I also tested whether the original 5,000-recipe corpus was limiting retrieval.

| Metric | 5k recipes | 20k recipes |
|---|---:|---:|
| Queries with enough relevant candidates | 6 / 20 | 10 / 20 |
| Low-coverage queries | 9 / 20 | 5 / 20 |
| Retrieval failures | 8 | 3 |
| Average filtered Precision@5 | 0.30 | 0.30 |

The larger corpus gave the system a lot more usable recipes, although ranking quality stayed about the same. That helped separate **dataset coverage problems** from **retrieval/ranking problems**.

## Nutrition and portioning

Nutrition is calculated using data from the **Canadian Nutrient File** rather than asking the LLM to estimate it.

For each recipe:

```text
ingredients
   ->
CNF nutrient lookup
   ->
calories + protein
   ->
portion adjustment
```

The supplied nutrition targets are treated as software inputs. The project does not try to determine a person's nutritional needs.

## Meal evaluation

After recipes are enriched with nutrition data, three parts of the graph run in parallel:

- **Taste Agent** — checks how well recipes match the user's preferences
- **Budget Agent** — calculates estimated recipe costs
- **Balance Agent** — looks at nutrition fit, variety, and repeated ingredients or meal styles

The optimizer combines those results and selects the weekly meals.

A critic then reviews the plan. If there is an issue that can actually be fixed using the existing candidate recipes, the revision node swaps recipes and sends the plan back through the critic.

## Grocery pricing

The current pricing system uses **Statistics Canada food-price data**.

Pricing is handled in Python rather than by the LLM. Pantry items are excluded from the grocery list, and ingredients with missing price data are tracked instead of being silently assigned a price of $0.

I'm currently working on adding an **MCP-based grocery pricing fallback** for ingredients that cannot be resolved locally.

## Recipe provenance

Every adapted recipe keeps its original Food.com recipe ID.

The LLM can choose and adapt a retrieved recipe, but Python restores the canonical recipe name and cooking time from the source data.

This prevents the model from accidentally changing the identity of the retrieved recipe.

## Tech stack

**Language**
- Python

**AI / orchestration**
- LangGraph
- LangChain
- Gemini
- Pydantic

**RAG**
- ChromaDB
- Hugging Face Sentence Transformers
- `all-MiniLM-L6-v2`

**Data**
- Pandas
- Food.com Recipes dataset
- Canadian Nutrient File
- Statistics Canada food-price data

## Project structure

```text
meal-planner/
├── main.py
├── workflow.py
├── prompts.py
│
├── rag/
│   ├── embeddings.py
│   ├── retriever.py
│   ├── vector_store.py
│   └── ingest.py
│
├── data/
│   ├── nutrition.py
│   ├── prices.py
│   ├── cnf/
│   └── recipes/
│
└── debug/
    ├── rag_benchmarks.json
    ├── rag_evaluation.py
    ├── agentic_rag_test.py
    └── corpus_experiment.py
```

## Setup

Clone the repo:

```bash
git clone https://github.com/BradHu018/meal-planner.git
cd meal-planner
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your Gemini API key:

```env
GOOGLE_API_KEY=your_api_key
```

Then run:

```bash
python main.py
```

## Running the RAG evaluation

Baseline vs agentic retrieval:

```bash
HF_HUB_OFFLINE=1 python -B -m debug.rag_evaluation \
  --mode compare
```

To evaluate the 20k corpus:

```bash
MEAL_PLANNER_CORPUS=20k \
HF_HUB_OFFLINE=1 \
python -B -m debug.rag_evaluation \
  --mode agentic
```

## Current limitations

A few things I'm still working on:

- Food.com cuisine tags are incomplete
- some very specific requests still have poor corpus coverage
- ingredient names do not always map cleanly between the recipe, nutrition, and pricing datasets
- grocery prices are still missing for some ingredients
- vector similarity does not always rank the best recipes near the top

## Grocery Pricing with MCP

The project now includes an **MCP-based grocery pricing fallback** for ingredients that cannot be resolved using the local Statistics Canada price dataset.

The pricing flow is:

```text
Selected meals
    ↓
Aggregate grocery ingredients
    ↓
Try Statistics Canada pricing
    ↓
price found?
 ┌───────────────┴───────────────┐
 yes                             no
  ↓                               ↓
use local price              MCP client
                                  ↓
                              MCP server
                                  ↓
                        external grocery provider
                                  ↓
                         package-level quote
 └───────────────────────┬───────────────
                         ↓
              deterministic Python pricing

## Data sources

- [Food.com Recipes and Interactions](https://cseweb.ucsd.edu/~jmcauley/datasets.html#foodcom)
- Canadian Nutrient File
- Statistics Canada food-price data
