import ast
import csv
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def recipe_ingredients():
    """Read exact ingredient lists by ID; compatible with the existing index."""
    path = Path(__file__).resolve().parents[1] / "data/recipes/recipes_rag.csv"
    with path.open(newline="", encoding="utf-8") as source:
        return {
            row["id"]: ast.literal_eval(row["ingredients"])
            for row in csv.DictReader(source)
        }


def get_vector_store():
    # Load embeddings only when retrieval is actually requested.
    from rag.vector_store import get_vector_store as open_store
    return open_store()


def retrieve_recipes(
    query: str,
    k: int = 5
):

    vector_store = (
        get_vector_store()
    )

    documents = (
        vector_store.similarity_search(
            query,
            k=k
        )
    )

    results = []

    for document in documents:

        results.append({
            "recipe_id":
                document.metadata[
                    "recipe_id"
                ],

            "name":
                document.metadata[
                    "name"
                ],

            "minutes":
                document.metadata[
                    "minutes"
                ],

            "ingredients": recipe_ingredients().get(str(document.metadata["recipe_id"]), []),

            "content":
                document.page_content
        })

    return results
