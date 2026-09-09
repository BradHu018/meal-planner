from rag.retriever import (
    retrieve_recipes
)


queries = [
    "Chinese tofu dinner",
    "quick chicken rice meal",
    "spicy Korean pork meal",
    "vegetarian noodle dinner",
]

results = retrieve_recipes(
    queries,
    k=5
)


print("\n=== QUERY ===")
print(queries)


print(
    "\n=== RETRIEVED RECIPES ==="
)


for index, recipe in enumerate(
    results,
    start=1
):

    print(
        f"\n{index}. "
        f"{recipe['name']}"
    )

    print(
        "Cooking time:",
        recipe["minutes"],
        "minutes"
    )