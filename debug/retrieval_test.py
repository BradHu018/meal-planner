from rag.retriever import retrieve_recipes


queries = [
    "Chinese tofu dinner",
    "quick chicken rice meal",
    "spicy Korean pork meal",
    "vegetarian noodle dinner",
]


for query in queries:

    results = retrieve_recipes(
        query,
        k=5
    )

    print("\n==============================")
    print("QUERY:")
    print(query)

    print("\nRETRIEVED RECIPES:")

    for index, recipe in enumerate(
        results,
        start=1
    ):
        print(
            f"\n{index}. {recipe['name']}"
        )

        print(
            "Cooking time:",
            recipe["minutes"],
            "minutes"
        )