from rag.vector_store import (
    get_vector_store
)


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

            "content":
                document.page_content
        })

    return results