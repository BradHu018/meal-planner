import ast
import shutil
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from rag.vector_store import (
    get_vector_store,
    VECTOR_DB_PATH
)


RECIPE_FILE = "data/recipes/recipes_rag.csv"


def parse_list(value):
    """
    Convert CSV string representation of a list
    back into an actual Python list.
    """

    if isinstance(value, list):
        return value

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return parsed

    except (ValueError, SyntaxError, TypeError):
        pass

    return []


def load_recipe_documents():

    df = pd.read_csv(RECIPE_FILE)

    documents = []
    ids = []

    for _, recipe in df.iterrows():

        ingredients = parse_list(
            recipe["ingredients"]
        )

        tags = parse_list(
            recipe["tags"]
        )

        steps = parse_list(
            recipe["steps"]
        )

        ingredients_text = ", ".join(
            ingredients
        )

        tags_text = ", ".join(
            tags
        )

        steps_text = " ".join(
            steps
        )

        content = f"""
Recipe name: {recipe["name"]}

Cooking time: {recipe["minutes"]} minutes

Ingredients:
{ingredients_text}

Description:
{recipe["description"]}

Tags:
{tags_text}

Instructions:
{steps_text}
""".strip()

        document = Document(
            page_content=content,

            metadata={
                "recipe_id": str(
                    recipe["id"]
                ),
                "name": recipe["name"],
                "minutes": int(
                    recipe["minutes"]
                )
            }
        )

        documents.append(
            document
        )

        ids.append(
            str(recipe["id"])
        )

    return documents, ids


def rebuild_vector_store():

    db_path = Path(
        VECTOR_DB_PATH
    )

    # During development:
    # delete old DB so we don't duplicate recipes
    if db_path.exists():
        shutil.rmtree(
            db_path
        )

    documents, ids = (
        load_recipe_documents()
    )

    print(
        f"Loaded {len(documents)} documents."
    )

    vector_store = (
        get_vector_store()
    )

    vector_store.add_documents(
        documents=documents,
        ids=ids
    )

    print(
        f"Added {len(documents)} recipes "
        f"to Chroma."
    )


if __name__ == "__main__":
    rebuild_vector_store()