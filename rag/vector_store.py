from langchain_chroma import Chroma

from rag.embeddings import embeddings


VECTOR_DB_PATH = "chroma_db"


def get_vector_store():

    return Chroma(
        collection_name="meal_planner_recipes",
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH
    )