from langchain_chroma import Chroma

from rag.embeddings import embeddings


from rag.config import VECTOR_DB_PATH, COLLECTION_NAME


def get_vector_store():

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_PATH)
    )
