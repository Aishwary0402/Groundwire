"""
Zone 3: Knowledge Base — ChromaDB wrapper.

Single shared vectorstore instance, persisted to disk under backend/data/chroma
so the index survives restarts during development.
"""

from functools import lru_cache
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

PERSIST_DIR = "data/chroma"
COLLECTION_NAME = "groundwire_documents"


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
