# app/storage/chroma.py

import chromadb
from chromadb.utils import embedding_functions

# Création du client Chroma persistent
chromadb_client = chromadb.PersistentClient(path="Arts-store-vdb")

def get_or_create_collection(
    name: str,
    embedding_function=None
):
    """
    Récupère une collection existante ou en crée une nouvelle
    """
    return chromadb_client.get_or_create_collection(
        name=name,
        embedding_function=embedding_function
    )
