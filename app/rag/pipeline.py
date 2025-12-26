# app/rag/pipeline.py

from app.storage.chroma import get_or_create_collection
from app.models.clip import clip_embedding_model

collection = get_or_create_collection(
    name="Multimodal_arts",
    embedding_function=clip_embedding_model
)