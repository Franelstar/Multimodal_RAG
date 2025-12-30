# app/services/embedding_service.py
import torch
import open_clip
from PIL import Image
import numpy as np
from typing import Union, List
import io

from app.core.config import settings
from app.core.logging_config import logger
from app.core.exceptions import EmbeddingGenerationException

class EmbeddingService:
    """Service de génération d'embeddings via CLIP"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Embedding service initialized on device: {self.device}")
        
        try:
            # Charger le modèle CLIP
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                settings.clip_model_name,
                pretrained='openai'
            )
            self.tokenizer = open_clip.get_tokenizer(settings.clip_model_name)
            
            self.model = self.model.to(self.device)
            self.model.eval()  # Mode évaluation (pas d'entraînement)
            
            logger.info(f"CLIP model loaded: {settings.clip_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            raise EmbeddingGenerationException(f"Model initialization failed: {str(e)}")
    
    def generate_image_embedding(self, image: Union[Image.Image, bytes, str]) -> np.ndarray:
        """
        Génère l'embedding d'une image
        
        Args:
            image: Image PIL, bytes, ou chemin vers l'image
            
        Returns:
            Embedding normalisé (shape: embedding_dimension,)
        """
        try:
            # Convertir en Image PIL si nécessaire
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, str):
                image = Image.open(image)
            
            # Préprocessing de l'image
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Génération de l'embedding
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Conversion en numpy
            embedding = image_features.cpu().numpy().flatten()
            
            logger.debug(f"Image embedding generated: shape {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Image embedding generation failed: {str(e)}")
            raise EmbeddingGenerationException(f"Image processing error: {str(e)}")
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Génère l'embedding d'un texte
        
        Args:
            text: Texte à encoder
            
        Returns:
            Embedding normalisé (shape: embedding_dimension,)
        """
        try:
            # Tokenization
            text_tokens = self.tokenizer([text]).to(self.device)
            
            # Génération de l'embedding
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Conversion en numpy
            embedding = text_features.cpu().numpy().flatten()
            
            logger.debug(f"Text embedding generated for: '{text[:50]}...'")
            return embedding
            
        except Exception as e:
            logger.error(f"Text embedding generation failed: {str(e)}")
            raise EmbeddingGenerationException(f"Text processing error: {str(e)}")
    
    def generate_batch_embeddings(
        self, 
        items: List[Union[str, Image.Image]], 
        item_type: str = "text"
    ) -> np.ndarray:
        """
        Génère des embeddings par batch (plus efficace)
        
        Args:
            items: Liste de textes ou images
            item_type: "text" ou "image"
            
        Returns:
            Array d'embeddings (shape: len(items), embedding_dimension)
        """
        embeddings = []
        
        for item in items:
            if item_type == "text":
                emb = self.generate_text_embedding(item)
            else:
                emb = self.generate_image_embedding(item)
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcule la similarité cosinus entre deux embeddings
        
        Returns:
            Score de similarité entre 0 et 1
        """
        # Cosine similarity = dot product (embeddings déjà normalisés)
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def check_health(self) -> str:
        """Vérifie que le service fonctionne"""
        try:
            # Test rapide avec un texte simple
            test_embedding = self.generate_text_embedding("test")
            return "healthy" if test_embedding.shape[0] == settings.embedding_dimension else "unhealthy"
        except:
            return "unhealthy"