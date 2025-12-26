# app/models/clip.py

from langchain.embeddings import OpenAIEmbeddings  # Exemple si tu utilises LangChain
# Ou si tu utilises OpenCLIP directement
from open_clip import create_model_and_transforms
import torch
from PIL import Image
import numpy as np

class CLIPEmbedding:
    """
    Wrapper pour transformer images ou texte en embeddings vectoriels pour Chroma.
    Peut être utilisé comme embedding_function dans Chroma.
    """

    def __init__(self, model_name="ViT-B-32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = create_model_and_transforms(
            model_name,
            pretrained="openai"
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    # -------- IMAGE EMBEDDING -------- #
    def embed_image(self, image: Image.Image) -> list[float]:
        """
        Retourne un embedding vectoriel pour une image PIL.
        """
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(img_tensor)
        # Normalisation L2
        embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding[0].cpu().numpy().tolist()

    # -------- TEXTE EMBEDDING -------- #
    def embed_text(self, text: str) -> list[float]:
        """
        Retourne un embedding vectoriel pour un texte.
        """
        import open_clip
        tokens = open_clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_text(tokens)
        embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding[0].cpu().numpy().tolist()

    # -------- INTERFACE POUR CHROMA -------- #
    def __call__(self, item):
        """
        Fonction unique utilisable comme embedding_function pour Chroma.
        Détecte le type (texte ou image PIL).
        """
        from PIL import Image

        if isinstance(item, Image.Image):
            return self.embed_image(item)
        elif isinstance(item, str):
            return self.embed_text(item)
        else:
            raise TypeError(f"Type non supporté : {type(item)}")
