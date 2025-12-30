# app/services/vectordb_service.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

from app.core.config import settings
from app.core.logging_config import logger
from app.models.artwork import SearchResult, TextChunk

class VectorDBInterface(ABC):
    """Interface abstraite pour les bases de données vectorielles"""
    
    @abstractmethod
    def create_collection(self, collection_name: str, dimension: int):
        """Crée une collection"""
        pass
    
    @abstractmethod
    def add_embeddings(
        self, 
        collection_name: str, 
        embeddings: np.ndarray, 
        metadata: List[Dict],
        ids: List[str]
    ):
        """Ajoute des embeddings avec métadonnées"""
        pass
    
    @abstractmethod
    def search(
        self, 
        collection_name: str, 
        query_embedding: np.ndarray, 
        top_k: int,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Recherche par similarité"""
        pass
    
    @abstractmethod
    def delete_by_id(self, collection_name: str, ids: List[str]):
        """Supprime des entrées par ID"""
        pass
    
    @abstractmethod
    def check_health(self) -> bool:
        """Vérifie la santé de la DB"""
        pass


class ChromaDBService(VectorDBInterface):
    """Implémentation ChromaDB"""
    
    def __init__(self):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        
        # Client ChromaDB persistant
        self.client = chromadb.PersistentClient(
            path=settings.chromadb_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        logger.info(f"ChromaDB initialized at {settings.chromadb_path}")
        
        # Collections pour images et textes
        self.images_collection = None
        self.texts_collection = None
    
    def create_collection(self, collection_name: str, dimension: int):
        """Crée ou récupère une collection ChromaDB"""
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"dimension": dimension}
            )
            
            if collection_name == "artworks_images":
                self.images_collection = collection
            elif collection_name == "artworks_descriptions":
                self.texts_collection = collection
            
            logger.info(f"Collection '{collection_name}' ready (dimension: {dimension})")
            return collection
            
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise
    
    def add_embeddings(
        self, 
        collection_name: str, 
        embeddings: np.ndarray, 
        metadata: List[Dict],
        ids: List[str]
    ):
        """Ajoute des embeddings dans ChromaDB"""
        collection = self.client.get_collection(collection_name)
        
        # ChromaDB accepte les listes Python
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        
        collection.add(
            embeddings=embeddings_list,
            metadatas=metadata,
            ids=ids
        )
        
        logger.info(f"Added {len(ids)} embeddings to '{collection_name}'")
    
    def search(
        self, 
        collection_name: str, 
        query_embedding: np.ndarray, 
        top_k: int,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Recherche dans ChromaDB"""
        collection = self.client.get_collection(collection_name)
        
        query_embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        results = collection.query(
            query_embeddings=[query_embedding_list],
            n_results=top_k,
            where=filter_dict  # Filtrage optionnel
        )
        
        # Formatter les résultats
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results
    
    def delete_by_id(self, collection_name: str, ids: List[str]):
        """Supprime des entrées dans ChromaDB"""
        collection = self.client.get_collection(collection_name)
        collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} entries from '{collection_name}'")
    
    def check_health(self) -> bool:
        """Vérifie que ChromaDB est accessible"""
        try:
            self.client.heartbeat()
            return True
        except:
            return False


class QdrantService(VectorDBInterface):
    """Implémentation Qdrant"""
    
    def __init__(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        self.client = QdrantClient(url=settings.qdrant_url)
        self.Distance = Distance
        self.VectorParams = VectorParams
        
        logger.info(f"Qdrant initialized at {settings.qdrant_url}")
    
    def create_collection(self, collection_name: str, dimension: int):
        """Crée une collection Qdrant"""
        from qdrant_client.models import Distance, VectorParams
        
        # Vérifier si la collection existe
        collections = self.client.get_collections().collections
        if collection_name not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Qdrant collection '{collection_name}' created")
    
    def add_embeddings(
        self, 
        collection_name: str, 
        embeddings: np.ndarray, 
        metadata: List[Dict],
        ids: List[str]
    ):
        """Ajoute des embeddings dans Qdrant"""
        from qdrant_client.models import PointStruct
        
        points = [
            PointStruct(
                id=ids[i],
                vector=embeddings[i].tolist(),
                payload=metadata[i]
            )
            for i in range(len(ids))
        ]
        
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logger.info(f"Added {len(ids)} embeddings to Qdrant '{collection_name}'")
    
    def search(
        self, 
        collection_name: str, 
        query_embedding: np.ndarray, 
        top_k: int,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Recherche dans Qdrant"""
        from qdrant_client.models import Filter
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=Filter(**filter_dict) if filter_dict else None
        )
        
        formatted_results = [
            {
                'id': hit.id,
                'metadata': hit.payload,
                'score': hit.score
            }
            for hit in results
        ]
        
        return formatted_results
    
    def delete_by_id(self, collection_name: str, ids: List[str]):
        """Supprime des entrées dans Qdrant"""
        self.client.delete(
            collection_name=collection_name,
            points_selector=ids
        )
        logger.info(f"Deleted {len(ids)} entries from Qdrant '{collection_name}'")
    
    def check_health(self) -> bool:
        """Vérifie que Qdrant est accessible"""
        try:
            self.client.get_collections()
            return True
        except:
            return False


class VectorDBService:
    """Service unifié de gestion de base de données vectorielle"""
    
    def __init__(self):
        # Choisir l'implémentation selon la config
        if settings.vector_db_type == "chromadb":
            self.db = ChromaDBService()
        elif settings.vector_db_type == "qdrant":
            self.db = QdrantService()
        else:
            raise ValueError(f"Unknown vector DB type: {settings.vector_db_type}")
        
        # Initialiser les collections
        self._init_collections()
    
    def _init_collections(self):
        """Initialise les collections nécessaires"""
        self.db.create_collection(
            "artworks_images", 
            settings.embedding_dimension
        )
        self.db.create_collection(
            "artworks_descriptions", 
            settings.embedding_dimension
        )
        logger.info("Vector DB collections initialized")
    
    def index_artwork_image(
        self, 
        artwork_id: str, 
        image_embedding: np.ndarray, 
        metadata: Dict
    ):
        """Indexe l'embedding d'une image d'œuvre"""
        self.db.add_embeddings(
            collection_name="artworks_images",
            embeddings=np.array([image_embedding]),
            metadata=[metadata],
            ids=[artwork_id]
        )
    
    def index_artwork_text_chunks(
        self, 
        chunks: List[TextChunk], 
        embeddings: np.ndarray
    ):
        """Indexe les chunks de texte d'une œuvre"""
        metadata = [
            {
                "artwork_id": chunk.artwork_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks
            }
            for chunk in chunks
        ]
        
        ids = [chunk.chunk_id for chunk in chunks]
        
        self.db.add_embeddings(
            collection_name="artworks_descriptions",
            embeddings=embeddings,
            metadata=metadata,
            ids=ids
        )
    
    def search_similar_images(
        self, 
        query_embedding: np.ndarray, 
        top_k: int
    ) -> List[SearchResult]:
        """Recherche les images les plus similaires"""
        results = self.db.search(
            collection_name="artworks_images",
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # Convertir en SearchResult
        search_results = []
        for result in results:
            metadata = result['metadata']
            
            # ChromaDB renvoie 'distance', Qdrant renvoie 'score'
            # Distance : plus bas = plus similaire
            # Score : plus haut = plus similaire
            similarity = 1 - result.get('distance', 0) if 'distance' in result else result.get('score', 0)
            
            search_results.append(SearchResult(
                artwork_id=result['id'],
                title=metadata.get('title', ''),
                artist=metadata.get('artist', ''),
                image_url=metadata.get('image_url', ''),
                similarity_score=similarity
            ))
        
        return search_results
    
    def get_artwork_context(
        self, 
        artwork_id: str, 
        question: str,
        embedding_service = None
    ) -> List[str]:
        """
        Récupère les chunks de contexte pertinents pour une question
        
        Si embedding_service est fourni, recherche sémantique par la question.
        Sinon, récupère tous les chunks de l'artwork.
        """
        if embedding_service:
            # Recherche sémantique dans les chunks
            question_embedding = embedding_service.generate_text_embedding(question)
            
            results = self.db.search(
                collection_name="artworks_descriptions",
                query_embedding=question_embedding,
                top_k=settings.top_k_results,
                filter_dict={"artwork_id": artwork_id}
            )
        else:
            # Récupérer tous les chunks de l'artwork
            # Note: nécessite une méthode de filtrage
            results = self.db.search(
                collection_name="artworks_descriptions",
                query_embedding=np.zeros(settings.embedding_dimension),  # Dummy query
                top_k=100,
                filter_dict={"artwork_id": artwork_id}
            )
        
        # Extraire le contenu des chunks
        context_chunks = [result['metadata']['content'] for result in results]
        return context_chunks
    
    def delete_artwork(self, artwork_id: str):
        """Supprime toutes les données d'une œuvre"""
        # Supprimer l'image
        self.db.delete_by_id("artworks_images", [artwork_id])
        
        # Supprimer les chunks de texte (nécessite de les lister d'abord)
        # TODO: implémenter listage des chunks par artwork_id
        logger.info(f"Deleted artwork {artwork_id}")
    
    def check_health(self) -> str:
        """Vérifie la santé de la base vectorielle"""
        return "healthy" if self.db.check_health() else "unhealthy"