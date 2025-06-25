"""
Voxify Vector Database Configuration
Chroma Collections Setup for Voice Embeddings
"""

import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional, Any
import os
from pathlib import Path
import logging
import json
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorDBConfig:
    """Configuration for Chroma vector database collections"""
    
    # Collection configurations
    VOICE_EMBEDDINGS_CONFIG = {
        "name": "voice_embeddings",
        "metadata": {
            "description": "Voice sample embeddings",
            "embedding_model": "coqui-vits-voice-encoder",  # ASSUMING WE'RE USING COQUI
            "dimension": 768,
            "distance_metric": "cosine",
            "feature_type": "voice_spectral",
            "created_version": "1.0"
        }
    }

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        embedding_function: Optional[Any] = None,
        collection_name: str = "voice_embeddings",
        distance_metric: str = "cosine"
    ):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.distance_metric = distance_metric

class ChromaVectorDB:
    """Chroma vector database manager for Voxify platform"""
    
    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        persist_directory: str = "data/chroma_db"
    ):
        self.config = config or VectorDBConfig(persist_directory=persist_directory)
        self.persist_directory = self.config.persist_directory
        
        # ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # initialize ChromaDB client (using persistent mode)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric}
        )
        
        # Initialize collections
        self.voice_embeddings_collection = None
        self._initialize_collection()
    

    def _initialize_collection(self):
        """Initialize required collection(s)"""

        config = VectorDBConfig.VOICE_EMBEDDINGS_CONFIG
        
        collection_name = config["name"]
        try:
            # Get or create collection
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=config["metadata"]
            )
            self.voice_embeddings_collection = collection
            logger.info(f"Initialized collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize collection {collection_name}: {str(e)}")
            raise
    

    def get_collection(self) -> chromadb.Collection:
        """Returns the voice_embeddings collection"""

        return self.voice_embeddings_collection
    

    def add_voice_embedding(self, 
                           voice_sample_id: str,
                           embedding: List[float],
                           metadata: Dict[str, Any]) -> None:
        """
        Add voice embedding to the voice_embeddings collection
        
        Parameters
        ----------
        voice_sample_id : str
            Unique identifier for the voice sample (UUID)
        embedding : List[float]
            Voice feature vector (768 dimensions)
        metadata : Dict[str, Any]
            Associated metadata for the voice sample
        """
        collection = self.get_collection()
        
        # Prepare document text for search
        document = f"{metadata.get('name', '')} {metadata.get('description', '')} {metadata.get('language', '')}"
        
        # Enhanced metadata with additional fields
        enhanced_metadata = {
            "user_id": metadata.get("user_id"),
            "language": metadata.get("language", "en-US"),
            "duration": float(metadata.get("duration", 0.0)),
            "quality_score": float(metadata.get("quality_score", 0.0)),
            "sample_rate": int(metadata.get("sample_rate", 22050)),
            "gender": metadata.get("gender"),
            "age_group": metadata.get("age_group"),
            "accent": metadata.get("accent"),
            "is_public": metadata.get("is_public", False),
            "created_at": metadata.get("created_at")
        }
        
        # Remove None values
        enhanced_metadata = {k: v for k, v in enhanced_metadata.items() if v is not None}
        
        collection.add(
            ids=[voice_sample_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[enhanced_metadata]
        )
        
        logger.debug(f"Added voice embedding for sample: {voice_sample_id}")


    def get_embedding(self, sample_id: str) -> Dict[str, List]:
        """
        Search for the voice embedding with the given sample_id
        
        Parameters
        ----------
        sample_id : str
            The id of the sample to be looked up.

        Returns
        -------
        Dict[str, List]
            A dictionary containing the embedding, metadata, and document for the given sample ID.
        """

        return self.get_collection().get(ids=[sample_id])

    
    def update_embedding_metadata(self,
                                 embedding_id: str,
                                 new_metadata: Dict[str, Any]) -> None:
        """Update metadata for an existing embedding"""
        collection = self.get_collection()
        
        # Get current record
        current = collection.get(ids=[embedding_id], include=["metadatas"])
        if not current["ids"]:
            raise ValueError(f"Embedding {embedding_id} not found in voice_embeddings")
        
        # Merge metadata
        current_metadata = current["metadatas"][0]
        updated_metadata = {**current_metadata, **new_metadata}
        
        collection.update(
            ids=[embedding_id],
            metadatas=[updated_metadata]
        )
        
        logger.debug(f"Updated metadata for {embedding_id} in voice_embeddings")
    

    def delete_embedding(self, embedding_id: str) -> None:
        """Delete an embedding from the voice_embeddings collection"""
        collection = self.get_collection()
        collection.delete(ids=[embedding_id])
        logger.debug(f"Deleted embedding {embedding_id} from voice_embeddings")
    

    def get_collection_count(self) -> Dict[str, Any]:
        """Get the count of embeddings for the voice_embeddings collection"""
        collection = self.get_collection()
        count = collection.count()
        
        return {
            "name": "voice_embeddings",
            "item count": count,
            "metadata": collection.metadata
        }
    

    def close(self):
        """Close database connections and persist data"""
        if hasattr(self.client, 'persist'):
            self.client.persist()
        logger.info("Vector database connections closed and data persisted")


# Factory function for easy initialization
def create_vector_db(config_path: str = None) -> ChromaVectorDB:
    """
    Create and initialize vector database instance.

    Parameters
    ----------
    config_path : str, optional
        Path to configuration file (for future use).
        
    Returns
    -------
    ChromaVectorDB
        Initialized vector database instance.
    """
    # Load configs
    configs = load_config(config_path)
    
    return ChromaVectorDB(
        config=configs.get("config"),
        persist_directory=configs.get("persist_directory")
    ) 


def load_config(config_path: str = None) -> Dict[str, Any]:
    config = {}
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)

    return {

        "persist_directory": os.getenv("VECTOR_DB_PATH") or config.get("persist_directory", "data/chroma_db"),
        "host": os.getenv("CHROMA_HOST") or config.get("host"),
        "port": int(os.getenv("CHROMA_PORT") or config.get("port", 8000)),
    }

