"""
Voice Embedding Generation and Management
Handles generation and storage of voice embeddings using Resemblyzer
"""

import numpy as np
from typing import Tuple, Optional
import torch
from pathlib import Path
import chromadb
from chromadb.config import Settings
import uuid
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import soundfile as sf

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    persist_directory="data/embeddings",
    anonymized_telemetry=False
))

# Initialize the voice embeddings collection
voice_collection = chroma_client.get_or_create_collection(
    name="voice_embeddings",
    metadata={"hnsw:space": "cosine"}
)

# Initialize Resemblyzer voice encoder
voice_encoder = VoiceEncoder()

def generate_voice_embedding(audio_path: str) -> Tuple[str, np.ndarray]:
    """
    Generate voice embedding from audio file using Resemblyzer.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Tuple of (embedding_id, embedding_vector)
    """
    try:
        # Load and preprocess audio
        wav = preprocess_wav(audio_path)
        
        # Generate embedding
        embedding = voice_encoder.embed_utterance(wav)
        
        # Store in ChromaDB
        embedding_id = str(uuid.uuid4())
        voice_collection.add(
            embeddings=[embedding.tolist()],
            ids=[embedding_id],
            metadatas=[{
                "audio_path": audio_path,
                "model": "resemblyzer",
                "embedding_dim": len(embedding)
            }]
        )
        
        return embedding_id, embedding
        
    except Exception as e:
        raise Exception(f"Error generating voice embedding: {str(e)}")

def get_voice_embedding(embedding_id: str) -> Optional[np.ndarray]:
    """
    Retrieve voice embedding from ChromaDB.
    
    Args:
        embedding_id: ID of the embedding to retrieve
        
    Returns:
        Embedding vector if found, None otherwise
    """
    try:
        result = voice_collection.get(ids=[embedding_id])
        if result and result['embeddings']:
            return np.array(result['embeddings'][0])
    except Exception as e:
        print(f"Error retrieving embedding: {e}")
    return None

def delete_voice_embedding(embedding_id: str) -> bool:
    """
    Delete voice embedding from ChromaDB.
    
    Args:
        embedding_id: ID of the embedding to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        voice_collection.delete(ids=[embedding_id])
        return True
    except Exception as e:
        print(f"Error deleting embedding: {e}")
        return False

def compare_embeddings(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compare two voice embeddings and return similarity score.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Similarity score between 0 and 1 (1 being identical)
    """
    # Normalize embeddings
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2)
    return float(similarity) 