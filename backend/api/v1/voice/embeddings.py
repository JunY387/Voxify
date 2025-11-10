"""
Voice Embedding Generation and Management
Handles generation and storage of voice embeddings using Resemblyzer
"""

import numpy as np
from typing import Tuple, Optional
import torch
from pathlib import Path
import uuid
import base64
import os
import requests
from resemblyzer import VoiceEncoder, preprocess_wav
import soundfile as sf
from database.vector_config import create_vector_db


# Initialize Resemblyzer voice encoder (local fallback)
voice_encoder = VoiceEncoder()

# Get the existing ChromaVectorDB instance
vector_db = create_vector_db()

# Modal remote embedding service configuration
VOICE_EMBED_USE_REMOTE = os.getenv("VOICE_EMBED_USE_REMOTE", "true").lower() == "true"
VOICE_EMBED_REMOTE_URL = os.getenv(
    "VOICE_EMBED_REMOTE_URL",
    "https://nnnnnjun-yang--voxify-voice-embed-fastapi-app.modal.run/extract_embedding"
)
VOICE_EMBED_TIMEOUT = int(os.getenv("VOICE_EMBED_TIMEOUT", "60"))


def _generate_embedding_remote(audio_path: str) -> np.ndarray:
    """
    Generate voice embedding using remote Modal service.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Embedding vector as numpy array
    """
    try:
        print(f"[DEBUG] Generating embedding via REMOTE Modal service...")
        print(f"[DEBUG] Remote URL: {VOICE_EMBED_REMOTE_URL}")
        
        # Read audio file and encode to base64
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        print(f"[DEBUG] Encoded audio size: {len(audio_b64)} characters")
        
        # Prepare request payload
        payload = {
            "audio_b64": audio_b64,
            "sample_rate": 16000
        }
        
        # Call remote API
        response = requests.post(
            VOICE_EMBED_REMOTE_URL,
            json=payload,
            timeout=VOICE_EMBED_TIMEOUT
        )
        
        if response.status_code != 200:
            error_detail = response.json().get('detail', 'Unknown error') if response.text else 'No response'
            raise Exception(f"Remote API returned status {response.status_code}: {error_detail}")
        
        # Parse response
        result = response.json()
        embedding_list = result.get('embedding')
        if not embedding_list:
            raise Exception("Remote API did not return embedding data")
        
        embedding = np.array(embedding_list)
        print(f"[DEBUG] Remote embedding generated successfully, shape: {embedding.shape}")
        
        return embedding
        
    except requests.exceptions.Timeout:
        raise Exception(f"Remote embedding API timed out after {VOICE_EMBED_TIMEOUT} seconds")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Remote embedding API request failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Remote embedding generation failed: {str(e)}")


def _generate_embedding_local(audio_path: str) -> np.ndarray:
    """
    Generate voice embedding using local Resemblyzer.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Embedding vector as numpy array
    """
    print(f"[DEBUG] Generating embedding via LOCAL Resemblyzer...")
    
    # Load and preprocess audio
    wav = preprocess_wav(audio_path)
    print(f"[DEBUG] Preprocessed audio shape: {wav.shape}")
    
    # Generate embedding
    embedding = voice_encoder.embed_utterance(wav)
    print(f"[DEBUG] Local embedding generated successfully, shape: {embedding.shape}")
    
    return embedding


def generate_voice_embedding(audio_path: str, user_id: str = None, **extra_metadata) -> Tuple[str, np.ndarray]:
    """
    Generate voice embedding from audio file using Resemblyzer (Remote or Local).

    Args:
        audio_path: Path to the audio file
        user_id: User ID for the embedding (optional)
        **extra_metadata: Additional metadata for the embedding


    Returns:
        Tuple of (embedding_id, embedding_vector)
    """
    try:
        print(f"[DEBUG] Generating embedding for: {audio_path}")
        print(f"[DEBUG] Mode: {'REMOTE (Modal)' if VOICE_EMBED_USE_REMOTE else 'LOCAL (CPU)'}")

        # Generate embedding (remote or local)
        if VOICE_EMBED_USE_REMOTE:
            try:
                embedding = _generate_embedding_remote(audio_path)
            except Exception as e:
                print(f"[DEBUG] Remote embedding failed: {e}")
                print(f"[DEBUG] Falling back to local embedding...")
                embedding = _generate_embedding_local(audio_path)
        else:
            embedding = _generate_embedding_local(audio_path)
        
        print(f"[DEBUG] Generated embedding shape: {embedding.shape}")
        
        # Store in ChromaDB using the existing vector_db
        embedding_id = str(uuid.uuid4())
        print(f"[DEBUG] Storing embedding with ID: {embedding_id}")

        # Use the existing ChromaVectorDB's add_voice_embedding method
        vector_db.add_voice_embedding(
            voice_sample_id=embedding_id,
            embedding=embedding.tolist(),
            metadata={
                "audio_path": audio_path,
                "model": "resemblyzer",
                "embedding_dim": len(embedding),
                "user_id": user_id,
                **extra_metadata,
            },
        )

        print("[DEBUG] Embedding stored successfully")

        # Verify storage
        verification = vector_db.get_embedding(embedding_id)
        if verification and verification.get("embeddings"):
            print("[DEBUG] ✅ Verification successful - embedding can be retrieved")
        else:
            print("[DEBUG] ❌ WARNING: Embedding storage verification failed")
            print(f"[DEBUG] Verification result: {verification}")

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
        print(f"[DEBUG] Attempting to retrieve embedding: {embedding_id}")
        result = vector_db.get_embedding(embedding_id)
        print(f"[DEBUG] ChromaDB result keys: {result.keys() if result else 'None'}")

        if result and result.get("embeddings") and len(result["embeddings"]) > 0:
            embedding_data = result["embeddings"][0]
            if embedding_data:
                embedding_array = np.array(embedding_data)
                print(f"[DEBUG] Successfully retrieved embedding, shape: {embedding_array.shape}")
                return embedding_array
            else:
                print(f"[DEBUG] Embedding data is empty for ID: {embedding_id}")
        else:
            print(f"[DEBUG] No embeddings found in result for ID: {embedding_id}")
            if result:
                print(f"[DEBUG] Result structure: embeddings={result.get('embeddings')}, ids={result.get('ids')}")

    except Exception as e:
        print(f"[DEBUG] Error retrieving embedding {embedding_id}: {e}")
        import traceback

        traceback.print_exc()
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
        vector_db.delete_embedding(embedding_id)
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


def debug_chromadb_status():
    """Debug function to check ChromaDB status"""
    try:
        collection_info = vector_db.get_collection_count()
        print(f"[DEBUG] ChromaDB collection info: {collection_info}")

        # Get all embeddings
        all_data = vector_db.get_collection().get()
        print(f"[DEBUG] All stored IDs: {all_data.get('ids', [])}")
        print(f"[DEBUG] Total embeddings in collection: {len(all_data.get('embeddings', []))}")

        return collection_info.get("item count", 0)
    except Exception as e:
        print(f"[DEBUG] Error checking ChromaDB status: {e}")
        return -1
