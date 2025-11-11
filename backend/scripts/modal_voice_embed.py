"""
Modal Voice Embedding Service
Extracts voice embeddings using resemblyzer on GPU for faster processing
"""

import modal
import base64
import io
import numpy as np

# Create Modal app
app = modal.App("voxify-voice-embed")

# Define the container image with resemblyzer
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "ffmpeg", 
        "libsndfile1", 
        "libsndfile1-dev",
        "libavcodec-extra",
        "libavformat-dev",
        "libavutil-dev",
        "libswresample-dev"
    )
    .pip_install(
        [
            "resemblyzer==0.1.4",
            "librosa==0.10.1",
            "soundfile==0.12.1",
            "numpy==1.26.4",
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "fastapi",
            "pydantic",
            "audioread",  # Additional audio backend for librosa
        ]
    )
)


@app.function(
    image=image,
    cpu=2.0,  # resemblyzer is CPU-based, 2 vCPU is sufficient
    timeout=300,
    scaledown_window=300,  # 5分钟无活动后自动缩容
    min_containers=0,  # 无请求时完全停止，按需启动
    memory=2048,  # 2GB memory
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import librosa
    from resemblyzer import VoiceEncoder
    
    # Initialize FastAPI
    fastapi_app = FastAPI()
    
    # Initialize resemblyzer encoder (global, loaded once)
    print("Loading Resemblyzer voice encoder...")
    encoder = VoiceEncoder(device="cpu")
    print("Resemblyzer encoder loaded successfully!")
    
    class EmbeddingRequest(BaseModel):
        audio_b64: str
        sample_rate: int = 16000
    
    class EmbeddingResponse(BaseModel):
        embedding: list
        shape: list
    
    @fastapi_app.post("/extract_embedding", response_model=EmbeddingResponse)
    async def extract_embedding(request: EmbeddingRequest):
        """
        Extract voice embedding from audio file
        
        Args:
            audio_b64: Base64 encoded audio data (WAV format recommended)
            sample_rate: Target sample rate for processing (default: 16000)
        
        Returns:
            embedding: Voice embedding as a list of floats
            shape: Shape of the embedding array
        """
        try:
            print(f"Received embedding extraction request, target SR: {request.sample_rate}")
            
            # Decode audio from base64
            try:
                audio_bytes = base64.b64decode(request.audio_b64)
                print(f"Decoded audio size: {len(audio_bytes)} bytes")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 audio data: {str(e)}")
            
            # Load audio with librosa
            try:
                audio_io = io.BytesIO(audio_bytes)
                wav, sr = librosa.load(audio_io, sr=request.sample_rate, mono=True)
                print(f"Loaded audio: shape={wav.shape}, sample_rate={sr}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load audio: {str(e)}")
            
            # Validate audio
            if len(wav) == 0:
                raise HTTPException(status_code=400, detail="Audio file is empty")
            
            # Extract embedding using resemblyzer
            try:
                print("Extracting voice embedding with resemblyzer...")
                embedding = encoder.embed_utterance(wav)
                print(f"Embedding extracted successfully: shape={embedding.shape}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")
            
            # Convert to list for JSON serialization
            embedding_list = embedding.tolist()
            
            return EmbeddingResponse(
                embedding=embedding_list,
                shape=list(embedding.shape)
            )
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"Unexpected error in extract_embedding: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    @fastapi_app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "service": "voxify-voice-embed"}
    
    return fastapi_app


@app.function(image=image, cpu=2.0, timeout=300)
def test_embedding():
    """
    Test function to verify the embedding extraction works
    Can be called with: modal run modal_voice_embed.py::test_embedding
    """
    import librosa
    import numpy as np
    from resemblyzer import VoiceEncoder
    
    print("Testing voice embedding extraction...")
    
    # Create a simple test audio (1 second of sine wave)
    sr = 16000
    duration = 1.0
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Initialize encoder
    encoder = VoiceEncoder(device="cpu")
    
    # Extract embedding
    embedding = encoder.embed_utterance(test_audio)
    
    print(f"Test successful!")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding sample (first 10 values): {embedding[:10]}")
    
    return True

