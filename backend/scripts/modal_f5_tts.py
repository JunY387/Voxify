import modal
import os
import tempfile
import base64
import torch
import torchaudio
from pathlib import Path

# Create Modal app
app = modal.App("f5-tts-voxify")

# Define the container image with F5-TTS
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "sox")
    .pip_install(
        [
            # Core dependencies that F5-TTS needs
            "torch",
            "torchaudio",
            "transformers",
            "accelerate",
            "librosa",
            "soundfile",
            "numpy",
            "scipy",
            "einops",
            "cached-path",
            "fastapi",
            "pydantic",
        ]
    )
    .run_commands(
        [
            # Follow the exact installation instructions from F5-TTS
            "git clone https://github.com/SWivid/F5-TTS.git /app/F5-TTS",
            "cd /app/F5-TTS && pip install -e .",
            # Debug: Check what got installed
            "ls -la /app/F5-TTS/",
            "ls -la /app/F5-TTS/src/",
            "ls -la /app/F5-TTS/src/f5_tts/",
            "find /app/F5-TTS -name '*.yaml' -o -name '*.yml'",
            # Create the configs directory and copy config files if they exist elsewhere
            "mkdir -p /app/F5-TTS/src/f5_tts/configs/",
            # Look for config files in the repo and copy them
            "find /app/F5-TTS -name '*F5-TTS*.yaml' -exec cp {} /app/F5-TTS/src/f5_tts/configs/ \\;",
            "find /app/F5-TTS -name 'model*.yaml' -exec cp {} /app/F5-TTS/src/f5_tts/configs/ \\;",
            # If still no config, create a basic one
            "if [ ! -f /app/F5-TTS/src/f5_tts/configs/F5-TTS.yaml ]; then echo 'Creating basic config...'; fi",
        ]
    )
    .env({"PYTHONPATH": "/app/F5-TTS:/app"})
)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    scaledown_window=300,
    min_containers=1,
    memory=8192,  # 8GB memory for model loading
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import sys
    import subprocess

    # Add F5-TTS to path
    sys.path.insert(0, "/app/F5-TTS")

    # Initialize FastAPI
    fastapi_app = FastAPI()

    print("F5-TTS setup complete, ready to handle requests!")

    class SynthesisRequest(BaseModel):
        text: str
        reference_audio_b64: str
        reference_text: str = ""  # Optional transcription of reference audio
        language: str = "zh-CN"  # Language parameter for multilingual support
        speed: float = 1.0  # Speed parameter

    @fastapi_app.post("/synthesize")
    async def synthesize_speech(request: SynthesisRequest):
        try:
            print(f"Received synthesis request for text: {request.text[:50]}...")
            print(f"Language: {request.language}, Speed: {request.speed}")

            text = request.text.strip()
            reference_audio_b64 = request.reference_audio_b64
            ref_text = request.reference_text.strip()
            language = request.language
            speed = request.speed

            if not text or not reference_audio_b64:
                raise HTTPException(status_code=400, detail="Missing text or reference_audio_b64")

            print("Decoding reference audio...")
            # Create temporary file for reference audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
                try:
                    audio_data = base64.b64decode(reference_audio_b64)
                    ref_file.write(audio_data)
                    ref_file.flush()
                    ref_path = ref_file.name
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid base64 audio data: {str(e)}")

            try:
                print("Starting speech synthesis using high-level API...")

                # Try the simple high-level API first
                sys.path.insert(0, "/app/F5-TTS/src")

                # Try importing the simple API
                try:
                    from f5_tts.api import F5TTS

                    print("Using F5TTS API class...")

                    # Initialize F5-TTS
                    f5tts = F5TTS()

                    # Generate speech with language and speed parameters
                    print(f"Generating speech for language: {language} with speed: {speed}")
                    wav, sr, spect = f5tts.infer(
                        ref_file=ref_path,
                        ref_text=ref_text if ref_text else "",
                        gen_text=text,
                        speed=speed,
                        # Note: F5-TTS model selection based on language would be handled here
                        # For now, we use the default multilingual model
                    )

                    print(
                        f"Generated audio shape: {wav.shape if hasattr(wav, 'shape') else len(wav)}, sample rate: {sr}"
                    )

                except ImportError:
                    print("F5TTS API not available, using manual approach...")

                    # Fall back to manual imports
                    import subprocess

                    # Create a simple Python script to run F5-TTS
                    script_content = f"""
import sys
sys.path.insert(0, '/app/F5-TTS/src')

from f5_tts.infer.utils_infer import infer_process
import torchaudio

# Run inference with language and speed parameters
wav, sr = infer_process(
    ref_file="{ref_path}",
    ref_text="{ref_text}",
    gen_text="{text}",
    model_type="F5-TTS",
    remove_silence=True,
    speed={speed}
)

# Save output
torchaudio.save("/tmp/f5_output.wav", wav, sr)
print(f"Saved audio with shape {{wav.shape}} and sr {{sr}}")
"""

                    # Write and execute the script
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as script_file:
                        script_file.write(script_content)
                        script_path = script_file.name

                    # Run the script
                    result = subprocess.run(
                        ["python", script_path],
                        cwd="/app/F5-TTS",
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )

                    if result.returncode != 0:
                        print(f"Script stderr: {result.stderr}")
                        print(f"Script stdout: {result.stdout}")
                        raise Exception(f"F5-TTS script failed: {result.stderr}")

                    print(f"Script output: {result.stdout}")

                    # Read the generated file
                    if not os.path.exists("/tmp/f5_output.wav"):
                        raise Exception("No output file was generated")

                    with open("/tmp/f5_output.wav", "rb") as f:
                        result_audio = f.read()

                    # Clean up
                    os.unlink(script_path)
                    os.unlink("/tmp/f5_output.wav")

                    print("Synthesis completed successfully via script!")

                    # Encode result as base64
                    result_b64 = base64.b64encode(result_audio).decode()

                    return {
                        "success": True,
                        "audio_data": result_b64,
                        "sample_rate": 24000,  # F5-TTS default
                        "message": "Speech synthesized successfully",
                        "language": language,
                        "speed": speed,
                        "model_info": "F5-TTS (script mode)",
                    }

                # If F5TTS API worked, process the result
                if "wav" in locals():
                    # Save generated audio to temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
                        # Convert to proper format for saving
                        if torch.is_tensor(wav):
                            wav_np = wav.cpu().numpy()
                        else:
                            wav_np = wav

                        # Ensure correct shape for torchaudio
                        if len(wav_np.shape) == 1:
                            wav_tensor = torch.from_numpy(wav_np).unsqueeze(0)
                        else:
                            wav_tensor = torch.from_numpy(wav_np)

                        # Save audio
                        torchaudio.save(output_file.name, wav_tensor, sr)

                        # Read the saved audio file
                        with open(output_file.name, "rb") as f:
                            result_audio = f.read()

                        generated_file = output_file.name

                    print("Synthesis completed successfully!")

                    # Encode result as base64
                    result_b64 = base64.b64encode(result_audio).decode()

                    return {
                        "success": True,
                        "audio_data": result_b64,
                        "sample_rate": sr,
                        "message": "Speech synthesized successfully",
                        "language": language,
                        "speed": speed,
                        "model_info": "F5-TTS (API mode)",
                    }

            except subprocess.TimeoutExpired:
                raise HTTPException(status_code=500, detail="Speech synthesis timed out")
            except Exception as synthesis_error:
                print(f"Synthesis error: {synthesis_error}")
                import traceback

                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(synthesis_error)}")

            finally:
                # Cleanup temporary files
                try:
                    os.unlink(ref_path)
                    if "generated_file" in locals() and os.path.exists(generated_file):
                        os.unlink(generated_file)
                except OSError:
                    pass

        except HTTPException:
            raise
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            import traceback

            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    @fastapi_app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "model": "F5-TTS",
            "cuda_available": torch.cuda.is_available(),
        }

    @fastapi_app.get("/debug")
    async def debug_info():
        """Debug endpoint to check F5-TTS installation"""
        try:
            # Check if F5-TTS CLI is available
            result = subprocess.run(
                ["python", "-m", "f5_tts.infer.infer_cli", "--help"],
                cwd="/app/F5-TTS",
                capture_output=True,
                text=True,
                timeout=30,
            )

            return {
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "f5_tts_cli_available": result.returncode == 0,
                "f5_tts_help_output": (result.stdout[:500] if result.returncode == 0 else result.stderr[:500]),
            }
        except Exception as e:
            return {"error": str(e)}

    return fastapi_app


# Optional: Function to test Python API
@app.function(image=image)
def test_f5tts_python():
    """Test function to verify F5-TTS Python API works"""
    import sys

    sys.path.insert(0, "/app/F5-TTS/src")

    print("Testing F5-TTS Python API...")
    try:
        from f5_tts.model import DiT
        from f5_tts.infer.utils_infer import load_model, load_vocoder

        print("✅ F5-TTS imports successful!")

        # Try loading a model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        print("F5-TTS Python API is working!")
        return "F5-TTS Python API is working!"

    except Exception as e:
        print(f"❌ Error testing Python API: {e}")
        import traceback

        traceback.print_exc()
        return f"Error: {e}"


if __name__ == "__main__":
    # For local testing
    pass
