"""
F5-TTS Service Integration
Handles voice cloning and TTS synthesis using F5-TTS (Local or Remote)
"""

import os
import uuid
import torch
import torchaudio
import requests
import base64
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import shutil
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VoiceCloneConfig:
    """Configuration for voice cloning"""

    name: str
    ref_audio_path: str
    ref_text: str
    description: Optional[str] = None
    language: str = "en-US"
    speed: float = 1.0
    clone_type: str = "upload"  # 'upload' or 'record'


@dataclass
class TTSConfig:
    """Configuration for TTS synthesis"""

    text: str
    ref_audio_path: str
    ref_text: str
    language: str = "en-US"
    speed: float = 1.0
    output_format: str = "wav"
    sample_rate: int = 24000


class F5TTSService:
    """F5-TTS service for voice cloning and synthesis (Local or Remote)"""

    def __init__(self, model_name: str = "F5TTS_v1_Base", use_remote: bool = True):
        self.model_name = model_name
        self.use_remote = use_remote
        self.remote_api_url = os.getenv(
            "F5_TTS_REMOTE_URL",
            "https://nnnnnjun-yang--f5-tts-voxify-fastapi-app.modal.run/synthesize",
        )
        self.request_timeout = int(os.getenv("F5_TTS_TIMEOUT", "120"))  # 2 minutes default

        # Local model setup (only if not using remote)
        if not self.use_remote:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = None
            self.initialized = False
            logger.info(f"F5TTS Service initialized with LOCAL device: {self.device}")
        else:
            self.device = "remote"
            self.model = None
            self.initialized = True
            logger.info(f"F5TTS Service initialized with REMOTE API: {self.remote_api_url}")

        # Storage paths
        self.base_path = Path("backend/data/voice_clones")
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _lazy_load_model(self):
        """Lazy load F5-TTS model when needed (only for local mode)"""
        if self.use_remote:
            return  # Skip model loading for remote mode

        if not self.initialized:
            try:
                # Import F5-TTS components
                logger.info("Loading F5-TTS model...")

                # Load F5-TTS model using the new API
                from f5_tts.api import F5TTS

                # Initialize F5TTS with GPU support
                self.model = F5TTS(model=self.model_name, device=str(self.device))

                # The vocoder is built into the F5TTS class
                self.vocoder = None  # Not needed with new API

                self.initialized = True
                logger.info(f"F5-TTS model loaded successfully on {self.device}")

            except ImportError as e:
                logger.error(f"Failed to import F5-TTS: {e}")
                logger.error("Please install F5-TTS with: pip install f5-tts")
                # For development, create a mock model
                self.model = None
                self.vocoder = None
                self.initialized = True
                logger.warning("Using mock F5-TTS model for development")
            except Exception as e:
                logger.error(f"Failed to load F5-TTS model: {e}")
                # For development, create a mock model
                self.model = None
                self.vocoder = None
                self.initialized = True
                logger.warning("Using mock F5-TTS model for development")

    def _synthesize_remote(self, config: TTSConfig) -> str:
        """Synthesize speech using remote F5-TTS API"""
        try:
            # Read reference audio and encode to base64
            logger.info(f"Reading reference audio: {config.ref_audio_path}")
            with open(config.ref_audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()

            # Prepare request payload
            payload = {
                "text": config.text,
                "reference_audio_b64": audio_b64,
                "reference_text": config.ref_text or "",  # Use ref_text or empty for auto-transcription
                "language": config.language,  # Pass language parameter for multilingual support
                "speed": config.speed,  # Pass speed parameter
            }

            logger.info("Sending request to remote F5-TTS API...")
            logger.info(f"Text length: {len(config.text)} characters")
            logger.info(f"Reference text: {config.ref_text[:50]}..." if config.ref_text else "Auto-transcription")

            # Make API request
            response = requests.post(self.remote_api_url, json=payload, timeout=self.request_timeout)

            logger.info(f"API response status: {response.status_code}")

            # Parse response
            if response.status_code != 200:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")

            try:
                result = response.json()
            except json.JSONDecodeError:
                raise Exception(f"Invalid JSON response: {response.text}")

            # Check if synthesis was successful
            if not result.get("success", False):
                error_msg = result.get("error") or result.get("detail") or "Unknown error"
                raise Exception(f"Remote synthesis failed: {error_msg}")

            # Decode audio data
            if "audio_data" not in result:
                raise Exception("No audio data in response")

            try:
                audio_data = base64.b64decode(result["audio_data"])
            except Exception as e:
                raise Exception(f"Failed to decode audio data: {e}")

            # Generate unique output filename
            output_id = str(uuid.uuid4())
            output_path = self.base_path / f"synthesis_{output_id}.wav"

            # Save audio file
            with open(output_path, "wb") as f:
                f.write(audio_data)

            file_size = os.path.getsize(output_path)
            logger.info(f"Remote synthesis completed: {output_path} ({file_size} bytes)")

            return str(output_path)

        except requests.exceptions.Timeout:
            raise Exception(f"Remote API request timed out after {self.request_timeout} seconds")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Remote API request failed: {e}")
        except Exception as e:
            logger.error(f"Remote synthesis error: {e}")
            raise

    def create_voice_clone(self, config: VoiceCloneConfig, sample_ids: List[str]) -> Dict:
        """
        Create a voice clone from voice samples

        Args:
            config: Voice clone configuration
            sample_ids: List of processed sample IDs

        Returns:
            Dictionary with clone information
        """
        try:
            self._lazy_load_model()

            clone_id = str(uuid.uuid4())
            clone_path = self.base_path / clone_id
            clone_path.mkdir(parents=True, exist_ok=True)

            # For F5-TTS, we don't need to train a model
            # Instead, we store the reference audio and text for later use
            clone_info = {
                "id": clone_id,
                "name": config.name,
                "description": config.description,
                "ref_audio_path": config.ref_audio_path,
                "ref_text": config.ref_text,
                "language": config.language,
                "speed": config.speed,
                "clone_type": config.clone_type,  # 'upload' or 'record'
                "sample_ids": sample_ids,
                "status": "ready",  # F5-TTS doesn't require training
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model_type": "f5_tts",
            }

            # Copy reference audio to clone directory
            ref_audio_dest = clone_path / "reference.wav"
            shutil.copy2(config.ref_audio_path, ref_audio_dest)
            clone_info["ref_audio_path"] = str(ref_audio_dest)

            # Save clone info as JSON
            with open(clone_path / "clone_info.json", "w", encoding="utf-8") as f:
                json.dump(clone_info, f, ensure_ascii=False, indent=2)

            logger.info(f"Voice clone created successfully: {clone_id}")
            return clone_info

        except Exception as e:
            logger.error(f"Failed to create voice clone: {e}")
            raise

    def synthesize_speech(self, config: TTSConfig, clone_id: Optional[str] = None) -> str:
        """
        Synthesize speech using F5-TTS

        Args:
            config: TTS configuration
            clone_id: Optional voice clone ID to use

        Returns:
            Path to generated audio file
        """
        try:
            self._lazy_load_model()

            # If clone_id is provided, load clone configuration
            if clone_id:
                clone_path = self.base_path / clone_id
                if not clone_path.exists():
                    raise ValueError(f"Voice clone not found: {clone_id}")

                # Load clone info
                with open(clone_path / "clone_info.json", "r", encoding="utf-8") as f:
                    clone_info = json.load(f)

                # Use clone's reference audio and text
                config.ref_audio_path = clone_info["ref_audio_path"]
                config.ref_text = clone_info["ref_text"]
                config.language = clone_info.get("language", config.language)
                config.speed = clone_info.get("speed", config.speed)

            # Generate unique output filename
            output_id = str(uuid.uuid4())
            output_path = self.base_path / f"synthesis_{output_id}.wav"

            logger.info("Synthesizing speech with F5-TTS...")
            logger.info(f"Mode: {'REMOTE API' if self.use_remote else 'LOCAL GPU'}")
            logger.info(f"Text: {config.text[:50]}...")
            logger.info(f"Reference audio: {config.ref_audio_path}")

            # Use remote API or local model
            if self.use_remote:
                return self._synthesize_remote(config)

            if self.model is None:
                # Mock mode for development
                logger.info("Using mock synthesis (F5-TTS not available)")

                # Create a simple mock audio file (just copy the reference audio)
                import shutil

                shutil.copy2(config.ref_audio_path, output_path)

                # Add a text file with the synthesized text for debugging
                text_path = output_path.with_suffix(".txt")
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(f"Synthesized text: {config.text}\n")
                    f.write(f"Reference text: {config.ref_text}\n")
                    f.write(f"Language: {config.language}\n")
                    f.write(f"Speed: {config.speed}\n")

                logger.info(f"Mock synthesis completed: {output_path}")

            else:
                # Real F5-TTS synthesis using new API
                try:
                    logger.info("Performing F5-TTS inference on GPU...")

                    # Use the F5TTS.infer method with the correct API
                    generated_audio = self.model.infer(
                        ref_file=config.ref_audio_path,
                        ref_text=config.ref_text,
                        gen_text=config.text,
                        speed=config.speed,
                        cross_fade_duration=0.15,
                        nfe_step=32,
                        cfg_strength=2.0,
                        sway_sampling_coef=-1.0,
                        remove_silence=True,
                    )

                    # Save generated audio
                    # F5TTS.infer returns (audio_array, sample_rate, spectrogram)
                    import torch
                    import numpy as np

                    if isinstance(generated_audio, tuple):
                        # Extract audio and sample rate from tuple
                        audio_data, sample_rate, _ = generated_audio
                    else:
                        # Fallback for unexpected format
                        audio_data = generated_audio
                        sample_rate = 24000

                    # Convert numpy array to tensor
                    if isinstance(audio_data, np.ndarray):
                        audio_data = torch.from_numpy(audio_data)
                    elif not isinstance(audio_data, torch.Tensor):
                        audio_data = torch.tensor(audio_data)

                    # Ensure correct shape for torchaudio.save (channels, samples)
                    if audio_data.dim() == 1:
                        audio_data = audio_data.unsqueeze(0)
                    elif audio_data.dim() > 2:
                        # If multiple dimensions, flatten to (channels, samples)
                        audio_data = audio_data.view(1, -1)

                    torchaudio.save(str(output_path), audio_data.cpu().float(), sample_rate)

                    logger.info(f"Real F5-TTS synthesis completed: {output_path}")

                except Exception as e:
                    logger.error(f"F5-TTS synthesis failed, falling back to mock: {e}")
                    # Fallback to mock
                    import shutil

                    shutil.copy2(config.ref_audio_path, output_path)

            logger.info(f"Speech synthesis completed: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to synthesize speech: {e}")
            raise

    def get_clone_info(self, clone_id: str) -> Dict:
        """Get information about a voice clone"""
        try:
            clone_path = self.base_path / clone_id
            if not clone_path.exists():
                # Don't log error for missing clones - this is expected during cleanup
                raise ValueError(f"Voice clone not found: {clone_id}")

            clone_info_path = clone_path / "clone_info.json"
            if not clone_info_path.exists():
                # Don't log error for missing clone info - this is expected during cleanup
                raise ValueError(f"Clone info not found: {clone_id}")

            with open(clone_info_path, "r", encoding="utf-8") as f:
                return json.load(f)

        except ValueError:
            # Re-raise ValueError without logging (expected for missing clones)
            raise
        except Exception as e:
            # Log only unexpected errors (file system errors, JSON parsing errors, etc.)
            logger.error(f"Failed to get clone info for {clone_id}: {e}")
            raise

    def list_clones(self) -> List[Dict]:
        """List all available voice clones"""
        try:
            clones = []
            for clone_dir in self.base_path.iterdir():
                if clone_dir.is_dir():
                    try:
                        clone_info = self.get_clone_info(clone_dir.name)
                        clones.append(clone_info)
                    except Exception as e:
                        logger.warning(f"Failed to load clone {clone_dir.name}: {e}")

            return sorted(clones, key=lambda x: x.get("created_at", ""), reverse=True)

        except Exception as e:
            logger.error(f"Failed to list clones: {e}")
            raise

    def delete_clone(self, clone_id: str) -> bool:
        """Delete a voice clone"""
        try:
            clone_path = self.base_path / clone_id
            if not clone_path.exists():
                return False

            shutil.rmtree(clone_path)
            logger.info(f"Voice clone deleted: {clone_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete clone: {e}")
            raise

    def validate_audio_file(self, audio_path: str) -> Tuple[bool, str]:
        """Validate audio file for F5-TTS"""
        try:
            if not os.path.exists(audio_path):
                return False, "Audio file not found"

            # Load audio and check properties
            audio, sample_rate = torchaudio.load(audio_path)
            duration = audio.shape[1] / sample_rate

            # Check duration (F5-TTS works better with 3-30 seconds)
            if duration < 3.0:
                return False, "Audio too short (minimum 3 seconds required)"
            if duration > 30.0:
                return False, "Audio too long (maximum 30 seconds recommended)"

            # Check sample rate
            if sample_rate < 16000:
                return False, "Sample rate too low (minimum 16kHz required)"

            return True, "Audio file is valid"

        except Exception as e:
            return False, f"Failed to validate audio: {str(e)}"


# Global service instance
_f5_tts_service = None


def get_f5_tts_service() -> F5TTSService:
    """Get the global F5-TTS service instance"""
    global _f5_tts_service
    if _f5_tts_service is None:
        # Check environment variable for remote/local mode
        use_remote = os.getenv("F5_TTS_USE_REMOTE", "true").lower() == "true"
        _f5_tts_service = F5TTSService(use_remote=use_remote)
    return _f5_tts_service
