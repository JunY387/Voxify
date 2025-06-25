#!/usr/bin/env python3
"""
F5-TTS Installation Script
Automatically installs F5-TTS and its dependencies
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description=""):
    """Run a shell command and handle errors"""
    try:
        logger.info(f"Running: {description or command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command}")
        logger.error(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        logger.error("Python 3.10 or higher is required for F5-TTS")
        return False
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_gpu_support():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            return "cuda"
        else:
            logger.info("CUDA not available, using CPU")
            return "cpu"
    except ImportError:
        logger.info("PyTorch not installed yet")
        return "unknown"

def install_pytorch():
    """Install PyTorch based on system configuration"""
    logger.info("Installing PyTorch...")
    
    system = platform.system().lower()
    gpu_support = check_gpu_support()
    
    if gpu_support == "cuda" or gpu_support == "unknown":
        # Install CUDA version
        torch_cmd = "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
    else:
        # Install CPU version
        torch_cmd = "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(torch_cmd, "Installing PyTorch")

def install_f5tts():
    """Install F5-TTS package"""
    logger.info("Installing F5-TTS...")
    
    # First install from PyPI
    if not run_command("pip install f5-tts", "Installing F5-TTS from PyPI"):
        logger.warning("PyPI installation failed, trying from source...")
        
        # Try installing from GitHub
        github_cmd = "pip install git+https://github.com/SWivid/F5-TTS.git"
        if not run_command(github_cmd, "Installing F5-TTS from GitHub"):
            logger.error("Failed to install F5-TTS from both PyPI and GitHub")
            return False
    
    return True

def install_additional_dependencies():
    """Install additional required packages"""
    logger.info("Installing additional dependencies...")
    
    packages = [
        "soundfile",
        "librosa",
        "numpy",
        "scipy",
        "transformers",
        "accelerate",
        "datasets",
        "tokenizers"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            logger.warning(f"Failed to install {package}")
    
    return True

def verify_installation():
    """Verify F5-TTS installation"""
    logger.info("Verifying F5-TTS installation...")
    
    try:
        import f5_tts
        logger.info("F5-TTS imported successfully")
        
        # Try importing key modules
        from f5_tts.infer.utils_infer import load_model
        logger.info("F5-TTS inference modules imported successfully")
        
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        return True
        
    except ImportError as e:
        logger.error(f"F5-TTS verification failed: {e}")
        return False

def create_model_directory():
    """Create directory for F5-TTS models"""
    model_dir = Path("data/f5_tts_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model directory created: {model_dir}")
    
    # Create voice clones directory
    clone_dir = Path("data/voice_clones")
    clone_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Voice clones directory created: {clone_dir}")

def main():
    """Main installation process"""
    logger.info("Starting F5-TTS installation...")
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Create necessary directories
    create_model_directory()
    
    # Install PyTorch
    if not install_pytorch():
        logger.error("Failed to install PyTorch")
        sys.exit(1)
    
    # Install F5-TTS
    if not install_f5tts():
        logger.error("Failed to install F5-TTS")
        sys.exit(1)
    
    # Install additional dependencies
    install_additional_dependencies()
    
    # Verify installation
    if verify_installation():
        logger.info("✅ F5-TTS installation completed successfully!")
    else:
        logger.error("❌ F5-TTS installation verification failed")
        sys.exit(1)
    
    logger.info("🎉 You can now use F5-TTS voice cloning features!")
    logger.info("To test the installation, run: python -c 'import f5_tts; print(\"F5-TTS is ready!\")'")

if __name__ == "__main__":
    main() 