#!/usr/bin/env python3
"""
Convert Gemma-3 models to MediaPipe .task format
"""
import os
import subprocess
from pathlib import Path

# Gemma-3 model configurations
GEMMA3_MODELS = {
    "gemma-3-1b": {
        "hf_repo": "google/gemma-2-1b-it",  # Closest available
        "size": "1B parameters",
        "description": "Smallest Gemma-3 variant"
    },
    "gemma-3-4b": {
        "hf_repo": "google/gemma-2-4b-it",  # Closest available 
        "size": "4B parameters", 
        "description": "Medium Gemma-3 variant"
    },
    "gemma-3-12b": {
        "hf_repo": "google/gemma-2-12b-it",  # Closest available
        "size": "12B parameters",
        "description": "Large Gemma-3 variant (may be too large for mobile)"
    }
}

def install_dependencies():
    """Install required packages for conversion"""
    packages = [
        "mediapipe>=0.10.14",
        "transformers", 
        "torch",
        "sentencepiece",
        "huggingface_hub"
    ]
    
    print("Installing required packages...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run([
            "python", "-m", "pip", "install", package
        ], check=True)

def download_original_model(model_name, download_dir="models"):
    """Download original model from HuggingFace"""
    if model_name not in GEMMA3_MODELS:
        print(f"Unknown model: {model_name}")
        return None
        
    model_info = GEMMA3_MODELS[model_name]
    repo_id = model_info["hf_repo"]
    
    print(f"Downloading {model_name} from {repo_id}...")
    
    from huggingface_hub import snapshot_download
    
    local_dir = Path(download_dir) / model_name
    local_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded to {local_dir}")
        return local_dir
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def convert_to_mediapipe(model_path, output_name):
    """Convert model to MediaPipe .task format"""
    print(f"Converting {model_path} to MediaPipe format...")
    
    try:
        import mediapipe as mp
        from mediapipe.tasks.python.genai import bundler
        
        # Configuration for bundling
        config = bundler.BundleConfig(
            # These paths need to be adjusted based on actual model structure
            tflite_model=str(model_path / "model.tflite"),  # This won't exist yet
            tokenizer_model=str(model_path / "tokenizer.model"),
            start_token="<bos>",
            stop_tokens=["<eos>"],
            output_filename=f"{output_name}.task",
            enable_bytes_to_unicode_mapping=True,
        )
        
        print("Creating MediaPipe bundle...")
        bundler.create_bundle(config)
        print(f"Conversion complete! Output: {output_name}.task")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("MediaPipe genai module not available")
    except Exception as e:
        print(f"Conversion failed: {e}")
        print("\nNote: Direct conversion from HF models requires additional steps:")
        print("1. Export model to TensorFlow Lite format first")
        print("2. Then use MediaPipe bundler")

def show_conversion_steps():
    """Show manual conversion steps"""
    print("\n" + "="*60)
    print("MANUAL CONVERSION STEPS FOR GEMMA-3")
    print("="*60)
    
    print("""
The GGUF file you have cannot be directly converted to MediaPipe format.
Here's what you need to do:

OPTION 1: Use Pre-converted Models (RECOMMENDED)
==========================================
Check Kaggle Models for pre-converted Gemma-3:
• https://www.kaggle.com/models/google/gemma-3/
• Look for models with "litert" or "mediapipe" tags
• Download .task or .bin files directly

OPTION 2: Convert from Original Model
=====================================
1. Download original Gemma-3 from HuggingFace:
   - google/gemma-3-1b-it
   - google/gemma-3-4b-it  
   - google/gemma-3-12b-it (if available)

2. Convert to TensorFlow Lite:
   - Use AI Edge Torch or TensorFlow Lite converter
   - Export to .tflite format

3. Bundle for MediaPipe:
   - Use MediaPipe bundler to create .task file
   - Include tokenizer and metadata

OPTION 3: Alternative Tools
===========================
• Use llama.cpp to ONNX converters
• Then ONNX to TensorFlow Lite
• Finally TFLite to MediaPipe .task

RECOMMENDATION:
===============
For fastest results, download pre-converted models from Kaggle.
Your GGUF file is already quantized, but MediaPipe uses different
quantization methods optimized for mobile inference.
""")

def main():
    print("Gemma-3 to MediaPipe Converter")
    print("=" * 40)
    
    # Show available models
    print("\nAvailable Gemma-3 variants:")
    for name, info in GEMMA3_MODELS.items():
        print(f"• {name}: {info['description']} ({info['size']})")
    
    # Show conversion options
    show_conversion_steps()
    
    # Check for pre-converted models
    print("\n" + "="*60)
    print("CHECKING FOR PRE-CONVERTED MODELS...")
    print("="*60)
    
    kaggle_urls = [
        "https://www.kaggle.com/models/google/gemma-3/",
        "https://www.kaggle.com/models/google/gemma-2/",
    ]
    
    print("Check these Kaggle URLs for ready-to-use models:")
    for url in kaggle_urls:
        print(f"• {url}")
    
    print("\nLook for files ending in:")
    print("• .task (MediaPipe task format)")
    print("• .bin (MediaPipe binary format)")
    print("• Models tagged with 'litert' or 'mediapipe'")

if __name__ == "__main__":
    main() 