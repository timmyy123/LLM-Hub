#!/usr/bin/env python3
"""
Download pre-converted MediaPipe models for Android LLM app
"""
import os
import requests
from pathlib import Path

# Available pre-converted models for MediaPipe
MODELS = {
    "gemma-2-2b": {
        "url": "https://www.kaggle.com/models/google/gemma-2/tfLite/gemma-2-2b-it-gpu-int8",
        "filename": "gemma-2-2b-it-gpu-int8.bin",
        "size": "~2GB",
        "description": "Gemma-2 2B model, good balance of size and performance"
    },
    "gemma-3-1b": {
        "url": "https://www.kaggle.com/models/google/gemma-3/tfLite/gemma-3-1b-it-int8",
        "filename": "gemma-3-1b-it-int8.task", 
        "size": "~1GB",
        "description": "Gemma-3 1B model, smallest and fastest"
    }
}

def download_model(model_name, download_dir="models"):
    """Download a pre-converted MediaPipe model"""
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {list(MODELS.keys())}")
        return False
    
    model_info = MODELS[model_name]
    download_path = Path(download_dir)
    download_path.mkdir(exist_ok=True)
    
    print(f"Model: {model_name}")
    print(f"Description: {model_info['description']}")
    print(f"Size: {model_info['size']}")
    print(f"Kaggle URL: {model_info['url']}")
    print(f"Filename: {model_info['filename']}")
    print("\nTo download:")
    print("1. Go to the Kaggle URL above")
    print("2. Click 'Download' button")
    print(f"3. Save as: {download_path / model_info['filename']}")
    
    return True

def list_models():
    """List all available pre-converted models"""
    print("Available MediaPipe Models:")
    print("=" * 50)
    for name, info in MODELS.items():
        print(f"• {name}")
        print(f"  Description: {info['description']}")
        print(f"  Size: {info['size']}")
        print(f"  Format: {info['filename'].split('.')[-1]}")
        print()

def main():
    print("MediaPipe Model Downloader")
    print("=" * 30)
    
    # List available models
    list_models()
    
    # Provide specific recommendations
    print("RECOMMENDATIONS:")
    print("• For best performance: gemma-2-2b")
    print("• For smallest size: gemma-3-1b")
    print()
    
    # Show download instructions
    model_choice = "gemma-2-2b"  # Default recommendation
    print(f"Downloading instructions for {model_choice}:")
    download_model(model_choice)

if __name__ == "__main__":
    main() 